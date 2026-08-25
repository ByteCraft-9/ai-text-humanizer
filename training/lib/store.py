"""Off-machine storage for checkpoints and datasets.

Kaggle wipes `/kaggle/working` when a session dies, which turns an
eight-hour training run into nothing. Everything that would be expensive to
recompute therefore gets pushed somewhere that outlives the container, and
pulled back automatically on the next run.

Google Drive is the one place checkpoints live. Uploads replace the file in
place, so syncing the same 2 GB checkpoint forty times costs 2 GB rather than
accumulating a copy per sync — which is why a git-backed store like the
Hugging Face Hub is the wrong tool for this particular job, and is not offered
here. Stage 4 still publishes the *finished* models to the Hub; that is a
handful of files written once.

Every operation is best-effort. A store that cannot reach its backend logs the
problem and returns False; training continues against local disk, because a
sync failure must never be the thing that ends a run.
"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path


class FileStore(ABC):
    """A flat namespace of named files that outlives the container."""

    @abstractmethod
    def push(self, local: Path, name: str) -> bool:
        """Upload `local`, replacing anything already stored under `name`."""

    @abstractmethod
    def pull(self, name: str, local: Path) -> bool:
        """Download `name` to `local`. False when it is not there."""

    @abstractmethod
    def exists(self, name: str) -> bool: ...

    @abstractmethod
    def describe(self) -> str: ...

    def pull_if_missing(self, name: str, local: Path) -> bool:
        """Fetch only when the local copy is absent. Returns True if present."""
        if local.is_file() and local.stat().st_size > 0:
            return True
        return self.pull(name, local)


class NullStore(FileStore):
    """No remote. Training still checkpoints to local disk."""

    def push(self, local: Path, name: str) -> bool:
        return False

    def pull(self, name: str, local: Path) -> bool:
        return False

    def exists(self, name: str) -> bool:
        return False

    def describe(self) -> str:
        return (
            "no remote store — checkpoints are local only and will NOT "
            "survive the session ending"
        )


class LocalDirStore(FileStore):
    """A plain directory. The test double for GoogleDriveStore, and what a
    local run uses. Not intended for Kaggle, where local disk is exactly the
    thing that does not survive."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.root / name

    def push(self, local: Path, name: str) -> bool:
        target = self._path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write beside the target and move, so an interrupted copy never
        # leaves a truncated checkpoint that loads and then misbehaves.
        staging = target.with_suffix(target.suffix + ".partial")
        shutil.copy2(local, staging)
        staging.replace(target)
        return True

    def pull(self, name: str, local: Path) -> bool:
        source = self._path(name)
        if not source.is_file():
            return False
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, local)
        return True

    def exists(self, name: str) -> bool:
        return self._path(name).is_file()

    def describe(self) -> str:
        return f"local directory {self.root}"


class GoogleDriveStore(FileStore):
    """Google Drive, via a service account.

    Uploads replace the file in place rather than adding a revision, so
    repeatedly syncing a large checkpoint does not accumulate storage.

    Setting this up, once:

    1. Google Cloud Console → create a project → enable the **Drive API**.
    2. Create a **service account**, then create a **JSON key** for it.
    3. In Google Drive, make a folder and **share it with the service
       account's email** (the `client_email` in the JSON) as **Editor**.
       This step is what people miss: a service account has its own Drive
       with zero quota, so it can only write into a folder you have shared
       with it, where the bytes count against *your* quota.
    4. Take the folder id from its URL:
       `drive.google.com/drive/folders/<THIS_PART>`.
    5. Upload the JSON key to Kaggle as a **private Dataset**, or paste it
       into Kaggle **Add-ons → Secrets**.
    """

    def __init__(self, folder_id: str, service_account_json: str | Path) -> None:
        self.folder_id = folder_id
        self._key = str(service_account_json)
        self._service = None

    def _client(self):
        if self._service is not None:
            return self._service
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        credentials = service_account.Credentials.from_service_account_file(
            self._key, scopes=["https://www.googleapis.com/auth/drive"]
        )
        self._service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        return self._service

    def _find(self, name: str) -> str | None:
        escaped = name.replace("'", "\\'")
        query = (
            f"name = '{escaped}' and '{self.folder_id}' in parents and trashed = false"
        )
        try:
            result = (
                self._client()
                .files()
                .list(q=query, fields="files(id, size)", pageSize=1, supportsAllDrives=True)
                .execute()
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [drive] lookup failed for {name}: {exc}", flush=True)
            return None
        files = result.get("files", [])
        return files[0]["id"] if files else None

    def push(self, local: Path, name: str) -> bool:
        from googleapiclient.http import MediaFileUpload

        try:
            media = MediaFileUpload(
                str(local), resumable=True, chunksize=16 * 1024 * 1024
            )
            existing = self._find(name)
            if existing:
                request = self._client().files().update(
                    fileId=existing, media_body=media, supportsAllDrives=True
                )
            else:
                request = self._client().files().create(
                    body={"name": name, "parents": [self.folder_id]},
                    media_body=media,
                    fields="id",
                    supportsAllDrives=True,
                )

            response = None
            while response is None:
                _, response = request.next_chunk()
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"  [drive] upload of {name} failed: {exc}", flush=True)
            return False

    def pull(self, name: str, local: Path) -> bool:
        import io

        from googleapiclient.http import MediaIoBaseDownload

        file_id = self._find(name)
        if not file_id:
            return False

        local.parent.mkdir(parents=True, exist_ok=True)
        staging = local.with_suffix(local.suffix + ".partial")
        try:
            request = self._client().files().get_media(fileId=file_id, supportsAllDrives=True)
            with io.FileIO(staging, "wb") as handle:
                downloader = MediaIoBaseDownload(handle, request, chunksize=16 * 1024 * 1024)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            staging.replace(local)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"  [drive] download of {name} failed: {exc}", flush=True)
            staging.unlink(missing_ok=True)
            return False

    def exists(self, name: str) -> bool:
        return self._find(name) is not None

    def describe(self) -> str:
        return f"Google Drive folder {self.folder_id}"


# ---------------------------------------------------------------------------
# Construction from the environment
# ---------------------------------------------------------------------------


def build_store(verbose: bool = True) -> FileStore:
    """Build the Drive store from the environment.

        DRIVE_FOLDER_ID                the folder shared with the service account
        DRIVE_SERVICE_ACCOUNT_JSON     path to the service-account key

    `CHECKPOINT_DIR` selects `LocalDirStore` instead, which exists for tests
    and offline development rather than for real runs.
    """
    folder = os.environ.get("DRIVE_FOLDER_ID")
    key = os.environ.get("DRIVE_SERVICE_ACCOUNT_JSON")

    if folder and key:
        if not Path(key).is_file():
            print(
                f"DRIVE_SERVICE_ACCOUNT_JSON points at {key}, which does not "
                f"exist. Checkpoints will not leave this machine."
            )
            store: FileStore = NullStore()
        else:
            store = GoogleDriveStore(folder, key)
    elif os.environ.get("CHECKPOINT_DIR"):
        store = LocalDirStore(os.environ["CHECKPOINT_DIR"])
    else:
        store = NullStore()

    if verbose:
        print(f"checkpoint store: {store.describe()}")
    return store
