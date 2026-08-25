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


SERVICE_ACCOUNT_QUOTA_HELP = """
Google refused the upload: service accounts have no Drive storage of their own.

A file created by a service account is owned by that service account, not by
you, so your personal quota never comes into it — and sharing one of your
folders with the service account does not change ownership. Google's suggested
workarounds, Shared Drives and domain-wide delegation, both require Google
Workspace and are unavailable to a personal @gmail.com account.

Authorise as yourself instead. Run this once on a machine with a browser:

    pip install google-auth-oauthlib google-api-python-client
    python scripts/drive_auth.py --client-secret client_secret.json

That writes drive_token.json. Upload it to Kaggle as a private dataset and
point DRIVE_KEY_PATH at it. Files will then be owned by you and counted
against your own storage.
"""


class GoogleDriveStore(FileStore):
    """Google Drive.

    Uploads replace the file in place rather than adding a revision, so
    repeatedly syncing a large checkpoint does not accumulate storage.

    Takes either credential file, and works out which from its contents:

    * **OAuth user credentials** (``drive_token.json`` from
      ``scripts/drive_auth.py``) — the one that works for a personal Google
      account. Files are owned by you and count against your quota.
    * **A service-account key** — only usable against a Google Workspace
      Shared Drive. Against a personal My Drive folder every upload fails with
      ``storageQuotaExceeded`` no matter how the folder is shared, because the
      service account would own the file and has no quota.
    """

    def __init__(self, folder_id: str, credentials_json: str | Path) -> None:
        self.folder_id = folder_id
        self._key = str(credentials_json)
        self._service = None
        self._warned_quota = False

    def _credentials(self):
        import json as _json

        with open(self._key, encoding="utf-8") as handle:
            data = _json.load(handle)

        if data.get("type") == "service_account":
            from google.oauth2 import service_account

            return service_account.Credentials.from_service_account_file(
                self._key, scopes=["https://www.googleapis.com/auth/drive"]
            ), True

        if not data.get("refresh_token"):
            raise RuntimeError(
                f"{self._key} is neither a service-account key nor OAuth user "
                f"credentials. Regenerate it with scripts/drive_auth.py."
            )

        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        credentials = Credentials(
            token=None,
            refresh_token=data["refresh_token"],
            client_id=data["client_id"],
            client_secret=data["client_secret"],
            token_uri="https://oauth2.googleapis.com/token",
            scopes=data.get("scopes") or ["https://www.googleapis.com/auth/drive.file"],
        )
        # Refresh eagerly so an expired grant is reported here, with a usable
        # message, rather than inside the first checkpoint upload.
        credentials.refresh(Request())
        return credentials, False

    def _client(self):
        if self._service is not None:
            return self._service
        from googleapiclient.discovery import build

        credentials, is_service_account = self._credentials()
        self._is_service_account = is_service_account
        self._service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        return self._service

    def _explain(self, exc: Exception) -> None:
        """Turn Google's generic 403 into the actual cause, once."""
        text = str(exc)
        if "storageQuota" in text or "storage quota" in text:
            if not self._warned_quota:
                print(SERVICE_ACCOUNT_QUOTA_HELP, flush=True)
                self._warned_quota = True
        elif "File not found" in text or "notFound" in text:
            print(
                f"  [drive] folder {self.folder_id} is not visible to these "
                f"credentials. With the drive.file scope only folders this app "
                f"created are reachable — use the folder id that "
                f"scripts/drive_auth.py printed, not one made in the web UI.",
                flush=True,
            )

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
            self._explain(exc)
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
            self._explain(exc)
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
            self._explain(exc)
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

        DRIVE_SERVICE_ACCOUNT_JSON     path to drive_token.json
        DRIVE_FOLDER_ID                optional; read from the token otherwise

    `CHECKPOINT_DIR` selects `LocalDirStore` instead, which exists for tests
    and offline development rather than for real runs.
    """
    folder = os.environ.get("DRIVE_FOLDER_ID")
    key = os.environ.get("DRIVE_SERVICE_ACCOUNT_JSON")

    # drive_auth.py records the folder it created, so the notebook only has to
    # supply one value.
    if key and not folder and Path(key).is_file():
        try:
            import json as _json

            with open(key, encoding="utf-8") as handle:
                folder = _json.load(handle).get("folder_id")
            if folder:
                print(f"using folder {folder} from {Path(key).name}")
        except Exception:
            folder = None

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
