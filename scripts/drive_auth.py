#!/usr/bin/env python3
"""Mint a Google Drive token for the training checkpoint store.

Run this **once, on your own machine** — it opens a browser. It produces a
single `drive_token.json` that you upload to Kaggle as a private dataset.

    pip install google-auth-oauthlib google-api-python-client
    python scripts/drive_auth.py --client-secret client_secret.json

Why OAuth rather than a service account
---------------------------------------
A service account has no Drive storage of its own, and a file it creates is
owned by *it*, not by you. Sharing one of your folders with it does not help:
the upload still fails with

    Service Accounts do not have storage quota.

Google's own suggested workarounds — Shared Drives and domain-wide delegation
— both require Google Workspace, so neither is available to a personal
@gmail.com account. Authorising as yourself is the path that works: the files
are owned by you and count against your own quota.

Scope
-----
Only `drive.file` is requested: per-file access to files this app creates. It
cannot see anything else in your Drive. That matters twice over — it is the
least access that does the job, and because `drive.file` is not a "sensitive"
scope, the OAuth app needs no verification review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
DEFAULT_FOLDER_NAME = "ai-text-humanizer-checkpoints"


def authorise(client_secret: Path):
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(client_secret), SCOPES)
    # Opens a browser and catches the redirect on a local port. Google retired
    # the copy-a-code-from-the-page flow in 2022, so this has to run somewhere
    # with a browser — which is why it is a local script and not a notebook
    # cell.
    return flow.run_local_server(port=0, prompt="consent")


def find_or_create_folder(service, name: str) -> str:
    """Return the id of our folder, creating it if needed.

    The folder is created through the API so that it counts as app-created and
    is therefore reachable under `drive.file`. A folder you made by hand in the
    Drive web UI is *not* visible to this scope, which would fail confusingly
    later.
    """
    query = (
        "mimeType = 'application/vnd.google-apps.folder' "
        f"and name = '{name}' and trashed = false"
    )
    found = service.files().list(q=query, fields="files(id, name)", pageSize=1).execute()
    files = found.get("files", [])
    if files:
        print(f"Using existing folder '{name}' ({files[0]['id']})")
        return files[0]["id"]

    created = (
        service.files()
        .create(
            body={"name": name, "mimeType": "application/vnd.google-apps.folder"},
            fields="id",
        )
        .execute()
    )
    print(f"Created folder '{name}' ({created['id']})")
    return created["id"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--client-secret",
        type=Path,
        default=Path("client_secret.json"),
        help="OAuth client JSON downloaded from Google Cloud Console",
    )
    parser.add_argument("--folder-name", default=DEFAULT_FOLDER_NAME)
    parser.add_argument("--output", type=Path, default=Path("drive_token.json"))
    args = parser.parse_args()

    if not args.client_secret.is_file():
        print(f"{args.client_secret} not found.\n", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        return 1

    from googleapiclient.discovery import build

    credentials = authorise(args.client_secret)
    if not credentials.refresh_token:
        print(
            "Google did not return a refresh token. Revoke this app's access at "
            "https://myaccount.google.com/permissions and run this again.",
            file=sys.stderr,
        )
        return 1

    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    folder_id = find_or_create_folder(service, args.folder_name)

    args.output.write_text(
        json.dumps(
            {
                "type": "oauth_user",
                "client_id": credentials.client_id,
                "client_secret": credentials.client_secret,
                "refresh_token": credentials.refresh_token,
                "folder_id": folder_id,
                "scopes": SCOPES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nWrote {args.output}")
    print(f"  folder id: {folder_id}")
    print(
        "\nThis file lets anything holding it write to that Drive folder. "
        "Upload it to Kaggle as a PRIVATE dataset, and do not commit it."
    )
    print(
        "\nOne more thing: in the Cloud Console OAuth consent screen, set the "
        "publishing status to 'In production'. While it is 'Testing', Google "
        "expires the refresh token after 7 days and training would stop being "
        "able to save. Because drive.file is not a sensitive scope, going to "
        "production needs no verification review."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
