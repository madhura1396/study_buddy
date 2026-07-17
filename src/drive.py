"""Browse and import study documents from Google Drive into DATA_DIR.

Supports both uploaded files (.docx/.txt/.pdf) and native Google Docs, which
have no downloadable binary of their own and must be exported via the API
(here, to .docx) before src/ingest.py's readers can handle them.
"""

from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from src.config import (
    DATA_DIR,
    GOOGLE_CLIENT_SECRET_PATH,
    GOOGLE_DRIVE_SCOPES,
    GOOGLE_TOKEN_PATH,
)

GOOGLE_DOC_MIME_TYPE = "application/vnd.google-apps.document"
GOOGLE_DOC_EXPORT_MIME_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
# No PDF here: src/ingest.py's readers only cover .txt/.docx (PDF support was
# dropped earlier since the project has no PDF study materials) — listing PDFs
# here would let them be "imported" and then silently skipped at ingestion.
SUPPORTED_MIME_TYPES = [
    GOOGLE_DOC_MIME_TYPE,
    GOOGLE_DOC_EXPORT_MIME_TYPE,
    "text/plain",
]


class MissingCredentialsError(RuntimeError):
    pass


def get_drive_service():
    """Authenticate via OAuth, caching the token in GOOGLE_TOKEN_PATH so this
    only requires a browser consent flow the first time (or after the cached
    token expires and can't be silently refreshed)."""
    if not GOOGLE_CLIENT_SECRET_PATH.exists():
        raise MissingCredentialsError(
            f"No client_secret.json found at {GOOGLE_CLIENT_SECRET_PATH}. "
            "Download OAuth credentials from your Google Cloud project's "
            "Drive API setup and place them there."
        )

    creds = None
    if GOOGLE_TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(GOOGLE_TOKEN_PATH), GOOGLE_DRIVE_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(GOOGLE_CLIENT_SECRET_PATH), GOOGLE_DRIVE_SCOPES
            )
            creds = flow.run_local_server(port=0)
        GOOGLE_TOKEN_PATH.write_text(creds.to_json())

    return build("drive", "v3", credentials=creds)


def list_importable_files(query: str = "") -> list[dict]:
    """Return [{id, name, mimeType}, ...] for Drive files and folders matching
    a name substring. Folders are included so a search result can be a whole
    folder of study docs, not just individual files; list_folder_contents()
    expands a folder into its importable files at import time."""
    service = get_drive_service()
    mime_filter = " or ".join(f"mimeType='{m}'" for m in [*SUPPORTED_MIME_TYPES, FOLDER_MIME_TYPE])
    q = f"({mime_filter}) and trashed=false"
    if query:
        q += f" and name contains '{query}'"

    results = (
        service.files()
        .list(q=q, fields="files(id, name, mimeType)", pageSize=50)
        .execute()
    )
    return results.get("files", [])


def list_folder_contents(folder_id: str) -> list[dict]:
    """Return the importable files (recursing into subfolders) inside a Drive folder."""
    service = get_drive_service()
    results = (
        service.files()
        .list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="files(id, name, mimeType)",
            pageSize=200,
        )
        .execute()
    )
    entries = results.get("files", [])

    files = [f for f in entries if f["mimeType"] in SUPPORTED_MIME_TYPES]
    for subfolder in [f for f in entries if f["mimeType"] == FOLDER_MIME_TYPE]:
        files.extend(list_folder_contents(subfolder["id"]))
    return files


def resolve_to_files(entries: list[dict]) -> list[tuple[dict, Optional[str]]]:
    """Expand any folders in a selection into their contained files, pairing
    each resulting file with a category: the Drive folder's name if it came
    from one, or None if it was selected directly (caller decides the
    category for plain file selections). Used right before download so
    folder selections "just work" the same as file selections, while
    preserving the folder's name as the local category."""
    files: list[tuple[dict, Optional[str]]] = []
    for entry in entries:
        if entry["mimeType"] == FOLDER_MIME_TYPE:
            files.extend((f, entry["name"]) for f in list_folder_contents(entry["id"]))
        else:
            files.append((entry, None))
    return files


def _dest_filename(name: str, mime_type: str) -> str:
    if mime_type == GOOGLE_DOC_MIME_TYPE:
        return name if name.lower().endswith(".docx") else f"{name}.docx"
    return name


def download_file(file: dict, dest_dir: Path = DATA_DIR) -> Path:
    """Download (or export, for native Google Docs) a Drive file into dest_dir."""
    service = get_drive_service()
    dest_path = dest_dir / _dest_filename(file["name"], file["mimeType"])

    if file["mimeType"] == GOOGLE_DOC_MIME_TYPE:
        request = service.files().export_media(
            fileId=file["id"], mimeType=GOOGLE_DOC_EXPORT_MIME_TYPE
        )
    else:
        request = service.files().get_media(fileId=file["id"])

    with open(dest_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    return dest_path
