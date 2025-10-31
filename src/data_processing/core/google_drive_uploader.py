import os
import json
import mimetypes
import logging
from typing import Optional

# Optional Google imports (graceful fallback if not installed locally)
try:
    from google.oauth2 import service_account  # type: ignore
    from googleapiclient.discovery import build  # type: ignore
    from googleapiclient.http import MediaFileUpload  # type: ignore
    _GDRIVE_IMPORTED = True
except Exception:
    service_account = None  # type: ignore
    build = None  # type: ignore
    MediaFileUpload = None  # type: ignore
    _GDRIVE_IMPORTED = False

logger = logging.getLogger(__name__)


SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def _load_service_account_credentials() -> Optional[object]:
    """Load service account credentials from env.

    Supported envs:
    - GDRIVE_SERVICE_ACCOUNT_JSON: path to JSON file or raw JSON string
    """
    if not _GDRIVE_IMPORTED:
        logger.debug("Google Drive libraries not installed; uploads disabled")
        return None

    raw = os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON", "").strip()
    if not raw:
        logger.debug("GDRIVE_SERVICE_ACCOUNT_JSON not set; Drive uploads disabled")
        return None

    try:
        if os.path.exists(raw):
            creds = service_account.Credentials.from_service_account_file(raw, scopes=SCOPES)  # type: ignore
        else:
            info = json.loads(raw)
            creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)  # type: ignore
        return creds
    except Exception as e:
        logger.error(f"Failed to load Google Drive credentials: {e}")
        return None


_drive_service = None


def _get_drive_service():
    global _drive_service
    if _drive_service is not None:
        return _drive_service
    if not _GDRIVE_IMPORTED:
        logger.debug("Google Drive libraries not installed; uploads disabled")
        return None
    creds = _load_service_account_credentials()
    if not creds:
        return None
    try:
        _drive_service = build("drive", "v3", credentials=creds, cache_discovery=False)  # type: ignore
        return _drive_service
    except Exception as e:
        logger.error(f"Failed to build Google Drive service: {e}")
        return None


def _ensure_subfolder(parent_id: str, name: str) -> Optional[str]:
    service = _get_drive_service()
    if not service:
        return None
    try:
        q = (
            f"name = '{name.replace("'", "\\'")}' and "
            f"mimeType = 'application/vnd.google-apps.folder' and "
            f"'{parent_id}' in parents and trashed = false"
        )
        res = service.files().list(q=q, fields="files(id, name)").execute()
        files = res.get("files", [])
        if files:
            return files[0]["id"]
        meta = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        created = service.files().create(body=meta, fields="id").execute()
        return created.get("id")
    except Exception as e:
        logger.error(f"Failed to ensure subfolder '{name}': {e}")
        return None


def _compute_target_parent(parent_id: str) -> Optional[str]:
    """Optionally create date-based subfolders if enabled.

    Env: GDRIVE_SUBFOLDER_BY_DATE=1 creates YYYY/MM/DD under parent.
    """
    service = _get_drive_service()
    if not service:
        return None
    if os.getenv("GDRIVE_SUBFOLDER_BY_DATE", "0").strip() not in {"1", "true", "True"}:
        return parent_id
    try:
        from datetime import datetime
        today = datetime.utcnow()
        year_id = _ensure_subfolder(parent_id, str(today.year))
        if not year_id:
            return parent_id
        month_id = _ensure_subfolder(year_id, f"{today.month:02d}")
        if not month_id:
            return year_id
        day_id = _ensure_subfolder(month_id, f"{today.day:02d}")
        return day_id or month_id
    except Exception as e:
        logger.error(f"Failed to compute date-based subfolder: {e}")
        return parent_id


def upload_file_to_drive(file_path: str, parent_folder_id: Optional[str] = None, mime_type: Optional[str] = None) -> Optional[str]:
    """Upload a local file to Google Drive.

    Returns the created file ID or None on failure.
    Controlled by env GDRIVE_ENABLED=1 and GDRIVE_PARENT_FOLDER_ID when parent_folder_id not passed.
    """
    enabled = os.getenv("GDRIVE_ENABLED", "0").strip() in {"1", "true", "True"}
    if not enabled:
        logger.debug("GDrive upload skipped (GDRIVE_ENABLED not set)")
        return None

    if not os.path.exists(file_path):
        logger.warning(f"GDrive upload skipped; file does not exist: {file_path}")
        return None

    service = _get_drive_service()
    if not service:
        return None

    parent_id = parent_folder_id or os.getenv("GDRIVE_PARENT_FOLDER_ID", "").strip()
    if not parent_id:
        logger.error("GDrive upload enabled but GDRIVE_PARENT_FOLDER_ID is not set")
        return None

    parent_id = _compute_target_parent(parent_id) or parent_id

    try:
        name = os.path.basename(file_path)
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(name)
        media = MediaFileUpload(file_path, mimetype=mime_type or "application/octet-stream", resumable=True)
        body = {"name": name, "parents": [parent_id]}
        created = service.files().create(body=body, media_body=media, fields="id").execute()
        file_id = created.get("id")
        logger.info(f"Uploaded to Google Drive: {name} -> {file_id}")
        return file_id
    except Exception as e:
        logger.error(f"Google Drive upload failed for {file_path}: {e}")
        return None


