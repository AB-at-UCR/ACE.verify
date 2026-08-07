"""Static media storage and URL helpers for the Streamlit upload card.

Preset clips live in ``frontend/static/``. User uploads are written under
``frontend/static/uploads/`` with sanitized, collision-safe names so Streamlit's
static middleware can serve them at ``app/static/...``.
"""

from __future__ import annotations

import os
import re
import uuid
import mimetypes
from pathlib import Path

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".jpg", ".jpeg", ".png"}
_SAFE_STEM = re.compile(r"[^A-Za-z0-9._-]+")
_VIDEO_STATIC_EXTS = (".mp4", ".mov", ".avi", ".webm")
_MIME_PATCHED = False


def ensure_static_video_mime() -> None:
    """Allow Streamlit's static handler to serve videos with real MIME types.

    Streamlit ≤1.50 forces non-allowlisted extensions (including ``.mp4``) to
    ``Content-Type: text/plain`` with ``X-Content-Type-Options: nosniff``, which
    prevents ``<video>`` playback. Extending the allowlist + registering MIME
    types keeps same-origin ``app/static/...`` delivery working for media.
    """
    global _MIME_PATCHED
    if _MIME_PATCHED:
        return
    mimetypes.add_type("video/mp4", ".mp4")
    mimetypes.add_type("video/quicktime", ".mov")
    mimetypes.add_type("video/x-msvideo", ".avi")
    mimetypes.add_type("video/webm", ".webm")
    try:
        from streamlit.web.server import app_static_file_handler as handler
        current = tuple(handler.SAFE_APP_STATIC_FILE_EXTENSIONS)
        missing = tuple(ext for ext in _VIDEO_STATIC_EXTS if ext not in current)
        if missing:
            handler.SAFE_APP_STATIC_FILE_EXTENSIONS = current + missing
    except Exception:
        # Older/newer Streamlit layouts may not expose this symbol; static
        # serving still works for images, and preview falls back to blob.
        pass
    _MIME_PATCHED = True


def static_dir(root_dir: str | Path) -> Path:
    return Path(root_dir).resolve() / "frontend" / "static"


def uploads_dir(root_dir: str | Path) -> Path:
    path = static_dir(root_dir) / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(filename: str) -> str:
    """Return a basename-only, extension-allowlisted filename.

    Rejects path traversal (``../``, absolute paths, nested separators) by
    keeping only the final path component and stripping unsafe characters.
    """
    if not filename or not str(filename).strip():
        raise ValueError("Empty filename")

    # Normalize separators, then drop any directory components.
    name = str(filename).replace("\\", "/").split("/")[-1]
    name = name.replace("\x00", "").strip().lstrip(".")
    if not name or name in {".", ".."}:
        raise ValueError(f"Invalid filename: {filename!r}")

    stem, ext = os.path.splitext(name)
    ext = ext.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {ext or '(none)'}")

    stem = _SAFE_STEM.sub("_", stem).strip("._") or "upload"
    stem = stem[:80]
    return f"{stem}{ext}"


def unique_upload_name(filename: str) -> str:
    """Sanitize *filename* and prefix a short UUID to avoid collisions."""
    return f"{uuid.uuid4().hex[:12]}_{sanitize_filename(filename)}"


def is_under_directory(path: str | Path, directory: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(directory).resolve())
        return True
    except (ValueError, OSError):
        return False


def is_upload_path(path: str | Path | None, root_dir: str | Path) -> bool:
    if not path:
        return False
    return is_under_directory(path, uploads_dir(root_dir))


def static_url_for(path: str | Path, root_dir: str | Path) -> str | None:
    """Map a filesystem path under ``frontend/static`` to an ``app/static/...`` URL."""
    try:
        rel = Path(path).resolve().relative_to(static_dir(root_dir))
    except (ValueError, OSError):
        return None
    # Reject anything that somehow escaped via symlinks outside static/
    if ".." in rel.parts:
        return None
    return "app/static/" + rel.as_posix()


def static_serving_enabled() -> bool:
    try:
        from streamlit import config as st_config
        return bool(st_config.get_option("server.enableStaticServing"))
    except Exception:
        return False


def save_upload_bytes(data: bytes, original_name: str, root_dir: str | Path) -> tuple[str, str, str]:
    """Persist an upload under ``frontend/static/uploads/``.

    Returns ``(absolute_path, display_name, static_url)``.
    """
    display_name = sanitize_filename(original_name)
    stored_name = unique_upload_name(original_name)
    dest = uploads_dir(root_dir) / stored_name

    # Final path-safety check before writing.
    if not is_under_directory(dest, uploads_dir(root_dir)):
        raise ValueError("Refusing to write outside the uploads directory")

    dest.write_bytes(data)
    url = static_url_for(dest, root_dir)
    if url is None:
        dest.unlink(missing_ok=True)
        raise RuntimeError("Saved upload is not addressable via static URL")
    return str(dest), display_name, url


def remove_upload_file(path: str | Path | None, root_dir: str | Path) -> None:
    """Delete a previously saved upload. Never touches preset static files."""
    if not path or not is_upload_path(path, root_dir):
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass
