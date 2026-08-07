"""Compact media preview rendered inside the upload card.

The preview is a small custom HTML component (iframe). Local/uploaded files are
shipped to the browser as base64 and turned into a blob URL via
``URL.createObjectURL``; the blob is released with ``URL.revokeObjectURL`` when
the component is torn down (i.e. when the file is removed or replaced).
Example clips are served through Streamlit's static file route when static
serving is enabled, falling back to the blob path otherwise.
"""

import os
import html
import json
import base64
from pathlib import Path

import streamlit as st

VIDEO_MIME = {".mp4": "video/mp4", ".mov": "video/quicktime", ".avi": "video/x-msvideo"}
IMAGE_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}

# Above this size, inlining base64 into the page would freeze the browser tab.
MAX_INLINE_BYTES = 80 * 1024 * 1024

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap');
:root {
    --cream:     #F5F0E8;
    --card-bg:   #EDE8DC;
    --ink:       #1A1A1A;
    --ink-soft:  #4A4A4A;
    --amber:     #F0A030;
    --amber-dark:#D4891A;
    --border:    #C8BFA8;
    --mono:      'DM Mono', monospace;
}
* { box-sizing: border-box; }
body { margin: 0; padding: 2px 6px 8px 2px; background: transparent; font-family: 'DM Sans', sans-serif; }

.pv-meta {
    display: flex; align-items: center; gap: 8px;
    margin: 0 0 6px 2px;
    font-family: var(--mono); font-size: 0.7rem; color: var(--ink-soft);
}
.pv-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; letter-spacing: 0.5px; }
.pv-tag {
    flex: 0 0 auto;
    background: var(--amber); color: var(--ink);
    border: 1.5px solid var(--ink); border-radius: 4px;
    padding: 1px 6px; font-size: 0.6rem; font-weight: 500;
    letter-spacing: 1px; text-transform: uppercase;
}

.pv-frame {
    background: #101010;
    border: 2px solid var(--ink); border-radius: 12px;
    box-shadow: 3px 3px 0 var(--ink);
    overflow: hidden;
}
.pv-media { display: block; width: 100%; height: 230px; object-fit: contain; background: #000; }

.pv-controls { display: flex; align-items: center; gap: 10px; padding: 8px 10px; background: var(--ink); }
.pv-btn {
    flex: 0 0 auto; width: 34px; height: 28px;
    display: flex; align-items: center; justify-content: center;
    background: var(--amber); color: var(--ink);
    border: 2px solid #000; border-radius: 6px;
    font-size: 0.7rem; font-weight: 700; cursor: pointer;
    box-shadow: 2px 2px 0 #000; transition: all 0.15s ease;
    padding: 0; line-height: 1;
}
.pv-btn:hover  { background: var(--amber-dark); transform: translate(-1px, -1px); box-shadow: 3px 3px 0 #000; }
.pv-btn:active { transform: translate(1px, 1px); box-shadow: 1px 1px 0 #000; }
.pv-seek { flex: 1 1 auto; min-width: 40px; accent-color: var(--amber); cursor: pointer; margin: 0; }
.pv-time { flex: 0 0 auto; font-family: var(--mono); font-size: 0.68rem; color: var(--cream); white-space: nowrap; }

.pv-note {
    padding: 0.9rem 1rem;
    font-family: var(--mono); font-size: 0.72rem; color: var(--cream);
    letter-spacing: 0.5px;
}

.pv-placeholder {
    border: 2px dashed var(--border); border-radius: 12px;
    background: rgba(255, 255, 255, 0.4);
    text-align: center; padding: 1.5rem 1rem;
    color: var(--ink-soft);
}
.pv-ph-icon { font-size: 2rem; }
.pv-ph-text { font-family: var(--mono); font-size: 0.72rem; letter-spacing: 2px; margin-top: 0.5rem; text-transform: uppercase; }
.pv-ph-sub  { font-size: 0.78rem; margin-top: 0.3rem; opacity: 0.75; }
"""

_SRC_JS = """
var media = document.getElementById("pv-media");
var objectUrl = null;

function revokeObjectUrl() {
    if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
        objectUrl = null;
    }
}
// Streamlit re-renders this component whenever the selected file changes or is
// removed, so releasing the blob on document teardown covers replace + remove.
window.addEventListener("pagehide", revokeObjectUrl);
window.addEventListener("beforeunload", revokeObjectUrl);

if (SRC_URL) {
    // Static URL (example media served by Streamlit's static file route).
    media.src = new URL(SRC_URL, window.parent.location.href).href;
} else if (B64) {
    var bin = atob(B64);
    var bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    objectUrl = URL.createObjectURL(new Blob([bytes], { type: MIME }));
    media.src = objectUrl;
}
"""

_CONTROLS_JS = """
var playBtn = document.getElementById("pv-play");
var muteBtn = document.getElementById("pv-mute");
var seek = document.getElementById("pv-seek");
var timeEl = document.getElementById("pv-time");

function fmt(t) {
    if (!isFinite(t) || isNaN(t)) return "0:00";
    var m = Math.floor(t / 60);
    var s = Math.floor(t % 60);
    return m + ":" + String(s).padStart(2, "0");
}
function sync() {
    seek.value = media.currentTime || 0;
    timeEl.textContent = fmt(media.currentTime) + " / " + fmt(media.duration);
}
media.addEventListener("loadedmetadata", function () { seek.max = media.duration || 100; sync(); });
media.addEventListener("timeupdate", sync);
media.addEventListener("play",  function () { playBtn.textContent = "\\u275A\\u275A"; });
media.addEventListener("pause", function () { playBtn.textContent = "\\u25B6"; });
media.addEventListener("error", function () { timeEl.textContent = "Preview unavailable"; });
playBtn.addEventListener("click", function () { media.paused ? media.play() : media.pause(); });
muteBtn.addEventListener("click", function () {
    media.muted = !media.muted;
    muteBtn.textContent = media.muted ? "\\uD83D\\uDD07" : "\\uD83D\\uDD0A";
});
seek.addEventListener("input", function () { media.currentTime = parseFloat(seek.value); });
"""


@st.cache_data(max_entries=6, show_spinner=False)
def _encode_file(path, mtime, size):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _example_static_url(root_dir, file_name):
    """Return a static URL for an example clip, or None if unavailable."""
    try:
        from streamlit import config as st_config
        if not st_config.get_option("server.enableStaticServing"):
            return None
    except Exception:
        return None
    if file_name and Path(root_dir, "frontend", "static", file_name).is_file():
        return f"app/static/{file_name}"
    return None


def _wrap(body, script=""):
    script_tag = f"<script>(function() {{ {script} }})();</script>" if script else ""
    return f"<!DOCTYPE html><html><head><style>{_CSS}</style></head><body>{body}{script_tag}</body></html>"


def render_media_preview(file_path, file_name, file_ext, is_example, root_dir):
    # Fallback state: nothing selected yet
    if not file_path or not os.path.exists(file_path):
        st.iframe(_wrap("""
        <div class="pv-placeholder">
            <div class="pv-ph-icon">🎞️</div>
            <div class="pv-ph-text">No media selected</div>
            <div class="pv-ph-sub">Upload a file or pick an example to preview it here</div>
        </div>
        """), height=175)
        return

    file_ext = (file_ext or "").lower()
    is_video = file_ext in VIDEO_MIME
    mime = VIDEO_MIME.get(file_ext) or IMAGE_MIME.get(file_ext, "application/octet-stream")
    display_name = html.escape(file_name or os.path.basename(file_path))
    tag = "example" if is_example else "uploaded"
    meta_row = f'<div class="pv-meta"><span class="pv-name">{display_name}</span><span class="pv-tag">{tag}</span></div>'

    src_url = _example_static_url(root_dir, file_name) if is_example else None
    b64 = ""
    if src_url is None:
        size = os.path.getsize(file_path)
        if size > MAX_INLINE_BYTES:
            body = f"""
            {meta_row}
            <div class="pv-frame">
                <div class="pv-note">⚠ File too large for inline preview ({size / (1024 * 1024):.0f} MB) — analysis still works.</div>
            </div>
            """
            st.iframe(_wrap(body), height=110)
            return
        b64 = _encode_file(file_path, os.path.getmtime(file_path), size)

    consts = (
        f"var SRC_URL = {json.dumps(src_url)};"
        f"var MIME = {json.dumps(mime)};"
        f"var B64 = {json.dumps(b64)};"
    )

    if is_video:
        body = f"""
        {meta_row}
        <div class="pv-frame">
            <video id="pv-media" class="pv-media" preload="auto" playsinline></video>
            <div class="pv-controls">
                <button id="pv-play" class="pv-btn" title="Play / pause">&#9654;</button>
                <button id="pv-mute" class="pv-btn" title="Mute / unmute">🔊</button>
                <input id="pv-seek" class="pv-seek" type="range" min="0" max="100" step="0.05" value="0">
                <span id="pv-time" class="pv-time">0:00 / 0:00</span>
            </div>
        </div>
        """
        st.iframe(_wrap(body, consts + _SRC_JS + _CONTROLS_JS), height=325)
    else:
        body = f"""
        {meta_row}
        <div class="pv-frame">
            <img id="pv-media" class="pv-media" alt="{display_name}">
        </div>
        """
        st.iframe(_wrap(body, consts + _SRC_JS), height=280)
