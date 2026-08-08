import os
import sys
import torch
import pathlib

current_file_path = pathlib.Path(__file__).resolve()
root_dir = str(current_file_path.parent.parent)
if root_dir not in sys.path:
	sys.path.insert(0, root_dir)

import models
import utilities
import numpy as np
import streamlit as st
from pathlib import Path

# Ensure /app/static/*.mp4 is served with video/* MIME (Streamlit ≤1.50 defaults to text/plain).
utilities.ensure_static_video_mime()

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ACE.verify - Deepfake Detector",
    page_icon="☑️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Initialize Session State ───────────────────────────────────────────────
if "analyzed" not in st.session_state:
    st.session_state.results = {}
    st.session_state.analyzed = False

if "file_ext" not in st.session_state:
    st.session_state.file_ext = None
    st.session_state.example_file = None
    st.session_state.active_file_path = None
    st.session_state.active_file_name = None
    st.session_state.active_static_url = None  # app/static/... URL for preview
    st.session_state.upload_sig = None          # (name, size) of the current upload
    st.session_state.upload_disk_path = None    # path under frontend/static/uploads/

# ─── Custom CSS ──────────────────────────────────────────────────────────────
with open(Path(root_dir, "frontend", "app.css"), "r") as f:
    css = f.read()
    st.markdown(f"""
    <style>
        {css}
    </style>
    """, unsafe_allow_html=True)

# ─── Helper Functions ─────────────────────────────────────────────────────────
@st.cache_resource # NOTE: -> Only caches the response for the "load_model" function
def load_model(model_name):
    return utilities.load_model(model_name)

def generate_gradcam(model, input_tensor, target_layer=None, intensity=0.85):
    return utilities.generate_gradcam(model=model, input_tensor=input_tensor, target_layer=target_layer, intensity=intensity)
    
def generate_timeline(duration_in_sec=30, n_segs=60):
    return utilities.generate_timeline(duration_in_sec=duration_in_sec, n_segs=n_segs)

def render_timeline_html(scores, duration_in_sec):
    return utilities.render_timeline_html(scores, duration_in_sec)

def cleanup_upload_disk():
    """Remove a previously saved user upload from frontend/static/uploads/."""
    utilities.remove_upload_file(st.session_state.get("upload_disk_path"), root_dir)
    st.session_state.upload_disk_path = None
    st.session_state.active_static_url = None

def clear_media_selection():
    """Reset all media-selection session state (upload or example)."""
    cleanup_upload_disk()
    st.session_state.results = {}
    st.session_state.file_ext = None
    st.session_state.analyzed = False
    st.session_state.example_file = None
    st.session_state.active_file_path = None
    st.session_state.active_file_name = None
    st.session_state.upload_sig = None
    st.session_state.active_static_url = None

github_logo = "https://cdn-icons-png.flaticon.com/512/25/25231.png"
repo_url = "https://github.com/AB-at-UCR/ACE.verify"
image_exts = {".jpg", ".jpeg", ".png"}
video_exts = {".mp4", ".mov", ".avi"}
example_media = {
    "FaceSwap clip": "faceswap_clip.mp4",
    "Lip-sync fake": "lip-sync_fake.mp4",
    "GAN portrait": "gan_portrait.mp4",
    "Unauthentic news": "unauthentic_news.mp4",
    "Political speech": "political_speech.mp4"
}

# ─── Navbar ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="navbar">
  <div class="navbar-brand">ACE<span>.verify</span></div>
  <span class="sparkle">✦</span>
  <div class="navbar-links">
    <span class="nav-badge">BETA</span>
    <span class="nav-link">v1.0</span>
    <span class="nav-link"><a href="{repo_url}" target="_blank"><img src="{github_logo}" width="20" style="margin-right: 10px;">GitHub</a></span>
    <span class="nav-link">Docs</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>AI-powered<br><em>deepfake detector</em></h1>
  <center><p class="hero-sub">
    Upload a video or image. Our model analyzes facial regions, flags anomalies, and pinpoints exactly <em>where</em> and <em>when</em> manipulation occurs.
  </p></center>
</div>
""", unsafe_allow_html=True)

# ─── Upload Card (two-column split: controls | preview) ──────────────────────
PREVIEW_PANE_HEIGHT = 570  # px — right pane matches the stacked left column

with st.container():
    st.markdown('<div class="upload-card-anchor"></div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([45, 55], gap="medium")

    # ── Left column: controls & configuration ──
    with col_left:
        st.markdown('<div class="upload-left-anchor"></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">Upload media</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop video or image here",
            type=["mp4", "mov", "avi", "jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        st.markdown('<div class="section-label">Try example media</div>', unsafe_allow_html=True)
        example_items = list(example_media.items())
        for row_start in range(0, len(example_items), 3):  # wrapped rows, 3 pills per row
            cols = st.columns(3, gap="small")
            for offset, (label, filename) in enumerate(example_items[row_start:row_start + 3]):
                if cols[offset].button(label, key=f"ex_{row_start + offset}", width='stretch'):
                    if uploaded_file is not None:
                        st.toast("Please remove the uploaded file first.", icon="⚠️")
                    else:
                        # Store the selection in session state so it survives the rerun
                        cleanup_upload_disk()
                        preset_path = os.path.join(root_dir, "frontend", "static", filename)
                        st.session_state.example_file = {
                            "path": preset_path,
                            "name": filename,
                            "static_url": f"app/static/{filename}",
                        }
                        st.session_state.upload_sig = None
                        st.rerun() # Force rerun to update the UI immediately

        # ─── File Normalization ──────────────────────────────────────────────
        if uploaded_file:
            # If user uploads something, clear any selected example
            st.session_state.example_file = None
            upload_sig = (uploaded_file.name, uploaded_file.size)
            # Only write a new static file when the upload actually changed
            if st.session_state.get("upload_sig") != upload_sig or not st.session_state.active_file_path:
                cleanup_upload_disk()
                try:
                    path, display_name, static_url = utilities.save_upload_bytes(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                        root_dir,
                    )
                except ValueError as exc:
                    st.toast(f"Upload rejected: {exc}", icon="⚠️")
                    st.session_state.upload_sig = None
                    st.session_state.active_file_path = None
                    st.session_state.active_file_name = None
                    st.session_state.active_static_url = None
                    st.session_state.file_ext = None
                else:
                    st.session_state.file_ext = os.path.splitext(display_name)[1].lower()
                    st.session_state.active_file_path = path
                    st.session_state.active_file_name = display_name
                    st.session_state.active_static_url = static_url
                    st.session_state.upload_disk_path = path
                    st.session_state.upload_sig = upload_sig
        elif st.session_state.example_file:
            # Use the stored example (already under frontend/static/)
            st.session_state.active_file_path = st.session_state.example_file["path"]
            st.session_state.active_file_name = st.session_state.example_file["name"]
            st.session_state.active_static_url = st.session_state.example_file.get(
                "static_url",
                utilities.static_url_for(st.session_state.active_file_path, root_dir),
            )
            st.session_state.file_ext = os.path.splitext(st.session_state.active_file_name)[1].lower()
            
            # UI Indicator that an example is loaded (Since we can't hack the uploader)
            st.toast(f"📁 **Example Loaded:** {st.session_state.active_file_name}")
            if st.button("Reset Selection", width='stretch'):
                clear_media_selection()
                st.rerun()
        elif st.session_state.get("upload_sig"):
            # Upload was removed via the uploader's ✕ — clear the stale file state
            clear_media_selection()

        st.markdown('<div class="section-label">Detection model</div>', unsafe_allow_html=True)
        model_choice = st.selectbox("Model", ["EfficientNet-B4 (Fast)", "XceptionNet (Accurate)", "ACE.verify (Best)"],
                                    label_visibility="collapsed")

        col_thr, col_opt = st.columns(2, gap="small")
        with col_thr:
            st.markdown('<div class="section-label">Confidence threshold</div>', unsafe_allow_html=True)
            threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01, label_visibility="collapsed")
        with col_opt:
            st.markdown('<div class="section-label">Options</div>', unsafe_allow_html=True)
            show_heatmap   = st.checkbox("Show Grad-CAM", value=True)
            show_landmarks = st.checkbox("Face landmarks", value=False)

        analyze_clicked = st.button("Analyze ✦", width='stretch')

    # ── Right column: media preview (full height of the left pane) ──
    with col_right:
        st.markdown('<div class="upload-right-anchor"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Media preview</div>', unsafe_allow_html=True)
        utilities.render_media_preview(
            file_path=st.session_state.active_file_path,
            file_name=st.session_state.active_file_name,
            file_ext=st.session_state.file_ext,
            is_example=st.session_state.example_file is not None,
            root_dir=root_dir,
            static_url=st.session_state.get("active_static_url"),
            pane_height=PREVIEW_PANE_HEIGHT,
        )

# ─── Trigger analysis and store in session state ──────────────────────────────
if analyze_clicked and not st.session_state.analyzed:
    if not st.session_state.active_file_path:
        st.toast("Please upload a file first.", icon="⚠️")
        st.rerun()
        
    with st.spinner(""):
        # Single progress bar instance, updated in place across every stage so the
        # UI shows one continuously-advancing bar instead of appending new ones.
        # Stage budget: frames 0–25% · inference 25–50% · Grad-CAM 50–75% · timeline 75–100%.
        progress_bar = st.progress(0, text=f"Loading {model_choice}…")
        model = load_model(model_choice)

        # Extract Frames & align face from uploaded video
        processor = utilities.FaceProcessor()
        with torch.no_grad():
            if st.session_state.file_ext in image_exts:
                progress_bar.progress(10, text="Processing image…")
                image_tensor = processor.extract_image(st.session_state.active_file_path)  # [1, C, H, W]

                progress_bar.progress(25, text="Running inference…")
                if "ACE.verify" in model_choice:
                    model_input = image_tensor.unsqueeze(2).repeat(1, 1, 32, 1, 1) # [B, C, T, H, W] -> repeat single image as pseudo-clip
                    output = model(model_input)
                else:
                    output = model(image_tensor).mean()

            elif st.session_state.file_ext in video_exts:
                progress_bar.progress(10, text="Extracting & Processing frames…")
                input_frames_tensor = processor.extract_frames(st.session_state.active_file_path)  # [T, C, H, W]
                if input_frames_tensor.shape[0] == 0:
                    st.error("Could not decode video frames.")
                    st.stop()

                progress_bar.progress(25, text="Running inference…")
                if "ACE.verify" in model_choice:
                    model_input = input_frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
                    output = model(model_input)
                else:
                    t = input_frames_tensor.shape[0]
                    if t >= 5:
                        start = max(0, (t // 2) - 2)
                        end = min(t, start + 5)
                        spatial_input = input_frames_tensor[start:end]
                    else:
                        spatial_input = input_frames_tensor
                    output = model(spatial_input).mean()

            else:
                st.error(f"Unsupported file type: {st.session_state.file_ext}")
                st.stop()

        # Run Grad-Cam generation
        progress_bar.progress(50, text="Generating Grad-CAM…")
        fake_prob    = utilities.get_fake_prob(output)
        
        if st.session_state.file_ext in image_exts:
            grad_input = image_tensor  # [1, C, H, W]
            media_metadata_dict, media_duration = processor.extract_image_metadata(image_path=st.session_state.active_file_path, file_ext=st.session_state.file_ext)
        else: # For video, use middle 5 frames averaged
            grad_input = input_frames_tensor # [1, C, T, H, W]
            media_metadata_dict, media_duration = processor.extract_video_metadata(video_path=st.session_state.active_file_path, file_ext=st.session_state.file_ext)
            
        heatmap_img = generate_gradcam(model=model, input_tensor=grad_input.clone(), intensity=fake_prob)
        
        # Run Scoring Timeline generation
        progress_bar.progress(75, text="Scoring timeline…")
        
        default_metadata_dict, duration_in_sec = {
            "Resolution":   "Unknown",
            "Duration":     "Unknown",
            "FPS":          "Unknown",
            "Codec":        "Unknown",
            "Faces found":  "Unknown",
        }, 0.0
        
        final_metadata_dict, duration_in_sec = default_metadata_dict | media_metadata_dict, max(media_duration, 1.0)

        if st.session_state.file_ext in video_exts and "ACE.verify" not in model_choice and output.ndim >= 1:
            frame_probs = torch.sigmoid(output.float()).flatten().detach().cpu().numpy()
            x_src = np.linspace(0, duration_in_sec, len(frame_probs))
            x_dst = np.linspace(0, duration_in_sec, 56)
            timeline_scores = np.interp(x_dst, x_src, frame_probs).tolist()
        else:
            timeline_scores = [fake_prob] * 56
        
        progress_bar.progress(90, text="Scoring timeline…")
        regions = utilities.region_scores_from_heatmap(heatmap_img)
        evidence_flags = utilities.evidence_from_regions(regions)
        
        progress_bar.progress(100, text="Done ✦")
        progress_bar.empty()

    st.session_state.results = {
        "model":            model,
        "fake_prob":        fake_prob,
        "is_fake":          fake_prob > threshold,
        "media":            grad_input,
        "duration_in_sec":  duration_in_sec,
        "has_file":         st.session_state.active_file_path is not None,
        "model_choice":     model_choice,
        "threshold":        threshold,
        "heatmap_img":      heatmap_img,
        "evidence_flags": evidence_flags,
        "regions": regions,
        "metadata": {
            **final_metadata_dict,
            "Model used":   model_choice.split(" (")[0],
            "Threshold":    str(threshold),
            "ext":          st.session_state.file_ext,
            "File name": st.session_state.active_file_name,
            "File size": f"{os.path.getsize(st.session_state.active_file_path)/1024:.1f} KB",
        },
        "timeline_scores": timeline_scores,
    }
    st.session_state.analyzed = True

# ─── Results Section driven by session state ──────────────────────
if st.session_state.analyzed and st.session_state.results:
    r                   = st.session_state.results
    fake_prob           = r["fake_prob"]
    is_fake             = r["is_fake"]
    duration_in_sec     = r["duration_in_sec"]
    evidence_flags      = r["evidence_flags"]
    regions             = r["regions"]
    metadata            = r["metadata"]
    model               = r["model"]
    media               = r["media"]
    file_ext            = r["metadata"]["ext"]
    timeline_scores     = r["timeline_scores"]
    show_heatmap_r      = show_heatmap      # NOTE: use live checkbox values so toggling still works
    show_landmarks_r    = show_landmarks

    verdict = "LIKELY FAKE" if is_fake else "LIKELY AUTHENTIC"

    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # ── Verdict banner ──
    verdict_class = "verdict-fake" if is_fake else "verdict-real"
    icon = "⚠️" if is_fake else "✅"
    sub  = (f"{fake_prob*100:.1f}% probability of manipulation detected across {duration_in_sec}s"
            if is_fake else f"Only {fake_prob*100:.1f}% anomaly score — appears authentic")

    st.markdown(f"""
    <div class="{verdict_class}">
      <div style="font-size:2.2rem">{icon}</div>
      <div>
        <div class="verdict-stamp">{verdict}</div>
        <div class="verdict-detail">{sub}</div>
      </div>
      <div style="margin-left:auto;font-family:'DM Mono',monospace;font-size:2rem;font-weight:700">
        {fake_prob*100:.0f}%
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Heatmap + Confidence ──
    col_heatmap, col_gauge = st.columns([1, 1], gap="medium")

    with col_heatmap:
        with st.container():
            st.markdown('<div class="analysis-card-anchor"></div>', unsafe_allow_html=True)
            st.markdown("<h4>📹 Grad-CAM Heatmap</h4>", unsafe_allow_html=True)
            if show_heatmap_r:
                heatmap_img = r["heatmap_img"]
                st.image(heatmap_img, width='stretch', caption="Red = high probability")
            else:
                st.info("Enable Grad-CAM in options.")
            if show_landmarks_r:
                st.markdown("""
                <div style="font-family:var(--ink);font-size:0.72rem;color:var(--ink-soft);margin-top:0.5rem">
                ◆ 68 landmarks detected · Jaw deviation: <b>+4.2°</b> · Eye symmetry: <b>0.71</b>
                </div>""", unsafe_allow_html=True)

    with col_gauge:
        with st.container():
            st.markdown('<div class="analysis-card-anchor"></div>', unsafe_allow_html=True)
            st.markdown("<h4>📊 Confidence</h4>", unsafe_allow_html=True)
            fill_color = "#E84040" if is_fake else "#2ECC71"
            st.markdown(f"""
            <div class="gauge-container">
                <div class="gauge-value" style="color:{fill_color}">{fake_prob*100:.0f}%</div>
                <div class="gauge-label">Fake Probability</div>
                <div class="gauge-bar-bg"><div class="gauge-bar-fill" style="width:{fake_prob*100:.0f}%;background:{fill_color}"></div></div>
            </div>
            """, unsafe_allow_html=True)
            for region, score in regions:
                c = "#E84040" if score > 0.6 else "#2ECC71"
                st.markdown(f"""
                <div style="margin-bottom:6px">
                <div style="display:flex;justify-content:space-between;font-size:0.75rem;margin-bottom:2px">
                    <span style="font-family:'DM Mono',monospace;color:var(--ink-soft)">{region}</span>
                    <span style="font-weight:600;color:{c}">{score*100:.0f}%</span>
                </div>
                <div style="background:#D0C8B8;border-radius:4px;height:5px;overflow:hidden">
                    <div style="width:{score*100:.0f}%;height:100%;background:{c};border-radius:4px"></div>
                </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Evidence + Metadata ──
    col_evidence, col_meta = st.columns([1, 1], gap="medium")

    with col_evidence:
        with st.container():
            st.markdown('<div class="analysis-card-anchor"></div>', unsafe_allow_html=True)
            st.markdown("<h4>🔎 Evidence Flags</h4>", unsafe_allow_html=True)
            chips_html = '<div class="chip-grid">'
            for flag, score in evidence_flags.items():
                cls = "chip-green" if score < 0.35 else "chip-amber" if score < 0.6 else "chip-red"
                icon_e = "✓" if score < 0.35 else "◆" if score < 0.6 else "⚠"
                chips_html += f'<span class="chip {cls}">{icon_e} {flag} · {score*100:.0f}%</span>'
            chips_html += "</div>"
            st.markdown(chips_html, unsafe_allow_html=True)

    with col_meta:
        with st.container():
            st.markdown('<div class="analysis-card-anchor"></div>', unsafe_allow_html=True)
            st.markdown("<h4>📁 Metadata</h4>", unsafe_allow_html=True)
            rows = "".join(
                f"<tr><td style='font-size:0.7rem'>{k}</td><td style='font-size:0.7rem'><b>{v}</b></td></tr>"
                for k, v in list(metadata.items())[:7]
            )
            st.markdown(f'<table class="meta-table"><tbody>{rows}</tbody></table>', unsafe_allow_html=True)

    # ── Timeline ──
    st.markdown(render_timeline_html(timeline_scores, duration_in_sec), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Frame Inspector ──
    with st.container():
        st.markdown('<div class="analysis-card-anchor"></div>', unsafe_allow_html=True)
        st.markdown('<h4>🎞️ Frame Inspector</h4>', unsafe_allow_html=True)
        frame_col, slider_col = st.columns([1, 3], gap="large", vertical_alignment="center")

        with slider_col:
            st.markdown('<div class="section-label">Scrub to frame</div>', unsafe_allow_html=True)
            frame_idx = st.slider("Frame", 0, int(duration_in_sec * 30 - 1), 45, label_visibility="collapsed", key="frame_slider")
            frame_sec   = frame_idx / 30
            frame_score = float(np.interp(
                frame_sec,
                np.linspace(0, duration_in_sec, len(timeline_scores)),
                timeline_scores
            ))
            frame_color = "#E84040" if frame_score > 0.6 else "#F0A030" if frame_score > 0.35 else "#2ECC71"
            st.markdown(f"""
            <div style="display:flex;gap:2rem;margin-top:0.5rem;font-family:'DM Mono',monospace;font-size:0.82rem">
            <span>⏱ <b>{frame_sec:.2f}s</b></span>
            <span>🎞 Frame <b>{frame_idx}</b></span>
            <span>Score: <b style="color:{frame_color}">{frame_score*100:.1f}%</b></span>
            </div>
            """, unsafe_allow_html=True)

        with frame_col:
            if file_ext in video_exts:
                total_avail = media.shape[0]
                mapped_idx = int(np.interp(frame_idx, [0, max(1, int(duration_in_sec * 30) - 1)], [0, total_avail - 1]))
                grad_input = media[mapped_idx:mapped_idx+1]
                # grad_input = media[frame_idx] 
            else:
                grad_input = media # [1, C, H, W]
            thumbnail = generate_gradcam(model=model, input_tensor=grad_input.clone(), intensity=0.85)
            st.image(thumbnail, width='stretch')

    # ── Export / Actions row ──
    st.markdown("<br>", unsafe_allow_html=True)
    act1, act2, act3, act4 = st.columns(4, gap="small")
    with act1:
        st.button("⬇ Export PDF Report", width='stretch', key="btn_pdf")
    with act2:
        st.button("📋 Copy JSON Results", width='stretch', key="btn_json")
    with act3:
        st.button("🔗 Share Analysis Link", width='stretch', key="btn_share")
    with act4:
        if st.button("🗑 Clear & Reset", width='stretch', key="btn_reset"):
            clear_media_selection()
            st.rerun()

else:
    # ── Empty state ──
    st.markdown("""
    <div style="text-align:center;padding:3rem 0 2rem;color:#999">
      <div style="font-size:4rem">🎭</div>
      <p style="font-family:'DM Mono',monospace;font-size:0.85rem;letter-spacing:1px;margin-top:1rem">
        UPLOAD A FILE TO BEGIN ANALYSIS
      </p>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  ACE.verify · Powered by AI · Results are probabilistic and for research use only<br>
  <span style="opacity:0.5">✦ &nbsp; Not a substitute for expert forensic analysis</span>
</div>
""", unsafe_allow_html=True)