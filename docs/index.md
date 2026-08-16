# ACE.verify Documentation

> :material-shield-check: **A multimodal deepfake detection platform** combining Vision Transformers, temporal modeling, and audio spectrogram analysis with an interactive web interface.

---

## Navigation Index

| Page | Description |
|---|---|
| [Architecture and Pipeline](Architecture-and-Pipeline.md) | :material-sitemap: Deep dive into the ML model architecture, video/audio processing pipeline, frame extraction, Grad-CAM generation, and confidence scoring. |
| [UI and Frontend Components](UI-and-Frontend-Components.md) | :material-monitor-dashboard: Documentation of the Streamlit web interface, component hierarchy, responsive layout, session state management, and custom CSS design system. |
| [API and Backend Reference](API-and-Backend-Reference.md) | :material-console: Detailed reference for all CLI entrypoints, data structures (HDF5), model classes, evaluation scripts, and progress streaming. |
| [Setup and Deployment](Setup-and-Deployment.md) | :material-docker: Developer and DevOps guide covering local setup, Conda/Docker environments, GPU/CUDA configuration, and Kubernetes/NRP deployment. |
| [Troubleshooting and FAQ](Troubleshooting-and-FAQ.md) | :material-help-circle: Solutions to common setup issues, GPU/FFMPEG errors, file format problems, and frequently asked questions. |

---

## Project Goals

ACE.verify aims to provide a robust, explainable, and user-friendly deepfake detection system that:

1. **Detects manipulated facial media** using a multimodal approach that analyzes both video frames and audio spectrograms.
2. **Explains its predictions** through Grad-CAM heatmaps that highlight which facial regions influenced the model's decision.
3. **Tracks manipulation over time** via a temporal fakeness timeline that allows analysts to identify exactly when in a video sequence manipulation occurs.
4. **Supports multiple model architectures** including EfficientNet-B4 (for speed), XceptionNet (for accuracy), and the ACE.verify multimodal model (for best overall performance).

---

## Module Summaries

### Core Training Package (`aceverify/`)

The installable Python package containing the model architecture, training loop, dataset handling, and preprocessing pipeline.

- **`model.py`** &mdash; `ACEVerifyModel`, a four-stream multi-domain detector: a ViT-B/16 spatial backbone (last 4 blocks trainable) pooled by localized artifact attention, a DCT/SRM frequency stream, a patch-level temporal-incoherence stream, and an `EfficientNet-B0` audio encoder, combined by attention fusion over modality tokens. The auxiliary streams are added to the spatial feature through a LayerScale gate so the pretrained backbone keeps a direct path to the classifier. Also exposes `load_from_checkpoint`, which infers the architecture from a checkpoint's keys.
- **`baseline_model.py`** &mdash; `ACEVerifyBaseline`, a frozen copy of the pre-enhancement architecture (ViT-B/16 + BiGRU + gated concat fusion), kept for ablations and metric-delta comparisons.
- **`modules.py`** &mdash; Building blocks: `FrequencyStream` (FFT-based orthonormal 2D DCT log-spectra with learnable radial band gating, plus fixed SRM high-pass residuals), `PatchArtifactAttention` (per-patch artifact map + attention pooling), `PatchTemporalIncoherence`, `TemporalCoherenceEncoder` (cross-frame attention + BiGRU), and `ModalityFusion`.
- **`losses.py`** &mdash; `DeepfakeCriterion`: BCE with label smoothing, a supervised contrastive term over the fused embedding, and a boundary-aware artifact sparsity term on the per-patch map.
- **`train.py`** &mdash; Training pipeline with `DeepfakeCriterion`, `AdamW` with separate learning rates for pretrained backbones and new heads, per-step warmup + cosine schedule, gradient clipping, AMP with an OOM-safe batch-size probe, and per-epoch metrics (accuracy, AUC, F1, VRAM, wall time) exported to CSV/JSON.
- **`dataset.py`** &mdash; `ACEDataset` PyTorch Dataset class that reads video frames and audio from HDF5 files, normalizes frames with the backbone's `mean`/`std`, applies clip-consistent augmentation during training (the same parameters for every frame, so no synthetic flicker is injected), and computes dB-scaled, per-sample standardized log-Mel spectrograms with `n_mels=128`, `n_fft=2048`, `hop_length=190` at 48 kHz. `read_labels` memoizes the HDF5 label scan, which otherwise costs ~45 s per epoch.
- **`preprocess.py`** &mdash; DFDC-style zip-to-HDF5 preprocessing pipeline. Extracts 16 frames per video starting at 5 seconds offset, detects faces via `MTCNN` (facenet-pytorch), crops face regions with margin (+80/-50 pixels), saves processed frames and 0.5s audio clips into HDF5 groups with `video` and `audio` datasets.
- **`visualize_data.py`** &mdash; Dataset visualization helper that displays 4 sampled frames and the Mel-spectrogram for a given HDF5 record.

### Web Application (`frontend/`)

The production Streamlit web application for interactive deepfake detection.

- **`app.py`** (533 lines) &mdash; Main application handling media uploads, example media selection, model switching, Grad-CAM generation, temporal timeline rendering, and detection results display. Uses `st.session_state` for state management across reruns.
- **`app.css`** (392 lines) &mdash; Custom CSS design system with CSS variables (cream/ink/amber color palette), responsive flexbox layouts, card components, and custom widget styling.

### Utility Modules (`utilities/`)

Shared helper modules used by both the training pipeline and web application.

- **`gradcam.py`** &mdash; `generate_gradcam()` produces Grad-CAM heatmaps by hooking the last Conv2d layer, computing gradient-weighted activations, upsampling to input size, and blending with the denormalized input image. Includes `region_scores_from_heatmap()` for facial region decomposition and `evidence_from_regions()` for interpretable evidence flags.
- **`preprocess.py`** &mdash; `FaceProcessor` class using MediaPipe Face Landmarker for face alignment. Computes affine transformation matrices based on eye corner positions to canonical coordinates. Provides `extract_image()`, `extract_frames()`, `get_face_count()`, and metadata extraction methods.
- **`media_preview.py`** &mdash; `render_media_preview()` renders compact video/image previews inside the upload card using HTML iframes with custom controls (play/pause, seek, mute) and base64/static-URL fallback.
- **`static_media.py`** &mdash; Static media storage and URL helpers. Handles upload file sanitization (path traversal prevention), collision-safe naming with UUID prefixes, MIME type patching for Streamlit static serving, and upload lifecycle management.
- **`timeline.py`** &mdash; `generate_timeline()` generates a temporal fakeness timeline using a Beta distribution with random spikes. `render_timeline_html()` renders an HTML timeline with color-coded segments (red > 65%, amber 35&ndash;65%, green < 35%).
- **`model.py`** &mdash; `load_model()` loads a model by name with JIT (TorchScript) support for the ACE.verify model. `get_fake_prob()` computes the sigmoid probability from a model output tensor.

### Evaluation & Benchmarking (`evaluation/`)

Standalone scripts for evaluating trained checkpoints and benchmarking against baseline models.

- **`evaluate.py`** &mdash; Standalone evaluation script that loads a checkpoint, runs inference on an HDF5 test set, and saves predictions to CSV.
- **`aceverify_test.py`** &mdash; Benchmarks the ACEVerifyModel (ViT-B/16 + GRU + audio fusion) against the test dataset.
- **`spatial2D_test.py`** &mdash; Benchmarks 2D spatial baseline models (timm `xception` and `efficientnet_b4`) using per-frame classification with video-level mean aggregation.
- **`timeSformer_test.py`** &mdash; Benchmarks the HuggingFace TimeSformer baseline (`facebook/timesformer-base-finetuned-k400`) for video classification.

### Model Wrappers (`models/`)

Alternative model implementations selectable from the web application UI.

- **`ace_verify.py`** &mdash; `ACEVerifyIntegration` class extending `ACEVerifyModel` with TorchScript model loading support.
- **`efficientnet.py`** &mdash; `DeepfakeEfficientNet` class wrapping `timm tf_efficientnet_b4` with `num_classes=1`.
- **`xception.py`** &mdash; `DeepfakeXception` class wrapping `timm xception` with `num_classes=1`.

### Deployment (`assets/templates/`, `scripts/`)

Kubernetes/NRP deployment manifests and helper scripts.

- **`nrp-pvc.yaml`** &mdash; 200 GiB `PersistentVolumeClaim` on `rook-ceph-block` storage class.
- **`nrp-gpu-job.yaml`** &mdash; `batch/v1 Job` for single-GPU training with `nvidia.com/gpu.product: NVIDIA-RTX-A6000` node selector.
- **`nrp-sweep-job.yaml`** &mdash; `batch/v1 Job` with 4 parallel completions for hyperparameter sweeps.
- **`copy-to-pvc.sh`** / **`copy-from-pvc.sh`** &mdash; Bash scripts using `kubectl` ephemeral pods and `kubectl cp` to stage files between local storage and a PVC.

---

## Quick Links

- :material-github: [GitHub Repository](https://github.com/AB-at-UCR/ACE.verify)
- :material-database: [Dataset: Deepfake Detection Challenge](https://www.kaggle.com/competitions/deepfake-detection-challenge/data)
- :material-google-drive: [Model Checkpoint on Google Drive](https://drive.google.com/file/d/1d3ln2laSfmXkKyXHZ1YhK_gb33nonaPO/view?usp=sharing)
