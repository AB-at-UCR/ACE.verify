# ACE.verify Wiki

> A multimodal deepfake detection platform combining Vision Transformers, temporal modeling, and audio spectrogram analysis with an interactive web interface.

---

## Navigation Index

| Wiki Page | Description |
|---|---|
| [Architecture and Pipeline](Architecture-and-Pipeline) | Deep dive into the ML model architecture, video/audio processing pipeline, frame extraction, Grad-CAM generation, and confidence scoring. |
| [UI and Frontend Components](UI-and-Frontend-Components) | Documentation of the Streamlit web interface, component hierarchy, responsive layout, session state management, and custom CSS design system. |
| [API and Backend Reference](API-and-Backend-Reference) | Detailed reference for all CLI entrypoints, data structures (HDF5), model classes, evaluation scripts, and progress streaming. |
| [Setup and Deployment](Setup-and-Deployment) | Developer and DevOps guide covering local setup, Conda/Docker environments, GPU/CUDA configuration, and Kubernetes/NRP deployment. |
| [Troubleshooting and FAQ](Troubleshooting-and-FAQ) | Solutions to common setup issues, GPU/FFMPEG errors, file format problems, and frequently asked questions. |

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

- **`model.py`** &mdash; `ACEVerifyModel` class combining a ViT-B/16 video backbone (last 4 blocks trainable), a bidirectional GRU temporal layer, `TemporalAttentionPooling`, an `EfficientNet-B0` audio spectrogram encoder, a gated multimodal fusion unit, and a 3-layer classifier MLP.
- **`train.py`** &mdash; Training pipeline with `BCEWithLogitsLoss` (pos_weight=2.0), `AdamW` optimizer (lr=5e-5, weight_decay=1e-4), `StepLR` scheduler (step_size=2, gamma=0.5), checkpoint resume, and per-epoch metrics CSV export.
- **`dataset.py`** &mdash; `ACEDataset` PyTorch Dataset class that reads video frames and audio from HDF5 files, applies data augmentation (ColorJitter, RandomErasing) during training, and computes Mel-spectrograms with `n_mels=32`, `n_fft=400`, `hop_length=160` at 44100 Hz.
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
- **`timeline.py`** &mdash; `generate_timeline()` generates a temporal fakeness timeline using a Beta distribution with random spikes. `render_timeline_html()` renders an HTML timeline with color-coded segments (red > 65%, amber 35-65%, green < 35%).
- **`model.py`** &mdash; `load_model()` loads a model by name with JIT (TorchScript) support for the ACE.verify model. `get_fake_prob()` computes the sigmoid probability from a model output tensor.

### Evaluation & Benchmarking (`evaluation/`)

Standalone scripts for evaluating trained checkpoints and benchmarking against baseline models.

- **`evaluate.py`** &mdash; Standalone evaluation script that loads a checkpoint, runs inference on an HDF5 test set, and saves predictions to CSV.
- **`aceverify_test.py`** &mdash; Benchmarks the ACEVerifyModel (ViT-B/16 + GRU + audio fusion) against the test dataset.
- **`spatial2D_test.py`** &mdash; Benchmarks 2D spatial baseline models (timm `xception` and `efficientnet_b4`) using per-frame classification.
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

- [GitHub Repository](https://github.com/AB-at-UCR/ACE.verify)
- [Dataset: Deepfake Detection Challenge](https://www.kaggle.com/competitions/deepfake-detection-challenge/data)
- [Model Checkpoint on Google Drive](https://drive.google.com/file/d/1d3ln2laSfmXkKyXHZ1YhK_gb33nonaPO/view?usp=sharing)
