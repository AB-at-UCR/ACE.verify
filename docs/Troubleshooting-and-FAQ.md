# Troubleshooting and FAQ

> Solutions to common setup issues, GPU/FFMPEG errors, file format problems, and frequently asked questions about the ACE.verify platform.

---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [GPU and CUDA Issues](#gpu-and-cuda-issues)
3. [FFmpeg and Preprocessing Issues](#ffmpeg-and-preprocessing-issues)
4. [File Format and Upload Issues](#file-format-and-upload-issues)
5. [Streamlit and Web App Issues](#streamlit-and-web-app-issues)
6. [Docker and Container Issues](#docker-and-container-issues)
7. [Kubernetes and NRP Issues](#kubernetes-and-nrp-issues)
8. [Model and Training Issues](#model-and-training-issues)
9. [Frequently Asked Questions](#frequently-asked-questions)

---

## Installation Issues

### `ModuleNotFoundError: No module named 'aceverify'`

**Cause**: The project was not installed in editable mode.

**Solution**:
```bash
pip install -e .
```

This reads `aceverify/pyproject.toml` and installs the `aceverify` package along with all dependencies.

---

### `aceverify-train: command not found`

**Cause**: The CLI entrypoints were not installed, or the Python environment is not activated.

**Solution**:
1. Ensure your conda environment or virtual environment is activated.
2. Reinstall the project:
   ```bash
   pip install -e .
   ```
3. Verify the entrypoints:
   ```bash
   aceverify-train --help
   aceverify-preprocess --help
   aceverify-evaluate --help
   ```

---

### `ImportError: cannot import name 'timm'` or `ImportError: No module named 'timm'`

**Cause**: The `timm` library is not installed in your environment.

**Solution**:
```bash
pip install timm>=0.9
```

Or reinstall the project, which includes `timm` as a dependency:
```bash
pip install -e .
```

---

### `ImportError: No module named 'mediapipe'`

**Cause**: The MediaPipe library is not installed. It is required by `utilities/preprocess.py` for face landmark detection and alignment.

**Solution**:
```bash
pip install mediapipe
```

---

### Python version compatibility error

**Cause**: Your Python version is outside the supported range (`>=3.10, <3.13`), as defined in `aceverify/pyproject.toml`.

**Solution**:
Install Python 3.10, 3.11, or 3.12. If using Conda:
```bash
conda create -n aceverify python=3.11
conda activate aceverify
pip install -e .
```

---

## GPU and CUDA Issues

### `RuntimeError: CUDA out of memory`

**Cause**: The GPU does not have enough VRAM for the current batch size or model.

**Solutions**:
1. Reduce the batch size: `--batch-size 4` or `--batch-size 2`
2. Reduce the number of frames per sample (modify `num_output_frames` in `aceverify/dataset.py`)
3. Use gradient accumulation (not built-in; requires manual implementation)
4. Use a GPU with more VRAM (16+ GiB recommended)

---

### `RuntimeError: CUDA not available` / `torch.cuda.is_available()` returns `False`

**Cause**: PyTorch was installed without CUDA support, or the CUDA drivers are not properly configured.

**Solutions**:
1. Verify CUDA drivers are installed: `nvidia-smi`
2. Install the CUDA-compatible PyTorch build:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```
3. Check CUDA version: `nvcc --version`
4. The code will automatically fall back to CPU if CUDA is not available:
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```

---

### `nvidia.com/gpu` resource request fails in Kubernetes

**Cause**: The node does not have GPU resources available, or the NVIDIA device plugin is not installed.

**Solutions**:
1. Check node GPU availability: `kubectl describe nodes | grep -i nvidia`
2. Verify the NVIDIA device plugin is running
3. Ensure your job's `nodeSelector` matches an available GPU node type (e.g., `NVIDIA-RTX-A6000`)
4. Check tolerations are set correctly for GPU nodes

---

## FFmpeg and Preprocessing Issues

### `RuntimeError: ffmpeg executable not found`

**Cause**: FFmpeg is not installed, not in `PATH`, or the `FFMPEG_BIN` environment variable is not set.

**Solutions**:
1. Install ffmpeg (see [Setup and Deployment](Setup-and-Deployment) for OS-specific instructions)
2. Set the `FFMPEG_BIN` environment variable:
   ```bash
   export FFMPEG_BIN=/usr/bin/ffmpeg
   ```
3. Pass the ffmpeg path explicitly:
   ```bash
   aceverify-preprocess input.zip subfolder --ffmpeg-bin /usr/bin/ffmpeg
   ```

---

### `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`

**Cause**: FFmpeg is not installed on your system.

**Solution**: Install ffmpeg:
```bash
# Ubuntu/Debian
sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg

# Conda
conda install -c conda-forge ffmpeg
```

---

### `subprocess.CalledProcessError` during frame extraction

**Cause**: The ffmpeg command failed, possibly due to a corrupt video file or an unsupported codec.

**Solutions**:
1. Verify the video file is not corrupt: `ffmpeg -i input.mp4`
2. Check if the video codec is supported by your ffmpeg version
3. Ensure the video has at least 5 seconds of content (frames are extracted at 5s offset)
4. Run with `--log-level DEBUG` for detailed error information

---

### `ValueError: No label json file found in zip`

**Cause**: The preprocessing pipeline expects a JSON file within the zip archive that maps video filenames to labels. No `.json` file was found.

**Solution**: Ensure your zip archive contains a JSON metadata file in DFDC format:
```json
{
    "video_file_1.mp4": {"label": "FAKE"},
    "video_file_2.mp4": {"label": "REAL"}
}
```

---

### No face detected during preprocessing

**Cause**: The MTCNN face detector failed to find a face in any of the 16 extracted frames. This can happen if:
- The video does not contain a visible face
- The face is too small or partially occluded
- The 5-second offset skips past the face-containing portion

**Solution**: The video is automatically skipped with a warning:
```
WARNING | No face detected in extracted frames for video_name
```

To handle this, ensure your training data contains clear face videos.

---

## File Format and Upload Issues

### `ValueError: Unsupported file extension: .xxx`

**Cause**: The uploaded or preprocessed file has an extension that is not in the allowed list.

**Allowed extensions** (from `utilities/static_media.py:16`):
- **Video**: `.mp4`, `.mov`, `.avi`
- **Image**: `.jpg`, `.jpeg`, `.png`

**Solution**: Convert your file to a supported format before uploading.

---

### `ValueError: Empty filename` or `ValueError: Invalid filename`

**Cause**: The `sanitize_filename()` function (in `utilities/static_media.py:61`) rejects filenames that are empty, contain only unsafe characters, or attempt path traversal.

**Solution**: Use a valid filename with no path separators, no null bytes, and no leading dots.

---

### Uploaded video does not play in the media preview

**Cause**: Streamlit versions ≤ 1.50 force non-allowlisted extensions (including `.mp4`) to `Content-Type: text/plain`, which prevents `<video>` playback.

**Solution**: The application automatically calls `utilities.ensure_static_video_mime()` on startup to patch this behavior. If the issue persists:
1. Verify `server.enableStaticServing = true` in `.streamlit/config.toml`
2. Restart the Streamlit server after changing the config
3. Check if your Streamlit version is very old or very new (the MIME patching handles common cases)

---

### `ValueError: Refusing to write outside the uploads directory`

**Cause**: The `save_upload_bytes()` function (in `utilities/static_media.py:124`) performs a path-safety check that rejects any file path outside the `frontend/static/uploads/` directory.

**Solution**: This is a security feature. The upload path is always resolved under the uploads directory to prevent path traversal attacks.

---

## Streamlit and Web App Issues

### Web app shows an empty state ("UPLOAD A FILE TO BEGIN ANALYSIS")

**Cause**: This is the expected empty state when no media has been uploaded or selected. The application displays this when `st.session_state.active_file_path` is `None`.

**Solution**: Upload a file or select an example media clip to begin analysis.

---

### `st.rerun()` loop / page keeps refreshing

**Cause**: If state is not properly managed, clicking example media buttons or the analyze button can cause repeated reruns.

**Solution**: The application is designed to handle this:
- Example media clicks use `st.rerun()` intentionally to update the UI immediately
- The `st.session_state.analyzed` flag prevents re-running analysis unintentionally
- Upload file deduplication uses `(name, size)` signatures to avoid reprocessing

If you are experiencing a loop, clear your browser cache and reload the page.

---

### Analysis fails with "Could not decode video frames"

**Cause**: The `FaceProcessor.extract_frames()` method (in `utilities/preprocess.py:121`) returned zero frames, meaning `cv2.VideoCapture` failed to decode the video.

**Solutions**:
1. Verify the video file is not corrupt
2. Check if the video codec is supported by OpenCV
3. Ensure the video file is a valid video (not a renamed image)

---

### Grad-CAM heatmap shows all black

**Cause**: The Grad-CAM computation may have failed to capture activations or gradients from the model's last convolutional layer.

**Solution**: Check that the model architecture has a Conv2d layer that can be hooked. The `generate_gradcam()` function (in `utilities/gradcam.py:7`) searches for the last `Conv2d` layer in the model hierarchy.

---

## Docker and Container Issues

### `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`

**Cause**: The NVIDIA Container Toolkit is not installed on the host machine.

**Solution**:
1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. Restart the Docker daemon: `sudo systemctl restart docker`
3. Alternatively, run without GPU access by removing the `--gpus all` flag

---

### Container starts but the web app is not accessible

**Cause**: Port mapping or network configuration issue.

**Solutions**:
1. Verify port mapping: `docker run -p 8501:8501 ...`
2. Check if the Streamlit server is listening on `0.0.0.0`:
   ```bash
   docker exec <container_id> curl -s http://0.0.0.0:8501
   ```
3. The Dockerfile sets `--server.address=0.0.0.0` to bind to all interfaces

---

### `pip install -e . --no-deps` fails inside Docker build

**Cause**: The `pyproject.toml` may not be correctly positioned in the build context, or package directories are not found.

**Solution**: The Dockerfile copies `aceverify/pyproject.toml` to the root as `pyproject.toml`:
```dockerfile
COPY aceverify/pyproject.toml ./pyproject.toml
COPY aceverify ./aceverify
```
This ensures `pip install -e .` can find both the project configuration and the package source.

---

## Kubernetes and NRP Issues

### `kubectl wait --for=condition=complete job/aceverify-train` times out

**Cause**: The training job is taking longer than the specified timeout, or the job failed to start.

**Solutions**:
1. Increase the timeout: `--timeout=72h`
2. Check the job status: `kubectl describe job aceverify-train`
3. Check pod logs: `kubectl logs -l job-name=aceverify-train`
4. Verify the PVC is bound and accessible

---

### `scripts/copy-to-pvc.sh` fails with "pod unready"

**Cause**: The ephemeral Alpine pod failed to start or mount the PVC.

**Solutions**:
1. Check PVC status: `kubectl get pvc aceverify-pvc`
2. Ensure the PVC is bound: `status: Bound`
3. Verify storage class availability: `kubectl get storageclass rook-ceph-block`
4. Manually create the pod and check events for error details

---

### Training job cannot find the HDF5 file

**Cause**: The data file was not staged to the PVC, or the path in the job manifest is incorrect.

**Solutions**:
1. Verify the data file is on the PVC:
   ```bash
   bash scripts/copy-from-pvc.sh aceverify-pvc /workspace/data ./out
   ```
2. Ensure the `--train_path` and `--test_path` in the job manifest point to `/workspace/data/...`
3. Run `bash scripts/copy-to-pvc.sh ./ aceverify-pvc /workspace` again to stage all files

---

## Model and Training Issues

### `KeyError: 'model_state_dict'` during checkpoint loading

**Cause**: The checkpoint file format is not a dict with a `model_state_dict` key.

**Solution**: The code handles both formats:
```python
checkpoint = torch.load(checkpoint_path, map_location=device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
```
If the checkpoint is a raw state dict, it is loaded directly. The `FileNotFoundError` exception is caught and logs a warning, continuing from pretrained ViT weights.

---

### `ValueError: Expected both real(0) and fake(1) labels` during training

**Cause**: The HDF5 dataset contains only one label class (all real or all fake), making balanced sampling impossible.

**Solution**: Ensure your HDF5 dataset contains both real (label=0) and fake (label=1) samples. The error message includes the found counts:
```
Expected both real(0) and fake(1) labels in path, found real=N, fake=M
```

---

### `ValueError: No samples selected from path`

**Cause**: The data loading function selected zero samples, possibly due to an empty HDF5 file or a very small `n` value.

**Solution**: Check that the HDF5 file contains data and the `n` parameter is sufficient:
```
Check file contents and requested n=N.
```

---

### `ValueError: Empty training loader for epoch N`

**Cause**: The training data loader returned zero batches for a given epoch.

**Solution**: Verify that:
1. The HDF5 training file exists and is accessible
2. The dataset contains samples at the requested indices
3. The batch size is not larger than the dataset size

---

### `ValueError: No validation samples were evaluated in epoch N`

**Cause**: The validation loop did not process any samples.

**Solution**: Verify that:
1. The HDF5 test file exists and is accessible
2. The test dataset contains samples
3. The `test_path` argument is correct

---

## Frequently Asked Questions

### What dataset is used for training?

The project uses the [Deepfake Detection Challenge (DFDC)](https://www.kaggle.com/competitions/deepfake-detection-challenge/data) dataset from Kaggle. The preprocessing pipeline (`aceverify/preprocess.py`) converts DFDC-style zip archives into HDF5 files with aligned face crops and audio features.

---

### How do I download the pre-trained model checkpoint?

The model checkpoint is available on Google Drive:
- **URL**: [https://drive.google.com/file/d/1d3ln2laSfmXkKyXHZ1YhK_gb33nonaPO/view?usp=sharing](https://drive.google.com/file/d/1d3ln2laSfmXkKyXHZ1YhK_gb33nonaPO/view?usp=sharing)

The web application (`app/services.py`) automatically downloads the checkpoint to `app/aceverify_final.pth` if it does not already exist, using the `gdown` library.

---

### Can I use the web app without a GPU?

Yes. The web application runs on CPU by default. GPU access improves inference throughput but is not required for the application to function.

---

### What is the difference between the three model options?

| Model | Architecture | Strengths | Weaknesses |
|---|---|---|---|
| EfficientNet-B4 (Fast) | `timm tf_efficientnet_b4` | Fast inference, low VRAM | Less accurate on complex fakes |
| XceptionNet (Accurate) | `timm xception` | Accurate spatial detection | No temporal modeling |
| ACE.verify (Best) | ViT-B/16 + Bi-GRU + audio fusion | Multimodal, temporal, best overall | Highest VRAM and compute requirements |

---

### Why does the face alignment use MediaPipe instead of MTCNN?

The preprocessing pipeline (`aceverify/preprocess.py`) uses **MTCNN** for face detection during dataset preparation (sufficient for one-time alignment). The web application (`utilities/preprocess.py`) uses **MediaPipe Face Landmarker** for real-time face alignment because it provides 68 facial landmarks, enabling more precise alignment based on eye corner positions (landmarks 33 and 263).

---

### What do the evidence flags mean?

The `evidence_from_regions()` function (in `utilities/gradcam.py:168`) maps Grad-CAM region scores to interpretable evidence categories:

| Evidence Flag | Formula | Meaning |
|---|---|---|
| Eye-blink anomaly | `Periocular × 1.10` | Unnatural blinking patterns (face swap) |
| Lip-sync mismatch | `Mouth × 1.10` | Audio-video desynchronization (lip-sync deepfakes) |
| Texture inconsistency | `(Periocular + Forehead) / 2` | Blending boundary artifacts |
| Compression artifacts | `(Forehead + Chin) / 2` | Double-compression artifacts |
| Head-pose jitter | `(Periocular + Chin) / 2` | Unstable head rotation |
| Skin-tone boundary | `(Mouth + Chin) / 2` | Color mismatch at face boundaries |

Each flag is color-coded in the UI: green (< 35%), amber (35&ndash;60%), red (> 60%).

---

### How are frames sampled from videos?

The `ACEDataset.__getitem__` method (`dataset.py:25`) samples 16 frames with a stride of 2:

```python
num_output_frames = 16
stride = 2
indices = np.arange(0, num_output_frames * stride, stride)
# = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
```

If the video has fewer frames than needed, linear interpolation is used instead.

---

### How do I contribute to the project?

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Ensure code passes existing tests and linting.
4. Write clear commit messages following conventional commits.
5. Open a Pull Request against the `main` branch.

---

### What is the license?

The project is licensed under the MIT License.

---

### Are the results a substitute for expert forensic analysis?

No. The footer of the web application explicitly states:

> ACE.verify &middot; Powered by AI &middot; Results are probabilistic and for research use only
> Not a substitute for expert forensic analysis
