# ACE.verify

ACE.verify is a deepfake detection project for analyzing facial media with a multimodal (video + audio) model, preprocessing utilities, and an interactive deployed webapp. The repository includes the training pipeline, dataset preprocessing scripts, visualization helpers, and saved experiment artifacts for baseline and ablation runs.

## Overview

The project is organized around three core workflows:

1. Preprocess raw DFDC-style videos into HDF5 datasets with aligned face crops and audio features.
2. Train and evaluate ACE.verify models on processed HDF5 data.
3. Inspect predictions through a deployed webapp interface with Grad-CAM overlays and optional face-landmark visualization.

## Dataset

The experiments in this repository were built around the Kaggle Deepfake Detection Challenge dataset:

[Deepfake Detection Challenge](https://www.kaggle.com/competitions/deepfake-detection-challenge/data)

The preprocessing pipeline expects raw videos and label metadata in the DFDC format, then converts them into HDF5 files for training and evaluation.

## Repository Layout

- `aceverify/`: training package, dataset definitions, preprocessing entry points, and model code.
- `frontend/`: deployed webapp for interactive inference and visualization.
- `models/`: model wrappers and backbone implementations used by the demo application.
- `utilities/`: Grad-CAM, preprocessing helpers, timeline rendering, and shared model utilities.
- `data/`: sample HDF5 files and trained checkpoints.
- `scripts/`: notebooks, ablation outputs, and generated experiment artifacts.
- `media/`: sample videos used by the demo.

## Requirements

- Python 3.10 or newer, up to Python 3.12 as declared in `pyproject.toml`.
- `ffmpeg` available on your system `PATH` for preprocessing videos.
- A working PyTorch installation compatible with your hardware.

The repository includes Conda environment files if you prefer managing dependencies through Conda:

- `conda_env.yml`
- `conda_env_new.yml`

## Installation

From a fresh clone, the simplest setup is:

```bash
conda env create -f conda_env_new.yml
conda activate aceverify
pip install -e .
```

If you are not using Conda, create a Python environment that satisfies the requirements in `pyproject.toml`, install the dependencies, and then install the project in editable mode with `pip install -e .`.

## Docker

The repository ships a CUDA-ready Dockerfile for local development and NRP-style deployment. It installs the project dependencies, exposes the Streamlit app on port 8501, and sets `PYTHONPATH=/workspace` so the package entrypoints and module imports work inside the container.

Build the image from the repository root:

```bash
docker build -t aceverify:latest .
```

Run the web app:

```bash
docker run --rm -it \
	--gpus all \
	-p 8501:8501 \
	-v "$PWD":/workspace \
	aceverify:latest
```

That command starts Streamlit on `http://localhost:8501` and uses the current checkout as the working tree, so local edits are visible inside the container.

Run training inside the container:

```bash
docker run --rm -it \
	--gpus all \
	-v "$PWD":/workspace \
	aceverify:latest \
	aceverify-train \
	--train_path /workspace/data/train_data-003.h5 \
	--test_path /workspace/data/test_data.h5 \
	--checkpoint-path /workspace/results/aceverify_final.pth
```

Run evaluation inside the container:

```bash
docker run --rm -it \
	--gpus all \
	-v "$PWD":/workspace \
	aceverify:latest \
	aceverify-evaluate \
	--h5 /workspace/data/test_data.h5 \
	--checkpoint /workspace/results/aceverify_final.pth
```

Run preprocessing inside the container:

```bash
docker run --rm -it \
	-v "$PWD":/workspace \
	aceverify:latest \
	aceverify-preprocess <zip_file> <subfolder> --output /workspace/data/processed_data.h5
```

If you are targeting NRP, build the image locally or in a registry-backed CI job, push it to your registry, and reference that image name in `assets/templates/nrp-gpu-job.yaml`.

## Preprocessing Data

The preprocessing entry point converts DFDC-style zip archives into HDF5 files by extracting frames, detecting faces, cropping to a consistent region, and storing paired video and audio data.

```bash
aceverify-preprocess <zip_file> <subfolder> --output processed_data.h5
```

Example:

```bash
aceverify-preprocess dfdc_train_part_00.zip dfdc_train_part_0 --output data/processed_data.h5
```

Useful options:

- `--temp-dir`: override the temporary extraction directory.
- `--ffmpeg-bin`: point to a specific `ffmpeg` executable.
- `--log-level`: set preprocessing verbosity.

## Training

The training pipeline expects separate HDF5 files for training and testing:

```bash
aceverify-train --train_path data/train_data.h5 --test_path data/test_data.h5
```

Training uses the `ACEVerifyModel` architecture defined in `aceverify/model.py`, which combines a pretrained vision backbone with temporal sequence modeling for video-level classification.

The script saves the final checkpoint to `aceverify_final.pth` and writes per-epoch metrics to a matching CSV file.

## Visualization

To inspect dataset samples and verify preprocessing output, use the visualization helper:

```bash
python aceverify/visualize_data.py --h5 data/train_data.h5 --index 0
```

This displays sampled frames and the associated spectrogram for the selected record.

## Deployed Webapp

Launch the interactive app from the repository root:

```bash
streamlit run frontend/app.py
```

The demo supports:

- Video and image uploads.
- Example media loaded from `media/`.
- Model selection across EfficientNet-B4, XceptionNet, and ACE.verify variants.
- Grad-CAM heatmaps and optional face-landmark overlays.

## Outputs and Artifacts

- Trained checkpoints are stored under `data/trained_model_paths/` and `scripts/` for experiment-specific runs.
- Ablation results, metric CSV files, and saved `.pth` weights are collected in `scripts/`.
- Generated visualizations and temporary preprocessing files are kept outside version control or in the designated `temp` directory when preprocessing runs.

## Notes

- Run all commands from the repository root unless otherwise noted.
- Ensure `ffmpeg` is installed before preprocessing video archives.
- If you are using a GPU, install the PyTorch build that matches your CUDA stack.

For questions about the original dataset, refer to the Kaggle competition page linked above.