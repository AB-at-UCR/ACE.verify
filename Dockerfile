FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace

# Install system dependencies (including OpenCV & Mediapipe dependencies libgl1 and libglib2.0-0)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY .streamlit ./.streamlit
COPY aceverify/pyproject.toml ./pyproject.toml
COPY aceverify ./aceverify
COPY evaluation ./evaluation
COPY frontend ./frontend
COPY models ./models
COPY utilities ./utilities
COPY README.md ./README.md
COPY README_NRP.md ./README_NRP.md

# Install Python dependencies with matching PyTorch ecosystem versions
RUN pip install --upgrade pip \
    && pip install \
        numpy \
        h5py \
        timm \
        scikit-learn \
        matplotlib \
        ffmpeg-python \
        facenet-pytorch \
        Pillow \
        scipy \
        streamlit \
        opencv-python-headless \
        mediapipe \
        sqlalchemy \
        dataset \
        alembic \
        torchaudio==2.5.1 \
        torchvision==0.20.1 \
    && pip install -e . --no-deps

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.enableStaticServing=true"]