FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY aceverify/pyproject.toml ./pyproject.toml
COPY aceverify ./aceverify
COPY ops ./ops
COPY frontend ./frontend
COPY models ./models
COPY utilities ./utilities
COPY media ./media
COPY README.md ./README.md
COPY README_NRP.md ./README_NRP.md

RUN pip install --upgrade pip \
    && pip install numpy h5py timm scikit-learn matplotlib ffmpeg-python facenet-pytorch Pillow scipy streamlit \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
    && pip install -e . --no-deps

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
