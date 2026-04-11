# =============================================================
# Ultra Doc-Intelligence — Universal Dockerfile
# Supports both CPU-only and NVIDIA GPU environments.
#
# Build for CPU (default):
#   docker build -t ultra-doc .
#   docker build -t ultra-doc --build-arg ENABLE_GPU=false .
#
# Build for GPU (NVIDIA CUDA):
#   docker build -t ultra-doc-gpu --build-arg ENABLE_GPU=true .
#
# Run CPU:
#   docker run -p 7860:7860 -v $(pwd)/models:/app/models ultra-doc
#
# Run GPU:
#   docker run --gpus all -p 7860:7860 -v $(pwd)/models:/app/models ultra-doc-gpu
# =============================================================

ARG ENABLE_GPU=false

# ── CPU base ──────────────────────────────────────────────────
FROM python:3.11-slim AS base-false

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    tesseract-ocr tesseract-ocr-eng \
    libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Standard CPU build of llama-cpp-python
RUN pip install --no-cache-dir llama-cpp-python

ENV N_GPU_LAYERS=0

# ── GPU base ──────────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS base-true

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    build-essential cmake git \
    tesseract-ocr tesseract-ocr-eng \
    libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 \
    && pip install --upgrade pip

# CUDA-accelerated build of llama-cpp-python
RUN CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
    pip install --no-cache-dir llama-cpp-python

ENV N_GPU_LAYERS=-1

# ── Select base from build arg ─────────────────────────────────
FROM base-${ENABLE_GPU}

WORKDIR /app

# ── Python dependencies (skip llama-cpp-python, already installed) ──
COPY requirements.txt .
RUN grep -v "^llama-cpp-python" requirements.txt \
    | pip install --no-cache-dir -r /dev/stdin

# ── Application code ──
COPY backend/ ./backend/
COPY tessdata/ ./tessdata/
COPY api.py app.py ./

# ── Runtime directories (mounted as volumes) ──
RUN mkdir -p uploads chroma_db models

# ── Environment ──
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "api.py"]
