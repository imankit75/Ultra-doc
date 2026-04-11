# =============================================================
# Ultra Doc-Intelligence — CPU Dockerfile
# Runs fully on CPU (no GPU required).
# Usage:
#   docker build -t ultra-doc .
#   docker run -p 7860:7860 -v $(pwd)/models:/app/models ultra-doc
# =============================================================

FROM python:3.11-slim

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python dependencies ----
# Install llama-cpp-python for CPU first (no CUDA flags)
RUN pip install --no-cache-dir llama-cpp-python

# Install remaining dependencies
COPY requirements.txt .
RUN grep -v "^llama-cpp-python" requirements.txt \
    | pip install --no-cache-dir -r /dev/stdin

# ---- Application code ----
COPY backend/ ./backend/
COPY tessdata/ ./tessdata/
COPY api.py app.py ./

# ---- Runtime directories (models + data are mounted as volumes) ----
RUN mkdir -p uploads chroma_db models

# ---- Environment ----
ENV KMP_DUPLICATE_LIB_OK=TRUE
# 0 = CPU only. Override with -e N_GPU_LAYERS=-1 for GPU
ENV N_GPU_LAYERS=0
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["python", "api.py"]
