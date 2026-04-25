"""
Configuration module for Ultra Doc-Intelligence.

Device is detected automatically at startup:
  CUDA available  →  GPU path (int4 weight-only via torchao, ~3 GB VRAM)
  No CUDA         →  CPU path (int4 weight-only via torchao)

Override via environment variable:
  FORCE_DEVICE=cpu   force CPU even on a GPU machine
  FORCE_DEVICE=cuda  assert GPU (will error if CUDA unavailable)

Performance tuning for 6 GB VRAM (RTX 4050):
  max_new_tokens      = 512   (RAG answers — rarely need more)
  extract_max_tokens  = 1024  (JSON extraction — structured output)
  extract_max_chars   = 6000  (document chars fed to extractor — halved)
  top_k               = 3     (fewer retrieved chunks = smaller prompt = faster)
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _detect_device() -> str:
    forced = os.getenv("FORCE_DEVICE", "").lower()
    if forced in ("cpu", "cuda"):
        return forced
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class AppConfig:
    """Central configuration for the application."""

    # --- LLM (Gemma 3 4B via HuggingFace Transformers) ---
    # Local GGUF takes priority; falls back to HF Hub if file not found.
    model_id: str = "google/gemma-4-E2B-it"
    local_model_dir: str = os.path.join("models", "gemma-3-4b-it-GGUF")
    gguf_file: str = "gemma-3-4b-it-Q4_K_M.gguf"
    device: str = field(default_factory=_detect_device)
    max_new_tokens: int = 512        # RAG answers — concise; 512 is plenty
    extract_max_tokens: int = 1024   # JSON extraction — more structured output
    extract_max_chars: int = 6000    # Max document chars fed into extractor prompt

    # --- Embedding ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Document Processing ---
    chunk_size: int = 800
    chunk_overlap: int = 100

    # --- RAG / Retrieval ---
    top_k: int = 3                   # Fewer chunks = smaller prompt = faster inference
    similarity_threshold: float = 0.15

    # --- Guardrails ---
    confidence_weights: dict = field(default_factory=lambda: {
        "similarity": 0.40,
        "agreement": 0.30,
        "coverage": 0.30,
    })
    min_confidence_to_answer: float = 0.10

    # --- Storage ---
    chroma_persist_dir: str = "./chroma_db"
    upload_dir: str = "./uploads"


config = AppConfig()
