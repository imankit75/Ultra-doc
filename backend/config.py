"""
Configuration module for Ultra Doc-Intelligence.

Device is detected automatically at startup:
  CUDA available            →  GPU path (NF4 4-bit via bitsandbytes, ~3 GB VRAM)
  Apple Silicon (MPS)       →  MPS path (float16, no quantization)
  No CUDA / No MPS          →  CPU path (int8 weight-only via torchao)

Auto model selection (overridable via ULTRA_DOC_MODEL env var):
  GPU with >= 4 GB VRAM     →  google/gemma-4-E2B-it  (2B)
  GPU with < 4 GB VRAM      →  google/gemma-3-1b-it   (1B)
  MPS                       →  google/gemma-4-E2B-it  (2B — unified memory)
  CPU with >= 5 GB free RAM →  google/gemma-4-E2B-it  (2B)
  CPU with < 5 GB free RAM  →  google/gemma-3-1b-it   (1B)

Environment overrides:
  FORCE_DEVICE=cpu|cuda|mps   — bypass auto-detection
  ULTRA_DOC_MODEL=<hf-id>     — use any HuggingFace model ID
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _detect_device() -> str:
    forced = os.getenv("FORCE_DEVICE", "").lower()
    if forced in ("cpu", "cuda", "mps"):
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"


def _available_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1024**3
    except Exception:
        return 16.0


def _total_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
        return 0.0
    except Exception:
        return 0.0


def _auto_select_model(device: str) -> str:
    override = os.getenv("ULTRA_DOC_MODEL", "")
    if override:
        return override

    MAIN  = "google/gemma-4-E2B-it"   # ~2B params
    SMALL = "google/gemma-3-1b-it"    # ~1B params — low-spec fallback

    if device == "cuda":
        return SMALL if _total_vram_gb() < 4.0 else MAIN
    if device == "cpu":
        return SMALL if _available_ram_gb() < 5.0 else MAIN
    return MAIN  # MPS: unified memory, 2B fits in 8 GB+


@dataclass
class AppConfig:
    device: str = field(default_factory=_detect_device)

    # Model — empty string triggers auto-selection in __post_init__
    model_id: str = ""
    local_model_dir: str = os.path.join("models", "gemma-3-4b-it-GGUF")
    gguf_file: str = "gemma-3-4b-it-Q4_K_M.gguf"

    # Inference token limits — auto-reduced on low-RAM machines
    max_new_tokens: int = 512
    extract_max_tokens: int = 1024
    extract_max_chars: int = 6000

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"

    # Document processing
    chunk_size: int = 800
    chunk_overlap: int = 100

    # RAG / Retrieval
    top_k: int = 3
    similarity_threshold: float = 0.15

    # Guardrails
    confidence_weights: dict = field(default_factory=lambda: {
        "similarity": 0.40,
        "agreement": 0.30,
        "coverage": 0.30,
    })
    min_confidence_to_answer: float = 0.10

    # Storage
    chroma_persist_dir: str = "./chroma_db"
    upload_dir: str = "./uploads"

    def __post_init__(self):
        if not self.model_id:
            self.model_id = _auto_select_model(self.device)
        # Reduce generation limits on very low-RAM CPU machines
        if self.device == "cpu" and _available_ram_gb() < 5.0:
            self.max_new_tokens = min(self.max_new_tokens, 256)
            self.extract_max_tokens = min(self.extract_max_tokens, 512)
            self.extract_max_chars = min(self.extract_max_chars, 3000)


config = AppConfig()
