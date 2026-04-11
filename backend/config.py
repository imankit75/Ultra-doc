"""
Configuration module for Ultra Doc-Intelligence.
Uses Gemma 3 4B locally via llama-cpp-python (GGUF format).
No LM Studio, no cloud APIs — fully offline.

Environment variables (set in Docker or .env):
  N_GPU_LAYERS  — number of layers to offload to GPU.
                  0  = CPU only (default, works everywhere)
                  -1 = ALL layers on GPU (fastest, needs NVIDIA GPU)
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    """Central configuration for the application."""

    # --- LLM (Gemma 3 4B GGUF) ---
    model_path: str = os.path.join("models", "gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf")
    n_ctx: int = 8192
    # Read from env: 0 = CPU, -1 = full GPU. Defaults to 0 (safe for all environments).
    n_gpu_layers: int = int(os.getenv("N_GPU_LAYERS", "0"))
    max_tokens: int = 2048

    # --- Embedding ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Document Processing ---
    chunk_size: int = 800
    chunk_overlap: int = 100

    # --- RAG / Retrieval ---
    top_k: int = 5
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


# Singleton config instance
config = AppConfig()

