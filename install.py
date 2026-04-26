#!/usr/bin/env python3
"""
Ultra Doc-Intelligence — Smart Environment Setup

Detects your hardware (NVIDIA GPU, Apple Silicon, AMD, or CPU-only),
measures available RAM and VRAM, selects the right PyTorch wheel,
installs all dependencies, and verifies every critical import works.

Usage:
    python install.py
"""

import re
import subprocess
import sys
import os
import platform

# Force UTF-8 output on Windows (default cp1252 breaks non-ASCII characters)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# Map (min_cuda_major, min_cuda_minor) → PyTorch wheel tag, ordered newest first.
_CUDA_WHEEL_MAP = [
    (12, 8, "cu128"),
    (12, 6, "cu126"),
    (12, 4, "cu124"),
    (12, 1, "cu121"),
    (11, 8, "cu118"),
]
_TORCH_INDEX_BASE = "https://download.pytorch.org/whl"
_TORCH_CPU_URL    = f"{_TORCH_INDEX_BASE}/cpu"


# ---------------------------------------------------------------------------
# Hardware detection helpers
# ---------------------------------------------------------------------------

def _nvidia_smi_output() -> str:
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        return r.stdout if r.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _get_cuda_version() -> tuple[int, int] | None:
    smi = _nvidia_smi_output()
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", smi)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _has_nvidia_gpu() -> bool:
    return bool(_nvidia_smi_output().strip())


def _get_vram_gb() -> float:
    """Parse total VRAM in GB from nvidia-smi."""
    smi = _nvidia_smi_output()
    # Matches "MiB / 6144MiB" style lines
    m = re.search(r"/\s*(\d+)\s*MiB", smi)
    if m:
        return int(m.group(1)) / 1024
    return 0.0


def _has_rocm() -> bool:
    """Detect AMD ROCm-capable GPU via rocm-smi."""
    try:
        r = subprocess.run(["rocm-smi", "--showid"], capture_output=True, text=True, timeout=5)
        return r.returncode == 0 and "GPU" in r.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _detect_apple_silicon() -> bool:
    """True on Apple M-series chips (arm64 macOS)."""
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def _get_total_ram_gb() -> float:
    """Get total system RAM in GB via psutil (must be installed first)."""
    try:
        import psutil
        return psutil.virtual_memory().total / 1024**3
    except Exception:
        pass
    # Fallback for platforms where psutil install failed
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / 1024**2
        elif platform.system() == "Darwin":
            r = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, timeout=5)
            return int(r.stdout.strip()) / 1024**3
    except Exception:
        pass
    return 0.0


def _cuda_available_in_torch() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _best_cuda_tag(major: int, minor: int) -> str:
    for req_major, req_minor, tag in _CUDA_WHEEL_MAP:
        if major > req_major or (major == req_major and minor >= req_minor):
            return tag
    return "cu121"  # safe minimum


# ---------------------------------------------------------------------------
# Pip helper
# ---------------------------------------------------------------------------

def _pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *args])


# ---------------------------------------------------------------------------
# Model tier recommendation
# ---------------------------------------------------------------------------

def _recommend_model(device_type: str, vram_gb: float, ram_gb: float) -> str:
    MAIN  = "google/gemma-4-E2B-it"
    SMALL = "google/gemma-3-1b-it"
    if device_type == "cuda":
        return MAIN if vram_gb >= 4.0 else SMALL
    if device_type == "mps":
        return MAIN  # unified memory
    # CPU
    return MAIN if ram_gb >= 8.0 else SMALL


# ---------------------------------------------------------------------------
# HF token check
# ---------------------------------------------------------------------------

def _check_env_file():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            if "HF_TOKEN" in f.read():
                return
    if os.environ.get("HF_TOKEN", ""):
        return

    print()
    print("=" * 60)
    print("  WARNING: HF_TOKEN not found")
    print("=" * 60)
    print("  Gemma models are gated on HuggingFace.")
    print("  Create a .env file in this folder with:")
    print()
    print("    HF_TOKEN=hf_your_token_here")
    print()
    print("  Get your token : https://huggingface.co/settings/tokens")
    print("  Accept license : https://huggingface.co/google/gemma-4-E2B-it")
    print("  Smaller model  : https://huggingface.co/google/gemma-3-1b-it")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Import verification
# ---------------------------------------------------------------------------

def _verify_imports(gpu: bool, mps: bool):
    print()
    print("=" * 60)
    print("  VERIFYING IMPORTS")
    print("=" * 60)

    checks = [
        ("torch",                 "import torch; assert torch.__version__"),
        ("torch CUDA",            "import torch; assert torch.cuda.is_available()" if gpu else None),
        ("torch MPS",             "import torch; assert torch.backends.mps.is_available()" if mps else None),
        ("transformers",          "import transformers; assert transformers.__version__"),
        ("accelerate",            "import accelerate"),
        ("torchao",               "import torchao"),
        ("bitsandbytes",          "import bitsandbytes" if gpu else None),
        ("gguf",                  "import gguf"),
        ("sentence-transformers", "from sentence_transformers import SentenceTransformer"),
        ("chromadb",              "import chromadb"),
        ("gradio",                "import gradio"),
        ("PyMuPDF (fitz)",        "import fitz"),
        ("python-docx",           "import docx"),
        ("pillow",                "from PIL import Image"),
        ("python-dotenv",         "from dotenv import load_dotenv"),
        ("python-dateutil",       "import dateutil"),
        ("psutil",                "import psutil"),
    ]

    all_passed = True
    for name, code in checks:
        if code is None:
            print(f"  [SKIP] {name:30s} (not needed on this platform)")
            continue
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  [ OK ] {name}")
        else:
            err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
            print(f"  [FAIL] {name:30s} — {err}")
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("  All imports OK!")
    else:
        print("  Some imports failed — re-run install.py to retry.")
    print("=" * 60)
    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Platform : {platform.system()} {platform.machine()} | Python {sys.version.split()[0]}")
    print()

    # Install psutil first (tiny, no CUDA deps) so we can read RAM accurately
    _pip("psutil", "-q")

    # Detect hardware
    has_nvidia = _has_nvidia_gpu() or _cuda_available_in_torch()
    has_rocm   = _has_rocm()
    has_mps    = _detect_apple_silicon()
    cuda_ver   = _get_cuda_version() if has_nvidia else None
    vram_gb    = _get_vram_gb() if has_nvidia else 0.0
    ram_gb     = _get_total_ram_gb()

    # Determine device type
    if has_nvidia:
        device_type = "cuda"
    elif has_mps:
        device_type = "mps"
    else:
        device_type = "cpu"

    # Print system summary
    print("=" * 60)
    print("  SYSTEM SUMMARY")
    print("=" * 60)
    print(f"  RAM        : {ram_gb:.1f} GB total")
    if has_nvidia:
        cuda_str = f"{cuda_ver[0]}.{cuda_ver[1]}" if cuda_ver else "unknown"
        print(f"  GPU        : NVIDIA detected  (CUDA {cuda_str})")
        if vram_gb > 0:
            print(f"  VRAM       : {vram_gb:.1f} GB")
    elif has_rocm:
        print(f"  GPU        : AMD ROCm detected")
    elif has_mps:
        print(f"  GPU        : Apple Silicon (MPS)")
    else:
        print(f"  GPU        : None detected — CPU mode")

    recommended_model = _recommend_model(device_type, vram_gb, ram_gb)
    print(f"  Model tier : {recommended_model}")
    print("=" * 60)
    print()

    # Warn if RAM is too low
    if ram_gb > 0 and ram_gb < 4.0:
        print("  WARNING: Less than 4 GB RAM detected.")
        print("  Ultra Doc may run slowly or fail on very low-memory systems.")
        print("  Recommended: 8 GB RAM minimum for a good experience.")
        print()

    _pip("--upgrade", "pip")

    # -------------------------------------------------------------------
    # Install dependencies based on device type
    # -------------------------------------------------------------------

    if has_nvidia:
        tag       = _best_cuda_tag(*cuda_ver) if cuda_ver else "cu121"
        torch_url = f"{_TORCH_INDEX_BASE}/{tag}"
        print(f"Backend  : NVIDIA GPU — bitsandbytes NF4 4-bit quantization")
        print(f"Torch    : CUDA wheel ({tag}) from {torch_url}")
        print()

        _pip("-r", "requirements_gpu.txt")

        # Force CUDA torch wheel — pip won't upgrade +cpu → +cuXXX automatically
        print(f"\nPinning CUDA torch ({tag}) - overriding any CPU wheel from deps...")
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "torch", "-y"],
            capture_output=True,
        )
        _pip("torch>=2.6.0", "--extra-index-url", torch_url)

    elif has_mps:
        print(f"Backend  : Apple Silicon (MPS) — float16, no quantization needed")
        print()
        _pip("-r", "requirements_cpu.txt")
        # torchao is still useful for potential CPU fallback
        # bitsandbytes is NOT installed — MPS is unsupported

    elif has_rocm:
        print(f"Backend  : AMD ROCm — CPU fallback (ROCm wheels require manual setup)")
        print(f"  NOTE: For best AMD GPU performance, install PyTorch ROCm manually.")
        print(f"  See: https://pytorch.org/get-started/locally/")
        print()
        _pip("-r", "requirements_cpu.txt")

    else:
        print(f"Backend  : CPU — torchao int8 weight-only quantization")
        print()
        _pip("-r", "requirements_cpu.txt")

    print()
    print("Done! All dependencies installed.")
    print()

    passed = _verify_imports(gpu=has_nvidia, mps=has_mps)

    print()
    _check_env_file()

    # Summary
    print("=" * 60)
    print("  NEXT STEPS")
    print("=" * 60)
    if has_nvidia:
        print(f"  Device : NVIDIA GPU (CUDA) — NF4 4-bit quantization")
    elif has_mps:
        print(f"  Device : Apple Silicon (MPS) — float16")
    else:
        print(f"  Device : CPU — int8 weight-only quantization")
    print(f"  Model  : {recommended_model}")
    if recommended_model == "google/gemma-3-1b-it":
        print(f"  Note   : Low-spec mode — using 1B model for your hardware")
        print(f"           Override: set ULTRA_DOC_MODEL=google/gemma-4-E2B-it")
    print()
    print("  First run will download the model from HuggingFace (~2-5 GB).")
    print("  Ensure HF_TOKEN is set in a .env file.")
    print()
    if passed:
        print("  Ready! Run:  python api.py")
    else:
        print("  Fix the failed imports above, then run:  python api.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
