#!/usr/bin/env python3
"""
Ultra Doc-Intelligence — Smart Environment Setup

Detects CPU vs GPU, selects the correct PyTorch CUDA wheel, and installs all
dependencies in one command.

Usage:
    python install.py
"""

import re
import subprocess
import sys
import platform


# Map (min_cuda_major, min_cuda_minor) → PyTorch wheel tag, ordered newest first.
# CUDA drivers are backward-compatible, so a CUDA 13.0 driver can run cu128 wheels.
_CUDA_WHEEL_MAP = [
    (12, 8, "cu128"),
    (12, 6, "cu126"),
    (12, 4, "cu124"),
    (12, 1, "cu121"),
    (11, 8, "cu118"),
]
_TORCH_INDEX_BASE = "https://download.pytorch.org/whl"
_TORCH_CPU_URL = f"{_TORCH_INDEX_BASE}/cpu"


def _nvidia_smi_output() -> str:
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        return r.stdout if r.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _get_cuda_version() -> tuple[int, int] | None:
    """Parse 'CUDA Version: X.Y' from nvidia-smi output."""
    smi = _nvidia_smi_output()
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", smi)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _has_nvidia_gpu() -> bool:
    """True if nvidia-smi finds any GPU."""
    smi = _nvidia_smi_output()
    return bool(smi.strip())


def _cuda_available_in_torch() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _best_cuda_tag(major: int, minor: int) -> str:
    """Return the newest PyTorch CUDA wheel tag compatible with this driver."""
    for req_major, req_minor, tag in _CUDA_WHEEL_MAP:
        if major > req_major or (major == req_major and minor >= req_minor):
            return tag
    return "cu121"  # safe minimum


def _pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])


def main() -> None:
    print(f"Platform : {platform.system()} {platform.machine()} | Python {sys.version.split()[0]}")

    gpu = _has_nvidia_gpu() or _cuda_available_in_torch()
    cuda_ver = _get_cuda_version() if gpu else None

    print(f"NVIDIA GPU: {'detected' if gpu else 'not found'}")
    if cuda_ver:
        print(f"CUDA driver: {cuda_ver[0]}.{cuda_ver[1]}")

    print()
    _pip("--upgrade", "pip", "--quiet")

    if gpu:
        tag = _best_cuda_tag(*cuda_ver) if cuda_ver else "cu121"
        torch_url = f"{_TORCH_INDEX_BASE}/{tag}"
        print(f"Backend  : GPU  — bitsandbytes 4-bit NF4 via CUDA")
        print(f"Torch    : CUDA wheel ({tag}) from {torch_url}")
        print()

        # Remove any CPU-only torch wheel first; pip won't upgrade +cpu → +cuXXX
        # automatically because both satisfy the >=2.6.0 constraint.
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "torch", "-y"],
            capture_output=True,
        )
        _pip("torch>=2.6.0", "--extra-index-url", torch_url)
        _pip("-r", "requirements_gpu.txt")
    else:
        print(f"Backend  : CPU  — optimum-quanto int4")
        print()
        _pip("-r", "requirements_cpu.txt")

    print()
    print("Done! All dependencies installed.")
    print()
    if gpu:
        print("GPU: Gemma 3 4B will load with 4-bit NF4 quantization (bitsandbytes).")
    else:
        print("CPU: Gemma 3 4B will load with int4 quantization (optimum-quanto).")
    print()
    print("First run will download Gemma 3 4B from HuggingFace (~2.5 GB).")
    print("If you hit a 401 error, set HF_TOKEN=<your_token> in a .env file.")


if __name__ == "__main__":
    main()
