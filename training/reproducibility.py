"""Reproducibility helpers: seeding and a per-run manifest.

The manifest records everything needed to re-create or audit a checkpoint:
seeds, dataset + tokenizer versions (with hashes), checkpoint path/hash,
optimizer and scheduler settings, loss configuration, and the software /
hardware environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch


def set_global_seed(seed: int, deterministic: bool = True) -> Dict:
    """Seed python, numpy (if present) and torch (CPU + CUDA); return what was set."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    numpy_seeded = False
    try:
        import numpy as np

        np.random.seed(seed)
        numpy_seeded = True
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Needed for deterministic cuBLAS; must be set before the first CUDA matmul.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    return {
        "seed": seed,
        "python_random": True,
        "numpy": numpy_seeded,
        "torch": True,
        "cuda": torch.cuda.is_available(),
        "deterministic_algorithms": deterministic,
    }


def sha256_file(path) -> Optional[str]:
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path) -> Optional[int]:
    p = Path(path)
    if not p.is_file():
        return None
    with p.open("rb") as stream:
        return sum(1 for _ in stream)


def describe_files(paths: Iterable) -> Dict[str, Dict]:
    out = {}
    for path in paths:
        p = Path(path)
        out[p.as_posix()] = {
            "exists": p.is_file(),
            "sha256": sha256_file(p),
            "rows": count_lines(p) if p.suffix == ".jsonl" else None,
        }
    return out


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def collect_environment() -> Dict:
    packages = {}
    for name in ("torch", "numpy", "sympy"):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            packages[name] = None
    cuda_available = torch.cuda.is_available()
    status = _git("status", "--porcelain")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": packages,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else "CPU",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
    }


def build_run_manifest(
    *,
    run_name: str,
    cfg: Dict,
    seed_info: Dict,
    dataset_files: Iterable,
    vocab_path: str,
    serializer_path: str,
    vocab_size: int,
    checkpoint_path: str,
    optimizer_info: Dict,
    scheduler_info: Dict,
    loss_info: Dict,
    extra: Optional[Dict] = None,
