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
