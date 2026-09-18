"""Compatibility shim: re-exports the one canonical model class.

train.py and predict.py import CalculusSolverModel from here. This module used
to wrap that import in try/except and, on ImportError, silently define an
entirely different LSTM-based CalculusSolverModel -- so a broken
model/transformer.py trained the wrong network with no error raised. The
import is now unconditional: if model/transformer.py cannot be imported, this
fails loudly at import time, before any training starts.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).parent.resolve())
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

# Hard import, deliberately no fallback. See module docstring.
from model.transformer import CalculusSolverModel, check_forward_contract  # noqa: E402

__all__ = ["CalculusSolverModel", "check_forward_contract"]
