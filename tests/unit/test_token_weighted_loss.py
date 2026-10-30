"""Token-class weighted loss: class mapping, vocabulary coverage, and safety."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from training.token_weights import (
    DEFAULT_CLASS_WEIGHTS,
    NUMERICAL,
    OTHER,
    PADDING,
    STRUCTURAL,
    VARIABLE,
    TokenWeightedCrossEntropy,
    build_token_weight_vector,
    classify_token,
)

ROOT = Path(__file__).resolve().parents[2]


def _real_vocab():
    raw = json.loads((ROOT / "tokenizer" / "vocab.json").read_text(encoding="utf-8"))
    flat = {}
    for key, value in raw.items():
        if not key.startswith("_") and isinstance(value, dict):
            flat.update(value)
    return flat


# -- class mapping ------------------------------------------------------------

@pytest.mark.parametrize(
    "token,expected",
    [
        ("STRUCT:NUMI", STRUCTURAL),
        ("NODE:TERM", STRUCTURAL),
        ("NODE:GRADIENT", STRUCTURAL),
        ("COEF:-7", NUMERICAL),
        ("EXP:3", NUMERICAL),
        ("VAR:x", VARIABLE),
        ("OPVAR:xy", VARIABLE),
        ("[PAD]", PADDING),
        ("OP:partial", OTHER),
        ("RULE:power_rule", OTHER),
        ("POINT:4", OTHER),
        ("[EOS]", OTHER),
    ],
)
def test_token_classes(token, expected):
    assert classify_token(token) == expected


def test_default_weights_match_spec():
    assert DEFAULT_CLASS_WEIGHTS[STRUCTURAL] == 1.0
    assert DEFAULT_CLASS_WEIGHTS[NUMERICAL] == 3.5
    assert DEFAULT_CLASS_WEIGHTS[VARIABLE] == 4.0


def test_real_vocab_weights_follow_token_class():
    vocab = _real_vocab()
    size = max(vocab.values()) + 1
    weights, report = build_token_weight_vector(vocab, size)
    for token, idx in vocab.items():
        cls = classify_token(token)
        expected = 0.0 if cls == PADDING else DEFAULT_CLASS_WEIGHTS[cls]
        assert weights[idx].item() == pytest.approx(expected), token


def test_weights_cover_active_vocabulary():
    """Every id that carries a token gets a positive weight, except [PAD]."""
    vocab = _real_vocab()
    size = max(vocab.values()) + 1
    weights, report = build_token_weight_vector(vocab, size)
    pad_id = vocab["[PAD]"]
    for token, idx in vocab.items():
        if idx == pad_id:
