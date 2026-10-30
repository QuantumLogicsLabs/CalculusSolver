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
            assert weights[idx].item() == 0.0
        else:
            assert weights[idx].item() > 0.0, f"{token} has no weight"
    counts = report["token_counts"]
    assert counts[NUMERICAL] > 0 and counts[VARIABLE] > 0 and counts[STRUCTURAL] > 0
    assert report["mapped_ids"] == len(set(vocab.values()))


def test_weights_follow_tokens_not_ids():
    """Permuting vocabulary ids must move the weights with the tokens."""
    vocab = {"[PAD]": 0, "STRUCT:OPEN": 1, "COEF:3": 2, "VAR:x": 3, "OP:diff": 4}
    shuffled = {"[PAD]": 0, "STRUCT:OPEN": 4, "COEF:3": 3, "VAR:x": 2, "OP:diff": 1}
    w1, _ = build_token_weight_vector(vocab, 5)
    w2, _ = build_token_weight_vector(shuffled, 5)
    for token in vocab:
        assert w1[vocab[token]].item() == pytest.approx(w2[shuffled[token]].item())
    assert w2[2].item() == 4.0 and w2[3].item() == 3.5


def test_conflicting_class_on_one_id_is_rejected():
    with pytest.raises(ValueError):
        build_token_weight_vector({"VAR:x": 1, "COEF:3": 1}, 3)


def test_non_positive_or_unknown_class_weights_rejected():
    with pytest.raises(ValueError):
        build_token_weight_vector({"VAR:x": 1}, 2, {"variable": 0.0})
    with pytest.raises(ValueError):
        build_token_weight_vector({"VAR:x": 1}, 2, {"bogus": 2.0})


# -- loss behaviour -----------------------------------------------------------

def test_uniform_weights_equal_plain_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(4, 6, 9)
    targets = torch.randint(1, 9, (4, 6))
    targets[:, -2:] = 0  # padding
    ours = TokenWeightedCrossEntropy(9, ignore_index=0)(logits, targets)
    ref = nn.CrossEntropyLoss(ignore_index=0)(logits.reshape(-1, 9), targets.reshape(-1))
    assert ours.item() == pytest.approx(ref.item(), rel=1e-6)


def test_weighted_loss_matches_manual_computation():
    logits = torch.tensor([[2.0, 0.5, 0.1], [0.2, 1.5, 0.3], [0.1, 0.2, 2.5]])
    targets = torch.tensor([1, 2, 0])  # last position is padding
    weights = torch.tensor([0.0, 1.0, 4.0])
    loss = TokenWeightedCrossEntropy(3, ignore_index=0, token_weights=weights)(logits, targets)

    nll = -torch.log_softmax(logits, dim=-1)
    expected = (1.0 * nll[0, 1] + 4.0 * nll[1, 2]) / (1.0 + 4.0)
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)


def test_heavy_class_dominates_the_mean():
