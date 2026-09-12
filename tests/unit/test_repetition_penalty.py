"""Task 1: CTRL-style soft repetition penalty.

Applied to raw logits BEFORE the grammar mask and before log_softmax, so the
softmax renormalises with the penalty already in place.

The design constraint that matters is scope, not strength. Structural tokens
repeat constantly and legitimately -- a minimal fraction carries STRUCT:OPEN
and STRUCT:CLOSE four times each -- so a global penalty would degrade every
multi-term answer. And STRUCT:CLOSE / [EOS] must never be made less
attractive: that is precisely the failure this change set exists to fix.

min_count=4 comes from data: across all 155,000 targets the most any single
content token occurs within one target is 4, and only in 4 targets. The
penalty therefore cannot fire on a correct answer.
"""

import collections
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="beam_search imports torch")

from inference.beam_search import _apply_repetition_penalty, beam_search  # noqa: E402
from inference.grammar import load_vocab  # noqa: E402
from tests.unit.slang_builders import MULTI_TERM, SIMPLE  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VOCAB = load_vocab(str(ROOT / "tokenizer" / "vocab.json"))
VOCAB_SIZE = max(VOCAB["id_to_token"]) + 1
TID = VOCAB["token_to_id"]
CONTENT = frozenset(
    i for i, t in VOCAB["id_to_token"].items()
    if not t.startswith(("STRUCT:", "NODE:", "["))
)
EXEMPT = frozenset({TID["[EOS]"], TID["STRUCT:CLOSE"]})

OPVAR_X, VAR_U = TID["OPVAR:x"], TID["VAR:u"]
CLOSE, EOS, TERM = TID["STRUCT:CLOSE"], TID["[EOS]"], TID["NODE:TERM"]


def _logits(**by_id):
    vec = torch.zeros(VOCAB_SIZE)
    for token_id, value in by_id.items():
        vec[int(token_id)] = value
    return vec


def _penalise(logits, tokens, penalty=1.2, min_count=4):
    return _apply_repetition_penalty(
        logits, tokens, penalty, min_count, EXEMPT, CONTENT
    )


# ── the transform ────────────────────────────────────────────────────────────

def test_positive_logit_is_divided():
    out = _penalise(_logits(**{str(OPVAR_X): 5.0}), [OPVAR_X] * 4)
    assert out[OPVAR_X] == pytest.approx(5.0 / 1.2)


def test_negative_logit_is_multiplied():
    """The sign branch is what keeps the transform discouraging for both
    signs -- dividing a negative logit would make it MORE attractive."""
    out = _penalise(_logits(**{str(VAR_U): -5.0}), [VAR_U] * 4)
    assert out[VAR_U] == pytest.approx(-5.0 * 1.2)
    assert out[VAR_U] < -5.0


# ── the escape hatch must never be penalised ─────────────────────────────────

@pytest.mark.parametrize("token_id,name", [(CLOSE, "STRUCT:CLOSE"), (EOS, "[EOS]")])
def test_escape_hatch_is_never_penalised(token_id, name):
    out = _penalise(_logits(**{str(token_id): 3.0}), [token_id] * 9)
    assert out[token_id] == pytest.approx(3.0), f"{name} was penalised"


def test_structural_tokens_are_never_penalised():
    """NODE:TERM recurs once per term in any multi-term answer."""
    out = _penalise(_logits(**{str(TERM): 3.0}), [TERM] * 9)
    assert out[TERM] == pytest.approx(3.0)


# ── gating ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seen,fires", [(1, False), (2, False), (3, False),
                                        (4, True), (7, True)])
def test_min_count_gate(seen, fires):
    out = _penalise(_logits(**{str(OPVAR_X): 5.0}), [OPVAR_X] * seen)
    changed = abs(float(out[OPVAR_X]) - 5.0) > 1e-9
    assert changed is fires


def test_penalty_of_one_is_a_noop():
    out = _penalise(_logits(**{str(OPVAR_X): 5.0}), [OPVAR_X] * 9, penalty=1.0)
    assert out[OPVAR_X] == pytest.approx(5.0)


def test_empty_history_is_a_noop():
    out = _penalise(_logits(**{str(OPVAR_X): 5.0}), [])
    assert out[OPVAR_X] == pytest.approx(5.0)


# ── must not touch correct answers ───────────────────────────────────────────

@pytest.mark.parametrize("ast", [SIMPLE, MULTI_TERM], ids=["simple", "multi_term"])
def test_penalty_cannot_fire_on_a_known_good_answer(ast):
    counts = collections.Counter(TID[t] for t in ast if TID[t] in CONTENT)
    assert max(counts.values()) < 4


def test_real_targets_stay_below_the_threshold():
    """Guards the premise. If a future generator emits a target reusing one
    content token 4+ times, the penalty would start firing on correct
    answers -- fail loudly here instead."""
    path = ROOT / "data" / "splits" / "test.jsonl"
    if not path.exists():
        pytest.skip("data/splits/test.jsonl not present")
    from tokenizer.slang_serializer import serialize_slang_math

    worst = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ast = json.loads(line).get("tgt_output_tokens")
            if ast is None:
                continue
            tokens = [t for t in serialize_slang_math(ast)
                      if not t.startswith(("STRUCT:", "NODE:", "["))]
            if tokens:
                worst = max(worst, max(collections.Counter(tokens).values()))
    assert worst <= 4, f"a real target reuses one content token {worst} times"


@pytest.mark.parametrize("ast", [SIMPLE, MULTI_TERM], ids=["simple", "multi_term"])
def test_oracle_still_solves_with_the_penalty_on(ast):
    gold = [TID[t] for t in ast + ["[EOS]"]]

    class Oracle:
        def __call__(self, src_tokens, tgt):
            step = tgt.shape[1] - 1
            logits = torch.full((1, tgt.shape[1], VOCAB_SIZE), -8.0)
            if step < len(gold):
                logits[0, -1, gold[step]] = 20.0
            return (logits, None, None)

    result = beam_search(
        model=Oracle(), src_tokens=torch.zeros(1, 8, dtype=torch.long),
        vocab_map=VOCAB, beam_size=5, max_len=64,
    )
    strings = [VOCAB["id_to_token"][t] for t in result["tokens"]]
    if strings and strings[0] == "[BOS]":
        strings = strings[1:]
    assert result["status"] == "solved"
    assert strings == ast
