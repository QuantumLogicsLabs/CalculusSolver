"""Task 2b: hard repeat guards in beam search.

Two hard masks, not score penalties -- a degenerate beam must be unable to
continue, not merely pay for it:

  max_token_run=4        cap on consecutive identical tokens. Real data never
                         exceeds 3 (only STRUCT:CLOSE reaches it, in 30,000
                         targets), so 4 leaves margin.
  no_repeat_ngram_size=2 block a content bigram this beam already produced.
                         Scoped to all-content n-grams: 77.4% of real targets
                         repeat SOME bigram, but ZERO of the 155,000 repeat a
                         content-only one. Unscoped, this default would
                         corrupt three quarters of all legitimate answers.
"""

import itertools

import pytest

torch = pytest.importorskip("torch", reason="beam_search imports torch")

from inference.beam_search import _repeat_blocked_tokens, beam_search  # noqa: E402
from inference.grammar import load_vocab  # noqa: E402
from tests.unit.slang_builders import MULTI_TERM, SIMPLE  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VOCAB = load_vocab(str(ROOT / "tokenizer" / "vocab.json"))
VOCAB_SIZE = max(VOCAB["id_to_token"]) + 1
TID = VOCAB["token_to_id"]
CONTENT = frozenset(
    i for i, t in VOCAB["id_to_token"].items()
    if not t.startswith(("STRUCT:", "NODE:", "["))
)
EXEMPT = frozenset({TID["[EOS]"], TID["STRUCT:CLOSE"]})


# ── the helper, in isolation ─────────────────────────────────────────────────

def test_run_cap_blocks_the_tail_at_k():
    seq = [TID["OPVAR:x"]] * 4
    assert TID["OPVAR:x"] in _repeat_blocked_tokens(seq, 4, 0, EXEMPT, CONTENT)


def test_run_cap_allows_below_k():
    seq = [TID["OPVAR:x"]] * 3
    assert _repeat_blocked_tokens(seq, 4, 0, EXEMPT, CONTENT) == set()


def test_run_cap_never_blocks_struct_close():
    """STRUCT:CLOSE legitimately runs to 3 and is how a beam terminates."""
    seq = [TID["STRUCT:CLOSE"]] * 9
    assert TID["STRUCT:CLOSE"] not in _repeat_blocked_tokens(seq, 2, 0, EXEMPT, CONTENT)


def test_run_cap_never_blocks_eos():
    seq = [TID["[EOS]"]] * 9
    assert TID["[EOS]"] not in _repeat_blocked_tokens(seq, 2, 0, EXEMPT, CONTENT)


def test_run_cap_disabled_by_zero():
    seq = [TID["OPVAR:x"]] * 20
    assert _repeat_blocked_tokens(seq, 0, 0, EXEMPT, CONTENT) == set()


# ── n-gram guard ─────────────────────────────────────────────────────────────

def test_ngram_blocks_a_repeated_content_bigram():
    seq = [TID["VAR:u"], TID["EXP:-3"], TID["VAR:u"]]
    assert TID["EXP:-3"] in _repeat_blocked_tokens(seq, 0, 2, EXEMPT, CONTENT)


def test_ngram_ignores_structural_bigrams():
    """NODE:TERM COEF:1 repeats in numerator and denominator of any fraction.
    Guarding it would break ordinary answers."""
    seq = [TID["NODE:TERM"], TID["COEF:1"], TID["STRUCT:CLOSE"],
           TID["NODE:TERM"]]
    assert TID["COEF:1"] not in _repeat_blocked_tokens(seq, 0, 2, EXEMPT, CONTENT)


def test_ngram_disabled_by_zero():
    seq = [TID["VAR:u"], TID["EXP:-3"], TID["VAR:u"]]
    assert _repeat_blocked_tokens(seq, 0, 0, EXEMPT, CONTENT) == set()


# ── end to end ───────────────────────────────────────────────────────────────

class Looper:
    """Adversarial: wants one cycle and nothing else."""

    def __call__(self, src_tokens, tgt):
        logits = torch.zeros(1, tgt.shape[1], VOCAB_SIZE)
        for token in ("OP:lagrange", "OPVAR:x", "VAR:u", "EXP:-3"):
            logits[0, -1, TID[token]] = 15.0
        return (logits, None, None)


def _run(model, max_len=32, **kwargs):
    result = beam_search(
        model=model, src_tokens=torch.zeros(1, 8, dtype=torch.long),
        vocab_map=VOCAB, beam_size=5, max_len=max_len, **kwargs,
    )
    strings = [VOCAB["id_to_token"][t] for t in result["tokens"]]
    if strings and strings[0] == "[BOS]":
        strings = strings[1:]
    longest = max((sum(1 for _ in g) for _, g in itertools.groupby(strings)), default=0)
    return result, strings, longest


def test_guard_bounds_a_degenerate_run():
    """Defence is layered, and the layers are independent.

    The grammar now caps an op-node's OPVAR run at MAX_OPVARS_PER_OP, so even
    with the beam guards disabled this run is bounded -- the unbounded 31-long
    run that motivated this work is no longer reachable at all. The beam guard
    tightens it further, and remains the only defence for repeats the grammar
    permits (non-consecutive reuse, cross-node cycles).
    """
    from inference.grammar import MAX_OPVARS_PER_OP

    _, _, unguarded = _run(Looper(), max_token_run=0, no_repeat_ngram_size=0)
    _, _, guarded = _run(Looper())
    assert unguarded <= MAX_OPVARS_PER_OP   # grammar bound alone
    assert guarded <= 2                     # guard tightens it further
    assert guarded < unguarded              # the guard is doing real work


@pytest.mark.parametrize("k", [2, 3, 4, 6])
def test_run_cap_is_respected_for_any_k(k):
    _, _, longest = _run(Looper(), max_token_run=k, no_repeat_ngram_size=0)
    assert longest <= k


# ── the guards must not harm correct answers ─────────────────────────────────

def _oracle(gold_tokens):
    gold = [TID[t] for t in gold_tokens + ["[EOS]"]]

    class Oracle:
        def __call__(self, src_tokens, tgt):
            step = tgt.shape[1] - 1
            logits = torch.full((1, tgt.shape[1], VOCAB_SIZE), -8.0)
            if step < len(gold):
                logits[0, -1, gold[step]] = 20.0
            return (logits, None, None)

    return Oracle()


@pytest.mark.parametrize("ast", [SIMPLE, MULTI_TERM], ids=["simple", "multi_term"])
def test_guards_do_not_break_correct_answers(ast):
    """Guards on by default -- a correct answer must be unaffected."""
    result, strings, _ = _run(_oracle(ast), max_len=64)
    assert result["status"] == "solved"
    assert strings == ast
