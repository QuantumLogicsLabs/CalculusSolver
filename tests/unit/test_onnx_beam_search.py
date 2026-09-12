"""The ONNX mirror must behave identically to the torch beam search.

deployment/ has no other test coverage, and the two implementations had
drifted: every termination and repetition fix landed on the torch side only,
so the ONNX path still discarded finished beams and could never emit [EOS].

The session is duck-typed -- only .run() is called -- so these drive it with
a stub and need no exported .onnx model (there isn't one in the repo).
"""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("onnxruntime", reason="ONNX deployment path dependency")
torch = pytest.importorskip("torch", reason="needed for the parity comparison")

from deployment.onnx_beam_search import _log_softmax, onnx_beam_search  # noqa: E402
from inference.beam_search import beam_search  # noqa: E402
from inference.grammar import EOS_TOKEN, load_vocab  # noqa: E402
from tests.unit.slang_builders import GRADIENT_2D, MULTI_TERM, SIMPLE  # noqa: E402
from tokenizer.slang_serializer import deserialize_slang_math  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VOCAB = load_vocab(str(ROOT / "tokenizer" / "vocab.json"))
VOCAB_SIZE = max(VOCAB["id_to_token"]) + 1
TID = VOCAB["token_to_id"]


class OracleSession:
    """Stub onnxruntime session: certain about every gold token."""

    def __init__(self, gold_ids):
        self.gold_ids = list(gold_ids)

    def run(self, output_names, feeds):
        tgt = feeds["tgt_in_seq"]
        step = tgt.shape[1] - 1
        logits = np.full((1, tgt.shape[1], VOCAB_SIZE), -8.0, dtype=np.float32)
        if step < len(self.gold_ids):
            logits[0, -1, self.gold_ids[step]] = 20.0
        return [logits]


class OracleModel:
    """The torch equivalent of OracleSession, for the parity test."""

    def __init__(self, gold_ids):
        self.gold_ids = list(gold_ids)

    def __call__(self, src_tokens, tgt):
        step = tgt.shape[1] - 1
        logits = torch.full((1, tgt.shape[1], VOCAB_SIZE), -8.0)
        if step < len(self.gold_ids):
            logits[0, -1, self.gold_ids[step]] = 20.0
        return (logits, None, None)


def _gold(ast):
    return [TID[t] for t in ast + [EOS_TOKEN]]


def _run_onnx(ast, **kwargs):
    result = onnx_beam_search(
        session=OracleSession(_gold(ast)),
        src_tokens=[0] * 8,
        vocab_map=VOCAB,
        beam_size=5,
        **kwargs,
    )
    strings = [VOCAB["id_to_token"][t] for t in result["tokens"]]
    if strings and strings[0] == "[BOS]":
        strings = strings[1:]
    return result, strings


def _run_torch(ast, **kwargs):
    result = beam_search(
        model=OracleModel(_gold(ast)),
        src_tokens=torch.zeros(1, 8, dtype=torch.long),
        vocab_map=VOCAB,
        beam_size=5,
        **kwargs,
    )
    strings = [VOCAB["id_to_token"][t] for t in result["tokens"]]
    if strings and strings[0] == "[BOS]":
        strings = strings[1:]
    return result, strings


# ── the masked-logit hazard (numpy only) ─────────────────────────────────────

def test_log_softmax_preserves_minus_inf():
    """np.log(softmax(x) + 1e-12) mapped masked tokens to a FINITE -27.6,
    so argpartition could select them -- and [EOS] is a low id, so that
    manufactured finished beams holding a truncated AST."""
    x = np.array([1.0, -np.inf, 2.0, -np.inf])
    out = _log_softmax(x)
    assert np.isneginf(out[1]) and np.isneginf(out[3])
    assert np.isfinite(out[0]) and np.isfinite(out[2])


def test_log_softmax_normalises():
    x = np.array([1.0, -np.inf, 2.0])
    assert np.exp(_log_softmax(x)[[0, 2]]).sum() == pytest.approx(1.0)


def test_log_softmax_all_masked():
    assert np.all(np.isneginf(_log_softmax(np.array([-np.inf, -np.inf]))))


# ── termination ──────────────────────────────────────────────────────────────

def test_onnx_reaches_solved():
    result, _ = _run_onnx(SIMPLE, max_len=32)
    assert result["status"] == "solved"


def test_onnx_returns_gold_exactly():
    _, strings = _run_onnx(SIMPLE, max_len=32)
    assert strings == SIMPLE


def test_onnx_strips_the_terminator():
    """onnx_solve.py strips [BOS] and RULE: but never [EOS]."""
    _, strings = _run_onnx(SIMPLE, max_len=32)
    assert EOS_TOKEN not in strings


def test_onnx_output_deserializes():
    _, strings = _run_onnx(SIMPLE, max_len=32)
    assert deserialize_slang_math(strings)["numi"]["terms"] == [
        {"coeff": 6, "var": {"x": 1}}
    ]


def test_onnx_score_is_finite():
    result, _ = _run_onnx(SIMPLE, max_len=32)
    assert np.isfinite(result["score"])


def test_onnx_gradient_is_reachable():
    result, strings = _run_onnx(GRADIENT_2D, max_len=64)
    assert result["status"] == "solved"
    assert strings == GRADIENT_2D


# ── parity: the actual anti-drift guard ──────────────────────────────────────

@pytest.mark.parametrize(
    "ast,max_len",
    [(SIMPLE, 32), (MULTI_TERM, 64), (GRADIENT_2D, 64)],
    ids=["simple", "multi_term", "gradient"],
)
def test_onnx_matches_torch(ast, max_len):
    onnx_result, onnx_strings = _run_onnx(ast, max_len=max_len)
    torch_result, torch_strings = _run_torch(ast, max_len=max_len)
    assert onnx_strings == torch_strings
    assert onnx_result["status"] == torch_result["status"]
    assert onnx_result["score"] == pytest.approx(torch_result["score"], abs=1e-4)


@pytest.mark.parametrize("beam_size", [1, 2, 5])
def test_parity_across_beam_widths(beam_size):
    onnx_result = onnx_beam_search(
        session=OracleSession(_gold(SIMPLE)), src_tokens=[0] * 8,
        vocab_map=VOCAB, beam_size=beam_size, max_len=32,
    )
    torch_result = beam_search(
        model=OracleModel(_gold(SIMPLE)),
        src_tokens=torch.zeros(1, 8, dtype=torch.long),
        vocab_map=VOCAB, beam_size=beam_size, max_len=32,
    )
    assert onnx_result["tokens"] == torch_result["tokens"]


# ── the bundle-size invariant ────────────────────────────────────────────────

def test_onnx_path_never_imports_torch():
    """The ONNX path exists solely to keep torch out of the Vercel bundle
    (~250MB serverless cap). inference/decoding.py is shared with the torch
    beam search, so a stray `import torch` there would silently break the
    deployment. Run in a subprocess because pytest itself has torch loaded.
    """
    import subprocess
    import sys

    probe = (
        "import sys;"
        "import deployment.onnx_beam_search, deployment.onnx_solve;"
        "import inference.decoding, inference.grammar;"
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=str(ROOT), capture_output=True
    )
    assert result.returncode == 0, (
        "torch leaked into the ONNX deployment path:\n"
        + result.stderr.decode(errors="replace")
    )
