"""Interface smoke tests for the model <-> training/inference boundary.

Four separate crashes traced back to callers treating
CalculusSolverModel.forward() as returning a single tensor, and to model
classes/signatures drifting apart. These are cheap (one tiny forward pass) and
catch that class of mismatch in seconds instead of hours into a training run.
train.py runs the same check_forward_contract() before its first step.
"""

import importlib

import pytest

torch = pytest.importorskip("torch")

from model.transformer import CalculusSolverModel, check_forward_contract  # noqa: E402

VOCAB_SIZE = 40
NUM_RULES = 5


def _tiny_model():
    torch.manual_seed(0)
    return CalculusSolverModel(
        vocab_size=VOCAB_SIZE, num_rules=NUM_RULES,
        hidden_dim=16, num_heads=2, num_layers=1, ffn_dim=32, dropout=0.0,
    ).eval()


def _dummy_batch(batch=3, src_len=6, tgt_len=4):
    src = torch.randint(1, VOCAB_SIZE, (batch, src_len))
    tgt = torch.randint(1, VOCAB_SIZE, (batch, tgt_len))
    return src, tgt


# -- the 3-tuple return contract ---------------------------------------------

def test_forward_returns_a_three_tuple():
    src, tgt = _dummy_batch()
    with torch.no_grad():
        output = _tiny_model()(src, tgt)
    assert isinstance(output, tuple) and len(output) == 3


def test_forward_output_shapes():
    src, tgt = _dummy_batch(batch=3, tgt_len=4)
    with torch.no_grad():
        decoder_logits, rule_logits, verifier_logits = _tiny_model()(src, tgt)
    assert decoder_logits.shape == (3, 4, VOCAB_SIZE)
    assert rule_logits.shape == (3, NUM_RULES)
    assert verifier_logits.shape == (3, 1)


def test_forward_accepts_teacher_forced_rule_ids():
    src, tgt = _dummy_batch(batch=2)
    with torch.no_grad():
        output = _tiny_model()(src, tgt, true_rule_ids=torch.tensor([0, 3]))
    assert len(output) == 3


def test_check_forward_contract_passes_for_the_canonical_model():
    check_forward_contract(_tiny_model(), vocab_size=VOCAB_SIZE, num_rules=NUM_RULES)


def test_check_forward_contract_restores_training_mode():
    model = _tiny_model().train()
    check_forward_contract(model, vocab_size=VOCAB_SIZE, num_rules=NUM_RULES)
    assert model.training


def test_check_forward_contract_rejects_a_single_tensor_return():
    class SingleTensorModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, src_seq, tgt_in_seq):
            return torch.zeros(src_seq.size(0), tgt_in_seq.size(1), VOCAB_SIZE)

    with pytest.raises(TypeError, match="3-tuple"):
        check_forward_contract(SingleTensorModel(), vocab_size=VOCAB_SIZE, num_rules=NUM_RULES)


def test_check_forward_contract_rejects_wrong_shapes():
    model = _tiny_model()
    with pytest.raises(ValueError, match="rule_logits"):
        check_forward_contract(model, vocab_size=VOCAB_SIZE, num_rules=NUM_RULES + 1)


# -- one canonical class, no silent fallback ---------------------------------

def test_solver_model_reexports_the_canonical_class():
    """solver_model.py used to define an LSTM model on ImportError. It must
    now be the exact same class as model/transformer.py."""
    import solver_model

    assert solver_model.CalculusSolverModel is CalculusSolverModel


def test_solver_model_has_no_fallback_path():
    """Structural check, not a string search -- the module docstring still
    describes the old fallback. No try/except and no class definitions: the
    module may only re-export."""
    import ast

    source = importlib.util.find_spec("solver_model").loader.get_source("solver_model")
    tree = ast.parse(source)
    assert not any(isinstance(node, ast.Try) for node in ast.walk(tree))
    assert not any(isinstance(node, ast.ClassDef) for node in ast.walk(tree))


def test_retired_architecture_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("model.architecture")


# -- every model beam_search can receive honours the contract ----------------

def test_legacy_pkl_model_honours_the_contract():
    from inference.solve import PklTransformerModel

    model = PklTransformerModel(vocab_size=VOCAB_SIZE, hidden_dim=16).eval()
    src, tgt = _dummy_batch()
    with torch.no_grad():
        output = model(src, tgt, true_rule_ids=None)
    assert isinstance(output, tuple) and len(output) == 3
    assert output[0].shape == (3, 4, VOCAB_SIZE)


def test_beam_search_runs_against_the_real_model_class():
    """End to end through beam_search with an actual CalculusSolverModel,
    not a stub -- the path the tuple-index crash was found on."""
    from pathlib import Path

    from inference.beam_search import beam_search
    from inference.grammar import load_vocab

    root = Path(__file__).resolve().parents[2]
    vocab = load_vocab(str(root / "tokenizer" / "vocab.json"))
    vocab_size = max(vocab["id_to_token"]) + 1
    num_rules = sum(1 for t in vocab["token_to_id"] if t.startswith("RULE:"))
    model = CalculusSolverModel(
        vocab_size=vocab_size, num_rules=num_rules,
        hidden_dim=16, num_heads=2, num_layers=1, ffn_dim=32, dropout=0.0,
    ).eval()
    with torch.no_grad():
        result = beam_search(
            model, torch.randint(1, vocab_size, (1, 8)), vocab, beam_size=2, max_len=6,
        )
    assert result["status"] in ("solved", "partial")
    assert isinstance(result["tokens"], list)


def test_beam_search_signature_has_no_compat_parameters():
    import inspect

    from inference.beam_search import beam_search

    params = inspect.signature(beam_search).parameters
    assert "src_positions" not in params
    assert "parent_child_pairs" not in params
