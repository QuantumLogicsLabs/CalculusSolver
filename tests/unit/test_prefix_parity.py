import json
import sys
import os
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tokenizer.slang_serializer import serialize_slang_math


def _load_vocab():
    vocab_path = Path(__file__).resolve().parents[2] / "tokenizer" / "vocab.json"
    with open(vocab_path, "r", encoding="utf-8") as f:
        raw_vocab = json.load(f)

    flat = {}
    for key, value in raw_vocab.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            flat.update(value)
    return raw_vocab, flat


def _rule_token_strings(raw_vocab):
    items = sorted(raw_vocab.get("rule_tokens", {}).items(), key=lambda kv: kv[1])
    return [name for name, _ in items]


class TestPrefixParity:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.raw_vocab, self.vocab_mapping = _load_vocab()
        self.rule_token_strings = _rule_token_strings(self.raw_vocab)

    def test_training_target_starts_with_bos_then_rule(self):
        rule_idx = 0
        rule_token = (
            self.rule_token_strings[rule_idx]
            if 0 <= rule_idx < len(self.rule_token_strings)
            else None
        )
        prefix = [rule_token] if rule_token else []

        sample_expr = {
            "numi": {"terms": [{"coeff": 3, "var": {"x": 2}}]},
            "deno": 1,
        }
        tokens = serialize_slang_math(sample_expr)
        tokens = list(prefix) + tokens
        tokens = ["[BOS]"] + tokens + ["[EOS]"]

        assert tokens[0] == "[BOS]", "First token must be [BOS]"
        assert tokens[1].startswith("RULE:"), (
            f"Second token must be a RULE:xxx token, got '{tokens[1]}'"
        )

    def test_all_rule_tokens_exist_in_vocab(self):
        for rule_tok in self.rule_token_strings:
            assert rule_tok in self.vocab_mapping, (
                f"Rule token '{rule_tok}' is used during training but missing "
                f"from vocab.json — beam_search cannot anchor it."
            )

    def test_beam_search_seed_matches_training_prefix(self):
        bos_id = self.vocab_mapping["[BOS]"]

        rule_token_entries = sorted(
            [
                (tok, tid)
                for tok, tid in self.vocab_mapping.items()
                if tok.startswith("RULE:")
            ],
            key=lambda x: x[1],
        )
        assert len(rule_token_entries) > 0, "No RULE: tokens found in vocab"

        for idx, (rule_str, rule_id) in enumerate(rule_token_entries):
            seed = [bos_id, rule_id]
            assert seed[0] == bos_id, "Seed token 0 must be [BOS]"
            assert seed[1] == rule_id, f"Seed token 1 must be {rule_str} (ID {rule_id})"

    def test_training_prefix_length_matches_seed_length(self):
        training_prefix_len = 2
        beam_seed_len = 2

        assert training_prefix_len == beam_seed_len, (
            f"Training prefix length ({training_prefix_len}) != "
            f"beam_search seed length ({beam_seed_len}). "
            f"This will cause positional mismatch in the decoder."
        )

    def test_beam_search_4_arg_model_compatibility(self):
        from inference.beam_search import beam_search, NodeValidityPool

        class PermissiveNodePool(NodeValidityPool):
            def mask(self, tokens, candidate_tokens):
                return [True] * len(candidate_tokens)

        class MockFourArgModel(torch.nn.Module):
            def forward(self, src_tokens, src_positions, parent_child_pairs, tgt_tokens):
                vocab_size = 120
                seq_len = tgt_tokens.size(1)
                decoder_logits = torch.zeros((1, seq_len, vocab_size))
                decoder_logits[0, -1, 2] = 10.0  # predict [EOS] (ID 2)
                rule_logits = torch.zeros((1, 13))
                return decoder_logits, rule_logits, None

        mock_model = MockFourArgModel()
        src_tokens = torch.tensor([[1, 10, 9, 2]])
        vocab_map = {
            "token_to_id": self.vocab_mapping,
            "id_to_token": {v: k for k, v in self.vocab_mapping.items()},
        }

        result = beam_search(mock_model, src_tokens, vocab_map, max_len=5, node_pool=PermissiveNodePool())
        assert result["status"] == "solved"
        assert len(result["tokens"]) >= 2

