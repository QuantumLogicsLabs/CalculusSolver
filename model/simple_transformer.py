"""
Simplified calculus solver model — a single standard encoder-decoder
Transformer, replacing the tree-encoder + separate RuleHead + separate
StepTracer design in model/transformer.py.

Rationale (see docs/KNOWN_ISSUES.md): the tree-structured model's
src_positions/parent_child_pairs were always passed as zero tensors by
every caller in this codebase, so the "tree" encoder was functioning as a
plain token encoder anyway. The separate RuleHead caused a real bug
(pooling only from token 0) and a circular training dependency with the
decoder. Rule prediction is now folded into the output sequence itself:
the decoder's first generated token (after [BOS]) is a RULE:xxx token,
and the rest of the sequence is the answer -- one clean sequence-generation
loss, no separate classifier, no coupling bug possible.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class SimpleCalculusModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
        max_len: int = 32,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.hidden_dim = hidden_dim

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=max_len * 2)

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    @staticmethod
    def _causal_mask(seq_len, device):
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        )

    def forward(self, src_seq, tgt_in_seq):
        device = src_seq.device

        src_padding_mask = src_seq == self.pad_id
        tgt_padding_mask = tgt_in_seq == self.pad_id

        src_emb = self.pos_encoding(self.token_embedding(src_seq))
        tgt_emb = self.pos_encoding(self.token_embedding(tgt_in_seq))

        tgt_mask = self._causal_mask(tgt_in_seq.size(1), device)

        decoder_output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

        logits = self.lm_head(decoder_output)
        return logits