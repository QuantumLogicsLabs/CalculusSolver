import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Takes a single tensor 'x' of shape (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe.narrow(1, 0, seq_len)


class SimpleCalculusModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
        max_len: int = 128,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.hidden_dim = hidden_dim

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=max_len)

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, src_seq: torch.Tensor, tgt_in_seq: torch.Tensor) -> torch.Tensor:
        device = src_seq.device

        src_emb = self.pos_encoding(self.token_embedding(src_seq) * math.sqrt(self.hidden_dim))
        tgt_emb = self.pos_encoding(self.token_embedding(tgt_in_seq) * math.sqrt(self.hidden_dim))

        src_key_padding_mask = src_seq == self.pad_id
        tgt_key_padding_mask = tgt_in_seq == self.pad_id

        tgt_seq_len = tgt_in_seq.size(1)
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len, device)

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        return self.lm_head(out)