import torch
import torch.nn as nn


class TreeDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim=512,
        num_heads=8,
        num_layers=8,
        ffn_dim=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    @staticmethod
    def _causal_mask(seq_len, device):
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        )

    def forward(
        self,
        target_tokens,
        encoder_output,
        rule_embeddings=None,
        validity_mask=None,
        tgt_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        target_emb = self.embedding(target_tokens)

        if rule_embeddings is not None:
            rule_prefix = rule_embeddings.unsqueeze(1)
            target_emb = torch.cat([rule_prefix, target_emb], dim=1)

        seq_len = target_emb.size(1)
        tgt_mask = self._causal_mask(seq_len, target_emb.device)

        decoder_output = self.decoder(
            target_emb,
            encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        if rule_embeddings is not None:
            decoder_output = decoder_output[:, 1:, :]

        logits = self.lm_head(decoder_output)
        if validity_mask is not None:
            logits = logits.masked_fill(~validity_mask, float("-inf"))

        return logits, decoder_output
