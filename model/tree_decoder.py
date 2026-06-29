import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class TreeDecoder(nn.Module):
    def __init__(self, vocab_size, num_rules=15, hidden_dim=512, nhead=8, num_layers=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rule_embeddings = nn.Embedding(num_rules, hidden_dim)

        decoder_layer = TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_ids, memory, rule_id, tgt_mask=None, validity_mask=None, memory_key_padding_mask=None):
        tgt = self.embedding(tgt_ids)
        r_emb = self.rule_embeddings(rule_id).unsqueeze(1)
        tgt = tgt + r_emb

        output = self.transformer_decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        logits = self.fc_out(output)
        if validity_mask is not None:
            logits = logits + validity_mask

        return logits
