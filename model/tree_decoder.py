import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class TreeDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, nhead=8, num_layers=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Native PyTorch Transformer Decoder backbone
        decoder_layer = TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, tgt_ids, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Autoregressively generates output SLaNg tokens using encoder hidden states (memory)
        tgt = self.embedding(tgt_ids)
        output = self.transformer_decoder(
            tgt, 
            memory, 
            tgt_mask=tgt_mask, 
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        logits = self.fc_out(output)
        return logits
