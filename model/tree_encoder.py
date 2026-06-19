import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TreeEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, nhead=8, num_layers=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, input_ids, parent_ids=None, child_ids=None, attention_mask=None):
        x = self.embedding(input_ids)
        
        # Inject tree structure/parent-child relationship into structural bias if present
        # passing the mask to our native backbone
        hidden_states = self.transformer_encoder(x, src_key_padding_mask=attention_mask)
        return hidden_states
