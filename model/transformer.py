import torch
import torch.nn as nn
from .tree_encoder import TreeEncoder
from .tree_decoder import TreeDecoder
from .rule_head import RuleHead

class CalculusSolverModel(nn.Module):
    def __init__(self, vocab_size, num_rules=15, hidden_dim=512, heads=8, encoder_layers=8, decoder_layers=8):
        super().__init__()
        # Initializing 3 components with their configurations 
        self.encoder = TreeEncoder(vocab_size, hidden_dim, heads, encoder_layers)
        self.decoder = TreeDecoder(vocab_size, hidden_dim, heads, decoder_layers)
        self.rule_head = RuleHead(hidden_dim, num_rules)
        
    def forward(self, input_ids, tgt_ids=None, attention_mask=None):
        # 1. Run Tree Encoder to get per-node hidden states
        memory = self.encoder(input_ids, attention_mask=attention_mask)
        
        # 2. Predict Calculus Rule from the root token state
        rule_logits = self.rule_head(memory)
        
        # 3. Autoregressive prediction using Decoder (if targets are provided during training)
        logits = None
        if tgt_ids is not None:
            # Generate a standard causal mask for subsequent tokens
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_ids.size(1), device=tgt_ids.device)
            logits = self.decoder(
                tgt_ids, 
                memory, 
                tgt_mask=tgt_mask, 
                memory_key_padding_mask=attention_mask
            )
            
        return logits, rule_logits