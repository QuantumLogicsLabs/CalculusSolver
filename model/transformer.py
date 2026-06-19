import torch
import torch.nn as nn
from .tree_encoder import TreeEncoder
from .tree_decoder import TreeDecoder
from .rule_head import RuleHead

class CalculusSolverModel(nn.Module):
    def __init__(self, vocab_size, num_rules=15, hidden_dim=512, heads=8, encoder_layers=8, decoder_layers=8):
        super().__init__()
        self.encoder = TreeEncoder(vocab_size, hidden_dim, heads, encoder_layers)
        self.decoder = TreeDecoder(vocab_size, num_rules, hidden_dim, heads, decoder_layers)
        self.rule_head = RuleHead(hidden_dim, num_rules)
        
    def forward(self, input_ids, tgt_ids=None, parent_ids=None, child_ids=None, attention_mask=None, validity_mask=None):
        memory = self.encoder(input_ids, parent_ids=parent_ids, child_ids=child_ids, attention_mask=attention_mask)
        rule_logits = self.rule_head(memory)
        
        # Get predicted rule ID for decoder conditioning
        pred_rule_id = torch.argmax(rule_logits, dim=-1)
        
        logits = None
        if tgt_ids is not None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_ids.size(1), device=tgt_ids.device)
            logits = self.decoder(
                tgt_ids, 
                memory, 
                pred_rule_id,
                tgt_mask=tgt_mask, 
                validity_mask=validity_mask,
                memory_key_padding_mask=attention_mask
            )
            
        return logits, rule_logits