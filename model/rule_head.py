import torch
import torch.nn as nn

class RuleHead(nn.Module):
    def __init__(self, hidden_dim=512, num_rules=15):
        super().__init__()
        # linear classification at Encoder root token(CLS or index 0) at hidden state 
        self.classifier = nn.Linear(hidden_dim, num_rules)
        
    def forward(self, encoder_hidden_states):
        # To find the first (root) token state of every sequence of Batch 
        root_token_state = encoder_hidden_states[:, 0, :]
        # To predict logistics of Calculus rules (e.g., power_rule, chain_rule)
        rule_logits = self.classifier(root_token_state)
        return rule_logits
