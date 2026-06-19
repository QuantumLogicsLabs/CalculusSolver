import torch
import torch.nn as nn

class RuleHead(nn.Module):
    def __init__(self, hidden_dim=512, num_rules=15):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_rules)
        self.rule_labels = [
            "power_rule", "chain_rule", "product_rule", "quotient_rule", 
            "integrate_power", "substitution", "by_parts", "constant_rule",
            "sum_rule", "trig_diff", "trig_int", "exponential", "logarithmic",
            "simplify", "done"
        ]
        
    def labels(self):
        return self.rule_labels

    def forward(self, encoder_hidden_states):
        root_token_state = encoder_hidden_states[:, 0, :]
        rule_logits = self.classifier(root_token_state)
        return rule_logits
