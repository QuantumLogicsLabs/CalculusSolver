import torch
import torch.nn as nn

class StepTracer(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.templates = [
            "Identify the outermost derivative operation layer.",
            "Apply the Power Rule to bring down power and decrement.",
            "Apply the Chain Rule to differentiate internal composite functions.",
            "Distribute integrals over individual sum additions.",
            "Perform variable integration substitution matching substitution paths.",
            "Evaluate definite integral boundaries sequentially.",
            "Simplify algebraic expressions matching target reduction forms."
        ]
        self.classifier = nn.Linear(hidden_dim, len(self.templates))

    def template_for(self, rule_id):
        idx = rule_id if rule_id < len(self.templates) else 0
        return self.templates[idx]

    def forward(self, hidden_states):
        return self.classifier(hidden_states.mean(dim=1))
