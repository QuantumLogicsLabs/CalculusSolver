import torch
import torch.nn as nn
from typing import List, Optional


class StepTracer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 512,
        templates: Optional[List[str]] = None,
    ):
        super().__init__()
        self.templates = templates or []
        self.num_templates = len(self.templates)
        self.classifier = nn.Linear(hidden_dim, self.num_templates)

    def forward(self, rule_ids, decoder_hidden_states):
        if decoder_hidden_states is not None:
            hidden = decoder_hidden_states.mean(dim=1)
        else:
            hidden = torch.nn.functional.one_hot(
                rule_ids, num_classes=self.num_templates
            ).float()

        logits = self.classifier(hidden)
        return logits

    def template_for(self, rule_id):
        index = rule_id.item() if isinstance(rule_id, torch.Tensor) else rule_id
        if index < 0 or index >= len(self.templates):
            return ""
        return self.templates[index]
