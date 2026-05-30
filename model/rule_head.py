import torch
import torch.nn as nn
from typing import List, Optional


class RuleHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        rule_labels: Optional[List[str]] = None,
    ):
        super().__init__()
        self.rule_labels = rule_labels or []
        self.num_rules = len(self.rule_labels)
        self.classifier = nn.Linear(hidden_dim, self.num_rules)
        self.rule_embeddings = nn.Embedding(self.num_rules, hidden_dim)

    def forward(self, encoder_out, root_mask=None):
        if root_mask is not None:
            root_mask = root_mask.float().unsqueeze(-1)
            pooled = (encoder_out * root_mask).sum(dim=1) / (
                root_mask.sum(dim=1) + 1e-6
            )
        else:
            pooled = encoder_out[:, 0, :]

        logits = self.classifier(pooled)
        return logits

    def embed_rules(self, rule_ids):
        return self.rule_embeddings(rule_ids)

    def labels(self) -> List[str]:
        return list(self.rule_labels)
