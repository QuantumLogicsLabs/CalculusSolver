import torch
import torch.nn as nn

class StepTracer(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.trace_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states):
        return self.trace_layer(hidden_states)