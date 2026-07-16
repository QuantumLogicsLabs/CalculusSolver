import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

def create_dummy_checkpoint(path):
    model = DummyModel()
    torch.save(model.state_dict(), path)

def validate_checkpoint(checkpoint_path, expected_shapes):
    state_dict = torch.load(checkpoint_path)
    for key, shape in expected_shapes.items():
        if key not in state_dict:
            raise KeyError(key)
        if state_dict[key].shape != shape:
            raise ValueError(shape)
    return True