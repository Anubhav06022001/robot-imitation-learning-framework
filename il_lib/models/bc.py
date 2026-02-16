# Simple BC MLP: state -> torque


import torch
import torch.nn as nn

class BC_MLP(nn.Module):
    def __init__(self, state_dim=4, out_dim=1, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)