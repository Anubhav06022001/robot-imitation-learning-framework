'''
# noisy_residual (act_dim)

-->The current “corrupted” sample that diffusion wants to denoise.

# bc_pred (act_dim)

-->The baseline torque predicted by BC_MLP.

This is crucial because diffusion needs to know where the residual is referenced from.
'''

# models/denoiser.py
# Denoiser input: noisy_residual (act_dim), state (state_dim), bc_pred (act_dim), t_emb (t_dim)
import torch
import torch.nn as nn

class DiffusionDenoiser(nn.Module):
    def __init__(self, state_dim, act_dim, t_dim=64, hidden=256):
        super().__init__()
        # input: noisy_res + state + bc_pred + t_emb
        self.in_dim = act_dim + state_dim + act_dim + t_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )
    def forward(self, noisy_res, state, bc_pred, t_emb):
        # shapes: noisy_res (B, act_dim) ; state (B, state_dim) ; bc_pred (B, act_dim) ; t_emb (B, t_dim)
        x = torch.cat([noisy_res, state, bc_pred, t_emb], dim=-1)
        return self.net(x)