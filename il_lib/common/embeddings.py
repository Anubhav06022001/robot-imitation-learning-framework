import torch
import numpy as np

def timestep_embedding(t, dim=64, device='cpu'):
    if t.ndim == 2:
        t = t.squeeze(-1)
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(0, half, dtype=torch.float32, device=device) / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(emb.size(0), 1, device=device)], dim=-1)
    return emb