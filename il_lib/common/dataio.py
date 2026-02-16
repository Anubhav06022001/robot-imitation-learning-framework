# utils/dataio.py
import torch
import numpy as np
import pandas as pd


def compute_and_save_normalizers(csv_path, out_path):          # creates the stats
    df = pd.read_csv(csv_path)
    s_cols = ['theta1','theta2','dtheta1','dtheta2']
    a_col = ['control']
    states = df[s_cols].values.astype(np.float32)
    actions = df[a_col].values.astype(np.float32)
    s_mean = states.mean(axis=0); s_std = states.std(axis=0) + 1e-8
    a_mean = actions.mean(axis=0); a_std = actions.std(axis=0) + 1e-8
    torch.save({'s_mean': s_mean, 's_std': s_std, 'a_mean': a_mean, 'a_std': a_std}, out_path)
    return s_mean, s_std, a_mean, a_std

def load_normalizers(path, device='cpu'):                  # loads the already computed stats and converts to Torch format.
    d = torch.load(path, map_location=device, weights_only=False)
    s_mean = torch.tensor(d['s_mean'], dtype=torch.float32, device=device)
    s_std  = torch.tensor(d['s_std'], dtype=torch.float32, device=device)
    a_mean = torch.tensor(d['a_mean'], dtype=torch.float32, device=device)
    a_std  = torch.tensor(d['a_std'], dtype=torch.float32, device=device)
    return s_mean, s_std, a_mean, a_std

def normalize_states(states_np, s_mean, s_std):
    return (states_np - s_mean) / s_std

def normalize_actions(actions_np, a_mean, a_std):
    return (actions_np - a_mean) / a_std

def load_residual_normalizers(path, device='cpu'):
    d = torch.load(path, map_location=device)
    r_mean = torch.tensor(d['r_mean'], dtype=torch.float32, device=device)
    r_std  = torch.tensor(d['r_std'],  dtype=torch.float32, device=device)
    return r_mean, r_std


def load_dataset(csv_path, state_cols=['theta1','theta2','dtheta1','dtheta2'], action_col='control'):
    """Return states_np, actions_np (numpy arrays)."""
    df = pd.read_csv(csv_path)
    states = df[state_cols].values.astype(np.float32)
    actions = df[[action_col]].values.astype(np.float32).reshape(-1)
    return states, actions

def normalize_arrays(states_np, actions_np, s_mean, s_std, a_mean, a_std):
    """Return normalized states_np, normalized actions_np (numpy arrays)."""
    s_norm = (states_np - s_mean.cpu().numpy()) / (s_std.cpu().numpy() + 1e-8)
    a_norm = (actions_np - a_mean.cpu().numpy()) / (a_std.cpu().numpy() + 1e-8)
    return s_norm, a_norm