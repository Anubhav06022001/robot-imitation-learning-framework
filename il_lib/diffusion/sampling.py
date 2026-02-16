import torch
from utils.embeddings import timestep_embedding

@torch.no_grad()
def ddim_sample(diff, model, state_t, bc_pred_t, H=1, t_dim=64, eta=0.0):
    # state_t: (B, state_dim) ; bc_pred_t: (B, act_dim)
    device = diff.device
    act_dim = bc_pred_t.shape[-1]
    # start noise in residual space
    x = torch.randn((state_t.size(0), act_dim), device=device)  # (B, act_dim)
    for i in reversed(range(diff.n_steps)):
        t_val = torch.full((state_t.size(0),), i, dtype=torch.long, device=device)
        t_emb = timestep_embedding(t_val, dim=t_dim, device=device)  # (B, t_dim)
        eps = model(x, state_t, bc_pred_t, t_emb)  # predict noise (residual noise)
        alpha_t = diff.alphas[i]
        alpha_bar_t = diff.alpha_bars[i]
        if i == 0:
            x = (1.0 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_bar_t).sqrt() * eps)
            break
        alpha_bar_prev = diff.alpha_bars[i-1]
        pred_x0 = (x - (1 - alpha_bar_t).sqrt() * eps) / alpha_bar_t.sqrt()
        dir_part = (1 - alpha_bar_prev).sqrt() * eps
        x = alpha_bar_prev.sqrt() * pred_x0 + dir_part
    return x  # normalized residual (B, act_dim)