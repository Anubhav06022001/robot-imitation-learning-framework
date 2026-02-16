import torch

class DiffusionPolicy:
    def __init__(self, model, act_dim, n_steps=100, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.n_steps = n_steps
        self.betas = torch.linspace(1e-4, 0.02, n_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    def q_sample(self, x0, t, noise):
        # x0: (B, act_dim)
        alpha_bar_t = self.alpha_bars[t].unsqueeze(-1)
        return alpha_bar_t.sqrt() * x0 + (1.0 - alpha_bar_t).sqrt() * noise