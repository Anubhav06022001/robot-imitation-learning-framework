import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from il_lib.models import PolicyNetwork, Discriminator
import torch

policy = PolicyNetwork(obs_dim=5, act_dim=1)
disc   = Discriminator(obs_dim=5, act_dim=1)


obs = torch.randn(10, 5)       # 10 samples
act = torch.randn(10, 1)

# Policy tests
dist = policy(obs)
sample_actions = dist.sample()
logp = dist.log_prob(sample_actions).sum(-1)

# Discriminator tests
logits = disc(obs, act)
probs  = torch.sigmoid(logits)

print("actions shape:", sample_actions.shape)   # expect (10,1)
print("logp shape:", logp.shape)               # expect (10,)
print("logits shape:", logits.shape)           # expect (10,1)
print("probs in [0,1]:", probs.min().item(), probs.max().item())
