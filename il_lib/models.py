import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128 ), nn.ReLU()
        )
        self.mu = nn.Linear(128, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))    
        
    def forward(self, x):
        x= self.fc(x)
        mu = self.mu(x)
        std = torch.exp(self.log_std)
        return Normal(mu,std)  

class Discriminator(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()      
        self.net = nn.Sequential(
            nn.Linear(obs_dim+act_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)                       # No sigmoid here; we return logits and apply sigmoid in loss/reward
            # nn.Linear(128, 1) , nn.Sigmoid()
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)
    
    def prob(self, obs, act):
        logits = self.forward(obs, act)
        return torch.sigmoid(logits)

    





