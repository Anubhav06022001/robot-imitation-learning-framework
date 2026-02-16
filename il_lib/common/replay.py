import random

class ReplayBuffer():

    def __init__(self):
        self.obs_list = []
        self.act_list = []

    def add_batch(self, obs_batch , act_batch):
        for o,a in zip(obs_batch, act_batch):
            self.obs_list.append(o)           # obs_batch: list or array of shape [B, obs_dim]
            self.act_list.append(a)           # act_batch: list or array of shape [B, act_dim]

    def sample(self, batch_size):
        idxs = random.sample(range(len(self.obs_list)),batch_size)
        obs_batch  = [self.obs_list[i] for i in idxs]
        act_batch = [self.act_list[i] for i in idxs]
        return obs_batch, act_batch

    def __len__(self):
        return len(self.obs_list)