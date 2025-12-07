import torch
import random
import torch.nn.functional as F
import numpy as np

class GAILTrainer:
    def __init__(self, policy, discriminator, env, expert_dataset, replay_buffer, cfg):
        self.policy = policy
        self.discriminator = discriminator
        self.env = env
        self.expert_dataset = expert_dataset
        self.replay_buffer = replay_buffer
        self.cfg = cfg

        self.device = cfg.get("device", "cpu")
        self.discriminator.to(self.device)
        self.policy.to(self.device)

        self.disc_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=cfg.get("disc_lr", 1e-3)
        )

        self.policy_optim = torch.optim.Adam(
            self.policy.parameters(),
            lr=cfg.get("policy_lr", 3e-4)
        )


    def collect_trajectories(self):
        obs_list = []
        act_list = []
        logp_list = []

        n_rollouts = self.cfg.get("rollouts_per_epoch", 10)

        for _ in range(n_rollouts):
            o,_ = self.env.reset()
            done = False

            while not done:
                o_tensor = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
                dist = self.policy(o_tensor)
                a_tensor = dist.sample()
                logp = dist.log_prob(a_tensor).sum(-1)

                a = a_tensor.squeeze(0).detach().numpy()

                next_o, _, terminated, truncated,_ = self.env.step(a)
                done = terminated or truncated

                obs_list.append(o)
                act_list.append(a)
                logp_list.append(logp.item())

                o = next_o

        obs_tensor = torch.from_numpy(np.array(obs_list, dtype=np.float32))
        act_tensor = torch.from_numpy(np.array(act_list, dtype=np.float32))
        logp_tensor = torch.tensor(logp_list, dtype=torch.float32)


        self.replay_buffer.add_batch(obs_list, act_list) 
        return {
            "obs": obs_tensor,
            "acts": act_tensor,
            "logp": logp_tensor,
        }

    def update_discriminator(self):
        # --- 1) sample expert batch ---
        batch_size = 64
        idxs_e = random.sample(range(len(self.expert_dataset)), batch_size)

        obs_e_list =[]
        act_e_list =[]
        for i in idxs_e:
            o_e,a_e = self.expert_dataset[i]
            obs_e_list.append(o_e)
            act_e_list.append(a_e)

        obs_e = torch.stack(obs_e_list).to(self.device)
        act_e = torch.stack(act_e_list).to(self.device)

        # --- 2) sample policy batch from replay buffer ---
        if len(self.replay_buffer) < batch_size:
            batch_size_pi = len(self.replay_buffer)
        else:
            batch_size_pi = batch_size

        obs_p_list, act_p_list = self.replay_buffer.sample(batch_size_pi)

        obs_p = torch.tensor(obs_p_list, dtype =torch.float32 , device = self.device)
        act_p = torch.tensor(act_p_list, dtype = torch.float32 , device = self.device)

        # --- 3) forward through discriminator  ---
        logits_e = self.discriminator(obs_e , act_e )
        logits_p = self.discriminator(obs_p , act_p)

        labels_e = torch.ones_like(logits_e)
        labels_p = torch.zeros_like(logits_p)

        logits = torch.cat([logits_e, logits_p], dim=0)    # [B+B',1]
        labels = torch.cat([labels_e, labels_p], dim=0)    # [B+B',1]

        # --- 5) binary cross-entropy with logits ---
        bce = torch.nn.BCEWithLogitsLoss()

        self.disc_optim.zero_grad()
        loss_d = bce(logits, labels)
        loss_d.backward()
        self.disc_optim.step()

        return loss_d.item()


    def update_policy(self, traj):
        obs  = traj["obs"].to(self.device)   # [N, obs_dim]
        acts = traj["acts"].to(self.device)  # [N, act_dim]
        # old_logp = traj["logp"].to(self.device)  # unused for now

        # 1) compute reward from discriminator 
        with torch.no_grad():
            logits = self.discriminator(obs, acts).squeeze(-1)  # [N]
            rewards = F.softplus(logits)                        # [N]

        # 2) compute discounted returns (Monte Carlo)
        gamma = self.cfg.get("gamma", 0.99)
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.stack(returns)  # [N]

        # 3) normalize returns to reduce variance
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 4) compute log π(a|s) under current policy
        dist = self.policy(obs)
        logp = dist.log_prob(acts).sum(-1)    
        """
        Discounted return is:

        Gt​=k≥t∑ ​γ k−tr(s k​,a k​)

        Policy gradient:

        ∇θ​J≈E[∇θ​logπθ​(a∣s)⋅Gt​]
        """

        # 5) policy loss: REINFORCE
        loss_pi = -(logp * returns).mean()

        # 6) optimize
        self.policy_optim.zero_grad()
        loss_pi.backward()
        self.policy_optim.step()

        return loss_pi.item()

        

    def train(self):
        epochs = self.cfg.get("epochs", 100)
        log_interval = self.cfg.get("log_interval", 10)

        for epoch in range(epochs):
            traj = self.collect_trajectories()
            loss_d = self.update_discriminator()
            loss_p = self.update_policy(traj)

            if (epoch+1)%log_interval == 0:
                print(f"Epoch {epoch+1}/{epochs} | D:{loss_d:.4f} | P: {loss_p:.4f}")
