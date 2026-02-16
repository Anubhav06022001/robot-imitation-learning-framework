import torch
import numpy as np

class AIRLTrainer:
    def __init__(self, env , policy, discriminator, replay_buffer, expert_dataset, cfg):
        self.env = env
        self.policy =policy
        self.discriminator = discriminator
        self.replay_buffer = replay_buffer
        self.expert_dataset = expert_dataset
        self.cfg = cfg


    def colllect_trjectories(self):
        obs_list = []
        act_list= []
        logp_list = []
        next_obs_list = []
        done_list = []

        n_rollouts = self.cfg.get("rollouts_per_epoch", 10)



        for _ in range(n_rollouts):
            o , _ = self.env.reset()
            done = False

            while not done:
                o_tensor = torch.tensor(o ,dtype = torch.float32).unsqueeze(0)
                dist = self.policy(o_tensor)
                a_tensor = dist.sample()
                logp = dist.log_prob(a_tensor).sum(-1)

                a = a_tensor.squeeze(0).detach().numpy()

                next_o, _, terminated, truncated, done = self.env.step(a)
                done = terminated or truncated

                obs_list.append(o)
                act_list.append(a)
                next_obs_list.append(next_o)
                logp_list.append(logp.item())
                done_list.append(done)

        obs_tensor = torch.from_numpy(np.array(obs_list , dtype = np.float32))
        next_obs_tensor = torch.from_numpy(np.array(next_obs_list, dtype= np.float32))
        act_tensor = torch.from_numpy(np.array(act_list, dtype = np.float32))
        logp_tensor = torch.tensor(logp_list, dtype = torch.float32)
        done_tensor = torch.tensor(done_list, dtype = torch.float32 )
        
        return {
            "obs_tensor": obs_tensor,
            "next_obs_tensor": next_obs_tensor , 
            "act_tensor" : act_tensor,
            "logp_tensor": logp_tensor,
            "done" : done_tensor
        }
    
    def update_discriminator(self, traj,expert_batch ):
        obs = traj["obs"] 
        acts = traj["act"]
        next_obs  = traj["next_obs"]
        done = traj["done"]
        logp = traj["logp"]

        logp = logp.detach()

        f_pi = self.compute_f( obs, acts, next_obs, done)
        logits_pi = f_pi - logp 

        f_exp = self.compute_f(obs_e ,acts_e, next_obs_e, done_e)
        logits_exp = f_exp 

        logits = torch.cat([logits_pi , logits_exp],dim = 0)

        labels = torch.cat([torch.ones_like(logits_exp), torch.zeros_like(logits_pi)], dim =0)

        bce = torch.nn.BCEWithLogitsLoss()
        
        self.disc_optim.zero_grad()
        loss_d = bce(logits , labels)
        loss_d.backward()
        self.disc_optim.step()
        return loss_d.item()


  
    def compute_f(self, obs, act,done, next_obs, gamma):
        r = self.reward_net(obs, act)
        v = self.value_net(obs)
        v_next = self.value_net(next_obs)
        f = r + gamma*(1-done)*v_next - v
        return f
    
    
    def update_policy(self,traj):
        obs = traj["obs"]
        acts = traj["acts"]
        next_obs = traj["next_obs"]
        done = traj["done"]

        # -------AIRL reward ----------------
        with torch.no_grad():
            rewards = self.compute_f(obs, acts, next_obs, done)

        # -------- compute returns --------
        gamma = self.cfg.get("gamma", 0.99)
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0,R)
        returns = torch.stack(returns)

        # normalize
        returns = (returns - returns.mean() )/ (returns.std() + 1e-8)

        #------ Policy gradient -----------

        dist = self.policy(obs)
        logp = dist.log_prob(acts).sum(-1)

        loss_pi = -(logp * returns).mean()

        self.policy_optim.zero_grad()
        loss_pi.backward()
        self.policy_optim.step()

        return loss_pi.item()

    def train(self):
        epochs = self.cfg.get("epochs", 200)
        batch_size = self.cfg.get("batch_size", 64)

        for epoch in range(epochs):

            # step 1) collect policy trajectories
            traj = self.collect_trajectories()

            # step 2) sample expert batch
            expert_batch = self.expert_dataset.sample(batch_size)

            # step 3) update discriminator (reward + value)
            loss_d = self.update_discriminator(traj, expert_batch)

            # step 4) update policy
            loss_pi = self.update_policy(traj)

            if (epoch + 1) % 10 == 0:
                print(
                    f"[AIRL] Epoch {epoch+1}/{epochs} | "
                    f"D loss: {loss_d:.4f} | Pi loss: {loss_pi:.4f}"
                )

