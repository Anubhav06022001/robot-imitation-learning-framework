import os
import sys
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from il_lib.envs import BrachiationEnv
from il_lib.models import PolicyNetwork, Discriminator
from il_lib.expert_dataset import ExpertDataset
from il_lib.replay import ReplayBuffer
from il_lib.trainer.gail_trainer import GAILTrainer


# ---------------------------------------------------------------------
# 1) Build env and sanity-check stepping
# ---------------------------------------------------------------------
xml_path = "/home/anubhav/Documents/dm_control/dm_control/suite/acrobot_multiple_allegro.xml"
env = BrachiationEnv(xml_path)

obs, info = env.reset()
print("obs shape:", obs.shape)

for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(i, "reward:", reward, "terminated:", terminated, "truncated:", truncated)

# Infer dimensions
obs_dim = obs.shape[-1]
sample_action = env.action_space.sample()
act_dim = sample_action.shape[-1] if hasattr(sample_action, "shape") else 1
print("obs_dim:", obs_dim, "act_dim:", act_dim)


# ---------------------------------------------------------------------
# 2) Build expert dataset
# ---------------------------------------------------------------------
csv_path = "/home/anubhav/Documents/dm_control/dm_control/suite/Data/merged_imitation_data.csv"
obs_cols = ["time", "theta1", "theta2", "dtheta1", "dtheta2"]
act_cols = ["control"]

ds = ExpertDataset(csv_path, obs_cols, act_cols)
print("expert dataset size:", len(ds))
print("first sample:", ds[0])


# ---------------------------------------------------------------------
# 3) Build replay buffer and sanity-check sampling
# ---------------------------------------------------------------------
buf = ReplayBuffer()

obs_batch = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
act_batch = [[0.1], [0.2]]

buf.add_batch(obs_batch, act_batch)
print("replay buffer size:", len(buf))

obs_samp, act_samp = buf.sample(1)
print("sample from replay:", obs_samp, act_samp)


# ---------------------------------------------------------------------
# 4) Load config (or make a small dict)
# ---------------------------------------------------------------------
cfg_path = os.path.join(ROOT, "configs", "default.yaml")
if os.path.exists(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
else:
    cfg = {
        "device": "cpu",
        "disc_lr": 1e-3,
        "policy_lr": 3e-4,
        "gamma": 0.99,
        "rollouts_per_epoch": 2,
    }


# ---------------------------------------------------------------------
# 5) Build policy & discriminator from your models.py
# ---------------------------------------------------------------------
policy = PolicyNetwork(
    obs_dim=obs_dim,
    act_dim=act_dim,
   
)

discriminator = Discriminator(
    obs_dim=obs_dim,
    act_dim=act_dim,

)

replay_buffer = buf     
expert_dataset = ds


# ---------------------------------------------------------------------
# 6) Build trainer and run a tiny step
# ---------------------------------------------------------------------
trainer = GAILTrainer(
    policy=policy,
    discriminator=discriminator,
    env=env,
    expert_dataset=expert_dataset,
    replay_buffer=replay_buffer,
    cfg=cfg,
)

traj = trainer.collect_trajectories()
print("traj shapes:", traj["obs"].shape, traj["acts"].shape, traj["logp"].shape)
print("replay size after collect:", len(replay_buffer))

loss_d = trainer.update_discriminator()
loss_p = trainer.update_policy(traj)
print("loss_d:", loss_d, "loss_p:", loss_p)
