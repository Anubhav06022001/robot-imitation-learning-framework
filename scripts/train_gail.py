import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import torch
from il_lib.envs import BrachiationEnv
from il_lib.models import PolicyNetwork, Discriminator
from il_lib.expert_dataset import ExpertDataset
from il_lib.replay import ReplayBuffer
from il_lib.trainer.gail_trainer import GAILTrainer


def main():
    # --------- Paths ---------
    xml_path = "add path to you xml file here"
    expert_path = "add you expert dataset path here"

    # --------- Environment ---------
    env = BrachiationEnv(xml_path=xml_path, simend=20.0)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # --------- Expert dataset ---------
    obs_columns = ["time", "theta1", "theta2", "dtheta1", "dtheta2"]
    act_columns = ["control"]
    expert_dataset = ExpertDataset(expert_path, obs_columns, act_columns)

    # --------- Replay buffer ---------
    replay_buffer = ReplayBuffer()  

    # --------- Models ---------
    policy = PolicyNetwork(obs_dim=obs_dim, act_dim=act_dim)
    discriminator = Discriminator(obs_dim=obs_dim, act_dim=act_dim)

    # --------- Config ---------
    cfg = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "disc_lr": 1e-3,
        "policy_lr": 3e-4,
        "gamma": 0.99,
        "epochs": 100,           
        "rollouts_per_epoch": 3,
        "log_interval": 5,
    }

    # --------- Trainer ---------
    trainer = GAILTrainer(
        policy=policy,
        discriminator=discriminator,
        env=env,
        expert_dataset=expert_dataset,
        replay_buffer=replay_buffer,
        cfg=cfg,
    )

    # --------- Train ---------
    trainer.train()

    # --------- Save models ---------
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(policy.state_dict(), "checkpoints/policy_gail.pth")
    torch.save(discriminator.state_dict(), "checkpoints/discriminator_gail.pth")
    print("Saved models to checkpoints/.")


if __name__ == "__main__":
    main()
