from il_lib.envs import BrachiationEnv
from il_lib.models import PolicyNetwork, RewardNet, ValueNet
from il_lib.trainer.airl_trainer import AIRLTrainer
from il_lib.datasets.expert_dataset import ExpertDataset

xml_path = "/home/anubhav/Documents/dm_control/dm_control/suite/acrobot_multiple_allegro.xml"
expert_csv = "/home/anubhav/Documents/dm_control/dm_control/suite/acrobot_Imitation_learning/robot-imitation-learning-framework/data/merged_imitation_data.csv"   
obs_cols = ["time", "theta1", "theta2", "dtheta1", "dtheta2"]
act_col = "control"

env = BrachiationEnv(xml_path)

policy = PolicyNetwork(obs_dim=5, act_dim=1)
reward_net = RewardNet(obs_dim=5, act_dim=1)
value_net = ValueNet(obs_dim=5)

expert_dataset = ExpertDataset(
    expert_csv, obs_cols, act_col
)

cfg = {
    "gamma": 0.99,
    "epochs": 200,
    "rollouts_per_epoch": 10,
    "batch_size": 64,
    "policy_lr": 3e-4,
    "disc_lr": 1e-3,
}

trainer = AIRLTrainer(
    env=env,
    policy=policy,
    reward_net=reward_net,
    value_net=value_net,
    expert_dataset=expert_dataset,
    cfg=cfg,
)

trainer.train()
