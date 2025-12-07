 # 🤖 Robot Imitation Learning Framework

A modular framework for **imitation learning in robotics**, currently focused on **dynamic brachiation control** using MuJoCo.  
The system is task-agnostic — provide expert demonstrations, and it learns the behavior.

---

## 🔥 Implemented & Upcoming Methods

Currently supported:
- GAIL – Generative Adversarial Imitation Learning ✔

Upcoming algorithms (will be uploaded soon):
- AIRL – Adversarial IRL  
- DAgger – Dataset Aggregation  
- Residual Diffusion Imitation Policies  
- GAIL-ES – Evolution Strategies + GAIL  

This repository is actively evolving to benchmark **multiple IL algorithms** under the same environment and training structure.

---

## 📁 Project Structure

```text
robot-imitation-learning-framework/
├─ il_lib/
│  ├─ envs.py                 # MuJoCo-based environments (BrachiationEnv)
│  ├─ models.py               # PolicyNetwork (Gaussian), Discriminator
│  ├─ expert_dataset.py       # Expert dataset loader (CSV -> tensors)
│  ├─ replay.py               # Replay buffer for policy (s,a) samples
│  ├─ utils.py                # Common utilities (logging, etc.)
│  ├─ losses.py               # IL-specific loss helpers (TODO)
│  └─ trainer/
│     ├─ gail_trainer.py      # GAIL training loop (current core)
│     ├─ policy_updater.py    # PPO/actor-critic helpers (TODO)
│     └─ value_updater.py     # Value learning (TODO)
│
├─ scripts/
│  ├─ train_gail.py           # Main training entrypoint
│  └─ evaluate_policy.py      # Policy rollout & diagnostics (TODO)
│
├─ configs/
│  └─ default.yaml            # Hyperparameter configs (TODO)
│
├─ tests/
│  ├─ test_env.py             # Env sanity tests
│  ├─ test_models.py          # Forward pass tests
│  └─ test_trainer_sanity.py  # Sanity check (TODO)
│
├─ data/
│  └─ .gitkeep                # Place expert data here
│
└─ README.md
```

## 📌 Expert Demonstration Format

Add your expert demonstrations to:  
```bash
`data/expert_data.csv`
```

| Column  | Description                         |
|---------|-------------------------------------|
| time    | simulation timestep                 |
| theta1  | shoulder joint angle                |
| theta2  | elbow joint angle                   |
| dtheta1 | shoulder joint angular velocity     |
| dtheta2 | elbow joint angular velocity        |
| control | expert torque                       |

The policy aims to match the expert **occupancy distribution** in state–action space.

---

## 🚀 Run Training (GAIL)

Install dependencies:

```bash
pip install torch mujoco gymnasium pandas numpy
```

Run training:

```bash
python scripts/train_gail.py
```

Checkpoints will be saved to:

```bash
checkpoints/policy_gail.pth
checkpoints/discriminator_gail.pth
```

🧠 Why Brachiation?

Brachiation is:

underactuated

highly dynamic

contact-rich

requires precise swing timing

These properties make it an ideal benchmark for adversarial imitation learning and learned control in challenging robotic locomotion tasks.

Our goal is to build a reproducible and extendable research platform for state-of-the-art IL in robotics.

👤 Author

Anubhav Tripathi

🙌 Contributions

Open for issues, pull requests, and extensions!
If you build an IL method on this framework, please share! 🚀





---


