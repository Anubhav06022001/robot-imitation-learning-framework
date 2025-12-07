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
---

## 📌 Expert Demonstration Format

> Dataset is **not included**.  
> Add your expert demonstrations at:  
> `data/merged_imitation_data.csv`

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
Run training:

python scripts/train_gail.py


Checkpoints will be saved to:

checkpoints/policy_gail.pth
checkpoints/discriminator_gail.pth

🧠 Why Brachiation?

Brachiation is underactuated and contact-rich, requiring precise swing dynamics and timing.
This makes it an excellent benchmark for studying robotic locomotion control through imitation learning.

Our goal is to provide a reproducible and extendable platform for IL research on such complex motion tasks.

👤 Author

Anubhav Tripathi

Feel free to contribute, open issues, or share results! 🚀


---

After replacing and saving:

```bash
git add README.md
git commit -m "Fix table and code blocks in README"
git push -u origin main --force