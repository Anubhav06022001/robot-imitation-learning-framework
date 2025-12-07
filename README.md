# Imitation Learning for Brachiation Robot (MuJoCo + GAIL)

This repository implements **Generative Adversarial Imitation Learning (GAIL)** for a MuJoCo-based brachiation (swinging) robot.  
The code is organized to be extended with more imitation learning algorithms like **AIRL, DAgger, Diffusion Policy, and GAIL-ES** in the future.

---

## Project Structure

```text
Brachiation_project/
├─ il_lib/
│  ├─ envs.py           # BrachiationEnv (MuJoCo -> Gym-style)
│  ├─ models.py         # PolicyNetwork (Gaussian), Discriminator
│  ├─ expert_dataset.py # Expert demo loader (CSV/Excel -> tensors)
│  ├─ replay.py         # Replay buffer for policy (s,a) pairs
│  ├─ utils.py          # Misc utilities (TODO: extend)
│  ├─ losses.py         # Loss helpers (TODO: extend)
│  └─ trainer/
│     ├─ gail_trainer.py      # GAILTrainer (core training loop)
│     ├─ policy_updater.py    # (TODO) PPO/actor-critic helpers
│     └─ value_updater.py     # (TODO) Value function training
├─ scripts/
│  ├─ train_gail.py     # Entry point to train GAIL
│  └─ evaluate_policy.py# (TODO) Policy evaluation / visualization
├─ configs/
│  └─ default.yaml      # (TODO) Hyperparameter configs
├─ tests/
│  ├─ test_env.py       # Simple environment sanity checks
│  ├─ test_models.py    # Policy + Discriminator tests
│  └─ test_trainer_sanity.py  # (TODO) Trainer sanity tests
├─ data/
│  └─ (your expert data goes here)
└─ README.md
