"""
config.py — MAPPOConfig dataclass.

All hyperparameters in one place. Can be saved/loaded as JSON.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class MAPPOConfig:
    # ── environment ──────────────────────────────────────────────────────────
    max_steps:       int   = 500
    num_envs:        int   = 8          # parallel environments (sequential sim)
    rollout_length:  int   = 256        # steps per rollout per env

    # ── PPO ──────────────────────────────────────────────────────────────────
    gamma:           float = 0.995
    gae_lambda:      float = 0.95
    clip_eps:        float = 0.15
    ppo_epochs:      int   = 4
    mini_batch_size: int   = 512        # samples per mini-batch during update

    # ── optimiser ────────────────────────────────────────────────────────────
    actor_lr:        float = 1e-4
    critic_lr:       float = 3e-4
    max_grad_norm:   float = 0.5

    # ── loss coefficients ────────────────────────────────────────────────────
    entropy_coef:    float = 0.02
    value_coef:      float = 1.0

    # ── reward ───────────────────────────────────────────────────────────────
    dense_reward_coef: float = 1.0     # used when dense_reward_anneal=False

    # Piecewise dense-reward annealing (sparse terminal rewards unchanged):
    #   first 40% → coef_early, next 40% → coef_mid, final 20% → coef_late
    dense_reward_anneal:            bool  = True
    dense_reward_coef_early:        float = 1.0
    dense_reward_coef_mid:          float = 0.5
    dense_reward_coef_late:         float = 0.2
    dense_reward_anneal_mid_start:  float = 0.4   # fraction of total_env_steps
    dense_reward_anneal_late_start: float = 0.8

    # ── model ────────────────────────────────────────────────────────────────
    n_spatial:         int   = 18
    n_scalar:          int   = 28
    n_actions:         int   = 6
    global_scalar_dim: int   = 64   # privileged critic vector (see critic_features.py)

    # ── training loop ────────────────────────────────────────────────────────
    total_env_steps:   int   = 4_000_000
    checkpoint_every:  int   = 100       # updates between checkpoints
    eval_every:        int   = 200       # updates between evaluations
    eval_matches:      int   = 50
    
    # ── checkpoint retention ─────────────────────────────────────────────────
    keep_last_checkpoints: int  = 3
    save_best_only:        bool = False
    best_metric:           str  = "estimated_score"
    best_metric_mode:      str  = "max"

    # ── league ───────────────────────────────────────────────────────────────
    league_baseline_frac:  float = 0.30
    league_ckpt_frac:      float = 0.50
    # remaining 20% = random historical

    # ── paths ────────────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints/mappo"
    log_dir:        str = "logs/mappo"

    # ── misc ─────────────────────────────────────────────────────────────────
    seed:           int   = 42
    device:         str   = "auto"       # "auto" → cuda if available else cpu

    # ── phase label (for logging) ────────────────────────────────────────────
    phase:          str   = "baseline"   # baseline | selfplay | finetune

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "MAPPOConfig":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
