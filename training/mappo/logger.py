"""
logger.py — Training and evaluation logger.

Writes CSV + JSONL.  TensorBoard is optional (silently skipped if unavailable).
Never used inside Agent.act() or Agent.__init__().
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
    _TB_AVAILABLE = True
except Exception:
    _TB_AVAILABLE = False


class TrainingLogger:
    """
    Logs one row per PPO update to:
      logs/mappo/train.csv
      logs/mappo/train.jsonl
    and optionally to TensorBoard.
    """

    TRAIN_FIELDS = [
        "update", "env_steps", "episodes_completed", "fps",
        "actor_loss", "critic_loss", "entropy", "approx_kl",
        "clip_fraction", "explained_variance",
        "mean_return", "mean_episode_length",
        "dense_reward_coef", "dense_reward_anneal_phase",
        "actor_lr", "critic_lr",
        "tracker_mean_step", "tracker_mean_boxes", "tracker_mean_items",
        "tracker_mean_bombs", "tracker_mean_kills",
        "checkpoint_path",
    ]

    EVAL_FIELDS = [
        "checkpoint_path", "num_matches", "fixed_seed_suite_id",
        "win_rate", "draw_rate", "loss_rate",
        "average_rank", "average_survival_steps",
        "average_kills", "average_boxes", "average_items", "average_bombs",
        "timeout_count", "error_count", "invalid_action_count", "fallback_uses",
        "estimated_mu", "estimated_sigma", "estimated_score",
    ]

    def __init__(self, log_dir: str = "logs/mappo", use_tensorboard: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._train_csv  = self.log_dir / "train.csv"
        self._train_jsonl = self.log_dir / "train.jsonl"
        self._eval_csv   = self.log_dir / "eval.csv"
        self._eval_jsonl = self.log_dir / "eval.jsonl"

        self._train_csv_init  = False
        self._eval_csv_init   = False

        self._tb: Any = None
        if use_tensorboard and _TB_AVAILABLE:
            tb_dir = self.log_dir / "tb"
            tb_dir.mkdir(parents=True, exist_ok=True)
            try:
                self._tb = _SummaryWriter(str(tb_dir))
            except Exception:
                self._tb = None

        self._start_time = time.perf_counter()

    # ── training ──────────────────────────────────────────────────────────────

    def log_train(self, metrics: dict[str, Any]) -> None:
        """Log one PPO update row."""
        row = {f: metrics.get(f, "") for f in self.TRAIN_FIELDS}
        self._append_csv(self._train_csv, self.TRAIN_FIELDS, row, self._train_csv_init)
        self._train_csv_init = True
        self._append_jsonl(self._train_jsonl, metrics)

        if self._tb is not None:
            step = metrics.get("env_steps", 0)
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and k not in ("env_steps",):
                    try:
                        self._tb.add_scalar(f"train/{k}", float(v), global_step=int(step))
                    except Exception:
                        pass

    # ── evaluation ───────────────────────────────────────────────────────────

    def log_eval(self, metrics: dict[str, Any]) -> None:
        """Log one evaluation run row."""
        row = {f: metrics.get(f, "") for f in self.EVAL_FIELDS}
        self._append_csv(self._eval_csv, self.EVAL_FIELDS, row, self._eval_csv_init)
        self._eval_csv_init = True
        self._append_jsonl(self._eval_jsonl, metrics)

        if self._tb is not None:
            step = metrics.get("update", 0)
            for k in ("average_rank", "win_rate", "estimated_score"):
                v = metrics.get(k)
                if isinstance(v, (int, float)):
                    try:
                        self._tb.add_scalar(f"eval/{k}", float(v), global_step=int(step))
                    except Exception:
                        pass

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _append_csv(path: Path, fields: list[str], row: dict, already_init: bool) -> None:
        mode = "a" if already_init and path.exists() else "w"
        with open(path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if mode == "w":
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _append_jsonl(path: Path, record: dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self._start_time

    def close(self) -> None:
        if self._tb is not None:
            try:
                self._tb.close()
            except Exception:
                pass
