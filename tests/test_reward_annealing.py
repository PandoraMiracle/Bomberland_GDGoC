"""
test_reward_annealing.py — Tests for dense reward annealing schedule.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.mappo.reward_builder import (
    RewardBuilder,
    TERMINAL,
    compute_annealed_dense_coef,
)
from training.mappo.config import MAPPOConfig


TOTAL = 1_000_000


class TestAnnealingSchedule:
    def test_early_phase_first_40_percent(self):
        for steps in (0, 100_000, 399_999):
            coef, phase = compute_annealed_dense_coef(
                steps, TOTAL, enabled=True,
                coef_early=1.0, coef_mid=0.5, coef_late=0.2,
            )
            assert coef == 1.0
            assert phase == "early"

    def test_mid_phase_middle_40_percent(self):
        for steps in (400_000, 600_000, 799_999):
            coef, phase = compute_annealed_dense_coef(
                steps, TOTAL, enabled=True,
                coef_early=1.0, coef_mid=0.5, coef_late=0.2,
            )
            assert coef == 0.5
            assert phase == "mid"

    def test_late_phase_final_20_percent(self):
        for steps in (800_000, 900_000, 1_000_000):
            coef, phase = compute_annealed_dense_coef(
                steps, TOTAL, enabled=True,
                coef_early=1.0, coef_mid=0.5, coef_late=0.2,
            )
            assert coef == 0.2
            assert phase == "late"

    def test_disabled_uses_fixed_coef(self):
        coef, phase = compute_annealed_dense_coef(
            900_000, TOTAL, enabled=False, fixed_coef=0.75,
        )
        assert coef == 0.75
        assert phase == "fixed"

    def test_config_integration(self):
        cfg = MAPPOConfig(total_env_steps=TOTAL, dense_reward_anneal=True)
        coef, phase = compute_annealed_dense_coef(
            850_000,
            cfg.total_env_steps,
            enabled=cfg.dense_reward_anneal,
            fixed_coef=cfg.dense_reward_coef,
            coef_early=cfg.dense_reward_coef_early,
            coef_mid=cfg.dense_reward_coef_mid,
            coef_late=cfg.dense_reward_coef_late,
            mid_start=cfg.dense_reward_anneal_mid_start,
            late_start=cfg.dense_reward_anneal_late_start,
        )
        assert coef == 0.2
        assert phase == "late"


class TestTerminalUnscaled:
    def test_terminal_not_scaled_by_dense_coef(self):
        rb = RewardBuilder(dense_reward_coef=0.0)
        assert rb.compute_terminal([0, 1, 2, 3], agent_id=0) == TERMINAL["rank_0_unique"]
        assert rb.compute_terminal([0, 0, 1, 2], agent_id=0) == TERMINAL["rank_0_shared"]
        assert rb.compute_terminal([1, 0, 2, 3], agent_id=0) == TERMINAL["rank_1"]

    def test_dense_scaled_setter(self):
        rb = RewardBuilder(dense_reward_coef=1.0)
        rb.set_dense_reward_coef(0.2)
        assert rb.dense_reward_coef == 0.2
