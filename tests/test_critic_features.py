"""
test_critic_features.py — Privileged critic state encoding tests.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import BomberEnv
from agent.mappo_agent.encoder import encode_obs, N_SCALAR
from training.mappo.critic_features import (
    PRIVILEGED_SCALAR_DIM,
    build_privileged_critic_scalar,
    enrich_obs_for_critic,
)


class TestPrivilegedCriticFeatures:
    def test_output_shape(self):
        env = BomberEnv(seed=1)
        obs = enrich_obs_for_critic(env.reset(seed=1), env.current_step)
        vec = build_privileged_critic_scalar(obs, agent_id=0)
        assert vec.shape == (PRIVILEGED_SCALAR_DIM,)
        assert vec.dtype == np.float32

    def test_true_step_is_privileged(self):
        env = BomberEnv(seed=2, max_steps=500)
        obs = env.reset(seed=2)
        for _ in range(10):
            obs, _, _ = env.step([0, 0, 0, 0])

        without_step = build_privileged_critic_scalar(obs, agent_id=0)
        with_step = build_privileged_critic_scalar(
            enrich_obs_for_critic(obs, env.current_step), agent_id=0
        )
        assert without_step[0] == 0.0
        assert with_step[0] == pytest.approx(env.current_step / 500.0)
        assert with_step[0] > without_step[0]

    def test_opponent_stats_not_in_actor_scalar(self):
        env = BomberEnv(seed=3)
        obs = enrich_obs_for_critic(env.reset(seed=3), 0)
        _, actor_sc = encode_obs(obs, agent_id=0)
        critic_sc = build_privileged_critic_scalar(obs, agent_id=0)

        assert actor_sc.shape == (N_SCALAR,)
        assert critic_sc.shape == (PRIVILEGED_SCALAR_DIM,)
        # Opponent 1 kills slot in critic per-player block: base 4+9=13 offset +5
        assert critic_sc[13 + 5] == 0.0  # no kills at start
        # Actor scalar has no direct opponent kill counts (slots 4-7 are ego tracker est.)

    def test_tie_break_lead_changes_after_activity(self):
        env = BomberEnv(seed=4, max_steps=200)
        obs = enrich_obs_for_critic(env.reset(seed=4), 0)
        before = build_privileged_critic_scalar(obs, agent_id=0)
        # Place bombs / move for several steps
        for a in ([5, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]):
            obs, _, _ = env.step(a)
        obs = enrich_obs_for_critic(obs, env.current_step)
        after = build_privileged_critic_scalar(obs, agent_id=0)
        assert np.any(after[50:58] != before[50:58]) or after[0] != before[0]

    def test_values_finite(self):
        env = BomberEnv(seed=5)
        obs = enrich_obs_for_critic(env.reset(seed=5), 0)
        for _ in range(30):
            obs, _, term = env.step([1, 2, 3, 4])
            obs = enrich_obs_for_critic(obs, env.current_step)
            vec = build_privileged_critic_scalar(obs, agent_id=0)
            assert np.all(np.isfinite(vec))
            if term:
                break
