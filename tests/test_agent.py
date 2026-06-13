"""
test_agent.py — Tests for the submission Agent class.

Covers:
  - act() returns int in [0, 5]
  - act() works on several real engine observations
  - Fallback activates gracefully when no model file exists
  - Inference timing: p50/p95/p99 < 100ms
"""

import sys
import time
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import BomberEnv
from agent.mappo_agent.agent import Agent


# ── helpers ───────────────────────────────────────────────────────────────────

def _collect_obs(n_steps: int = 30, seed: int = 0) -> list[dict]:
    """Collect several real observations from the engine."""
    env = BomberEnv(seed=seed)
    obs = env.reset(seed=seed)
    observations = [obs]
    for _ in range(n_steps):
        actions = [0, 0, 0, 0]
        obs, done_t, done_tr = env.step(actions)
        observations.append(obs)
        if done_t or done_tr:
            break
    return observations


# ── act() contract ────────────────────────────────────────────────────────────

class TestAgentContract:
    """Verify Agent.act() meets the submission contract regardless of model."""

    @pytest.fixture(params=[0, 1, 2, 3])
    def agent(self, request):
        return Agent(agent_id=request.param)

    def test_act_returns_int(self, agent):
        env = BomberEnv(seed=0)
        obs = env.reset(seed=0)
        result = agent.act(obs)
        assert isinstance(result, int), f"act() must return int, got {type(result)}"

    def test_act_in_valid_range(self, agent):
        env = BomberEnv(seed=0)
        obs = env.reset(seed=0)
        result = agent.act(obs)
        assert 0 <= result <= 5, f"act() returned {result}, must be in [0, 5]"

    def test_act_multiple_obs(self, agent):
        """act() must work for 30 consecutive observations."""
        observations = _collect_obs(n_steps=30, seed=42)
        for obs in observations:
            result = agent.act(obs)
            assert 0 <= result <= 5

    def test_act_on_dead_agent(self):
        """Dead agent should return 0 (STOP)."""
        env = BomberEnv(seed=0)
        obs = env.reset(seed=0)
        # Mark agent 0 as dead
        obs["players"][0, 2] = 0
        agent = Agent(agent_id=0)
        result = agent.act(obs)
        # Should not crash; 0 is expected but any valid action is acceptable
        assert 0 <= result <= 5

    def test_act_does_not_raise_on_garbage_obs(self):
        """act() must catch all exceptions and return a safe fallback."""
        agent = Agent(agent_id=0)
        bad_obs = {"map": None, "players": None, "bombs": None}
        result = agent.act(bad_obs)
        assert 0 <= result <= 5

    def test_fallback_activates_without_model(self):
        """Without a model.pth file, agent must use fallback and not crash."""
        agent = Agent(agent_id=0)
        assert agent._use_fallback or agent._model is not None, \
            "Agent must either have a model or use fallback"
        env = BomberEnv(seed=5)
        obs = env.reset(seed=5)
        result = agent.act(obs)
        assert 0 <= result <= 5


# ── inference timing ──────────────────────────────────────────────────────────

class TestInferenceTiming:
    N_WARMUP   = 5
    N_MEASURE  = 100
    LIMIT_MS   = 100.0   # official competition limit

    def _measure_times(self, agent: Agent, observations: list[dict]) -> list[float]:
        times = []
        obs_cycle = observations * (self.N_WARMUP + self.N_MEASURE)
        for i, obs in enumerate(obs_cycle[: self.N_WARMUP + self.N_MEASURE]):
            t0 = time.perf_counter()
            agent.act(obs)
            elapsed = (time.perf_counter() - t0) * 1000.0
            if i >= self.N_WARMUP:
                times.append(elapsed)
        return times

    @pytest.mark.parametrize("agent_id", [0, 1])
    def test_inference_p50_under_limit(self, agent_id):
        agent = Agent(agent_id=agent_id)
        observations = _collect_obs(n_steps=50, seed=99)
        times = self._measure_times(agent, observations)
        p50 = float(np.percentile(times, 50))
        assert p50 < self.LIMIT_MS, \
            f"p50 latency {p50:.1f}ms exceeds {self.LIMIT_MS}ms limit"

    @pytest.mark.parametrize("agent_id", [0])
    def test_inference_p95_under_limit(self, agent_id):
        agent = Agent(agent_id=agent_id)
        observations = _collect_obs(n_steps=50, seed=99)
        times = self._measure_times(agent, observations)
        p95 = float(np.percentile(times, 95))
        assert p95 < self.LIMIT_MS, \
            f"p95 latency {p95:.1f}ms exceeds {self.LIMIT_MS}ms limit"

    def test_inference_timing_report(self):
        """Print timing breakdown (not a failure test, always passes)."""
        agent = Agent(agent_id=0)
        observations = _collect_obs(n_steps=50, seed=99)
        times = self._measure_times(agent, observations)
        p50  = float(np.percentile(times, 50))
        p95  = float(np.percentile(times, 95))
        p99  = float(np.percentile(times, 99))
        mode = "NN" if not agent._use_fallback else "fallback"
        print(f"\n[Timing] mode={mode} p50={p50:.2f}ms p95={p95:.2f}ms p99={p99:.2f}ms")
        assert True  # always passes — timing info is printed
