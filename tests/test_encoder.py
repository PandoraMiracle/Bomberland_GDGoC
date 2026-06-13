"""
test_encoder.py — Tests for the MAPPO observation encoder.

Covers:
  - Output shapes (18, 13, 13) spatial + (22,) scalar
  - Engine direction verification: our ACTION_DELTAS match game.py
  - Safe handling of empty bombs array
  - Seat-relative opponent encoding correctness
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import BomberEnv
from agent.mappo_agent.encoder import (
    encode_obs, N_SPATIAL, N_SCALAR, GRID_H, GRID_W, ACTION_DELTAS
)
from agent.mappo_agent.tracker import AgentTracker


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def real_obs():
    """Fresh observation from the actual engine."""
    env = BomberEnv(seed=42)
    return env.reset(seed=42)


@pytest.fixture
def empty_bombs_obs(real_obs):
    """Observation with no bombs active."""
    obs = {k: v.copy() if hasattr(v, 'copy') else v for k, v in real_obs.items()}
    obs["bombs"] = np.zeros((0, 4), dtype=np.int8)
    return obs


# ── shape tests ───────────────────────────────────────────────────────────────

class TestEncoderShapes:
    def test_spatial_shape(self, real_obs):
        spatial, scalar = encode_obs(real_obs, agent_id=0)
        assert spatial.shape == (N_SPATIAL, GRID_H, GRID_W), \
            f"Expected ({N_SPATIAL},{GRID_H},{GRID_W}), got {spatial.shape}"

    def test_scalar_shape(self, real_obs):
        spatial, scalar = encode_obs(real_obs, agent_id=0)
        assert scalar.shape == (N_SCALAR,), \
            f"Expected ({N_SCALAR},), got {scalar.shape}"

    def test_spatial_dtype(self, real_obs):
        spatial, _ = encode_obs(real_obs, agent_id=0)
        assert spatial.dtype == np.float32

    def test_scalar_dtype(self, real_obs):
        _, scalar = encode_obs(real_obs, agent_id=0)
        assert scalar.dtype == np.float32

    @pytest.mark.parametrize("agent_id", [0, 1, 2, 3])
    def test_all_agent_ids(self, real_obs, agent_id):
        spatial, scalar = encode_obs(real_obs, agent_id=agent_id)
        assert spatial.shape == (N_SPATIAL, GRID_H, GRID_W)
        assert scalar.shape  == (N_SCALAR,)

    def test_empty_bombs(self, empty_bombs_obs):
        spatial, scalar = encode_obs(empty_bombs_obs, agent_id=0)
        assert spatial.shape == (N_SPATIAL, GRID_H, GRID_W)
        assert scalar.shape  == (N_SCALAR,)

    def test_values_finite(self, real_obs):
        spatial, scalar = encode_obs(real_obs, agent_id=0)
        assert np.all(np.isfinite(spatial))
        assert np.all(np.isfinite(scalar))

    def test_values_in_range(self, real_obs):
        spatial, scalar = encode_obs(real_obs, agent_id=0)
        # Spatial channels should be in [0, 1]
        assert float(spatial.min()) >= -1e-6
        assert float(spatial.max()) <= 1.0 + 1e-6


# ── direction verification ─────────────────────────────────────────────────────

class TestDirectionMapping:
    """
    Unit test: our ACTION_DELTAS must agree with engine/game.py.
    Strategy: spawn fresh env, apply each action, observe player movement.
    """

    def _apply_action_single(self, action: int, agent_id: int = 0) -> tuple[int,int,int,int]:
        """
        Returns (before_row, before_col, after_row, after_col) for given action
        on agent 0 starting from (1,1) on a fresh map (guaranteed clear corner).
        """
        env = BomberEnv(seed=0)
        obs = env.reset(seed=0)
        p_before = obs["players"][agent_id]
        brow, bcol = int(p_before[0]), int(p_before[1])

        actions = [0] * 4
        actions[agent_id] = action
        obs2, _, _ = env.step(actions)
        p_after = obs2["players"][agent_id]
        arow, acol = int(p_after[0]), int(p_after[1])
        return brow, bcol, arow, acol

    def test_stop_no_movement(self):
        br, bc, ar, ac = self._apply_action_single(0)
        assert (ar, ac) == (br, bc), "STOP should not move player"

    def test_left_decreases_row(self):
        br, bc, ar, ac = self._apply_action_single(1)
        # Agent 0 starts at (1,1); LEFT = dx=-1, but can't go to row=0 (wall)
        # So player stays at (1,1) — verify it's at most br, not br+1 or bc+1
        assert ac == bc, "LEFT should not change col"
        # Engine won't move into border wall, so ar <= br
        assert ar <= br, "LEFT should decrease row or stay (if at border)"

    def test_right_increases_row(self):
        br, bc, ar, ac = self._apply_action_single(2)
        assert ac == bc, "RIGHT should not change col"
        assert ar >= br, "RIGHT should increase row or stay"

    def test_up_decreases_col(self):
        br, bc, ar, ac = self._apply_action_single(3)
        assert ar == br, "UP should not change row"
        assert ac <= bc, "UP should decrease col or stay (if at border)"

    def test_down_increases_col(self):
        br, bc, ar, ac = self._apply_action_single(4)
        assert ar == br, "DOWN should not change row"
        assert ac >= bc, "DOWN should increase col or stay"

    def test_our_deltas_match_engine(self):
        """Explicit delta-match: if movement succeeds, delta must match our table."""
        env = BomberEnv(seed=1)
        obs = env.reset(seed=1)
        # Agent 1 starts at (11,11) — try all movement actions
        for action in (1, 2, 3, 4):
            env2 = BomberEnv(seed=1)
            o = env2.reset(seed=1)
            br, bc = int(o["players"][1][0]), int(o["players"][1][1])
            acts = [0, action, 0, 0]
            o2, _, _ = env2.step(acts)
            ar, ac = int(o2["players"][1][0]), int(o2["players"][1][1])
            actual_dx, actual_dy = ar - br, ac - bc
            expected_dx, expected_dy = ACTION_DELTAS[action]
            # Movement may be blocked — if it moved, it must match our expected delta
            if (ar, ac) != (br, bc):
                assert (actual_dx, actual_dy) == (expected_dx, expected_dy), \
                    f"Action {action}: expected delta {(expected_dx,expected_dy)}, got {(actual_dx,actual_dy)}"


# ── seat-relative encoding test ───────────────────────────────────────────────

class TestSeatRelativeEncoding:
    def test_self_channel_marks_own_position(self, real_obs):
        for aid in range(4):
            spatial, _ = encode_obs(real_obs, agent_id=aid)
            players = real_obs["players"]
            if int(players[aid, 2]) == 1:
                row, col = int(players[aid, 0]), int(players[aid, 1])
                assert spatial[5, row, col] == 1.0, \
                    f"Agent {aid}: self channel should mark ({row},{col})"

    def test_opponent_channels_non_overlapping_with_self(self, real_obs):
        for aid in range(4):
            spatial, _ = encode_obs(real_obs, agent_id=aid)
            players = real_obs["players"]
            if int(players[aid, 2]) != 1:
                continue
            row, col = int(players[aid, 0]), int(players[aid, 1])
            # Channels 6,7,8 should NOT include self position (players overlap is possible,
            # but opponent channels are keyed to opponent ids, not self)
            for opp_ch in (6, 7, 8):
                opp_id = (aid + (opp_ch - 5)) % 4
                if opp_id == aid:
                    continue
                opp_row = int(players[opp_id, 0])
                opp_col = int(players[opp_id, 1])
                if int(players[opp_id, 2]) == 1:
                    assert spatial[opp_ch, opp_row, opp_col] == 1.0


# ── tracker integration ───────────────────────────────────────────────────────

class TestTrackerIntegration:
    def test_tracker_changes_scalar_features(self):
        env = BomberEnv(seed=7, max_steps=200)
        obs = env.reset(seed=7)
        tracker = AgentTracker(agent_id=0)
        tracker.reset()

        _, scalar_before = encode_obs(obs, agent_id=0, tracker=tracker)
        assert scalar_before[2] == 0.0  # est_step / MAX_STEPS
        assert scalar_before[6] == 0.0  # est_bombs_pl

        actions = [5, 0, 0, 0]  # place bomb
        obs, _, _ = env.step(actions)
        tracker.update(obs, actions[0])

        _, scalar_after = encode_obs(obs, agent_id=0, tracker=tracker)
        assert scalar_after.shape == (N_SCALAR,)
        assert scalar_after[2] > scalar_before[2]
        assert scalar_after[6] >= scalar_before[6]

    def test_tracker_episode_rollout_increments(self):
        env = BomberEnv(seed=11, max_steps=80)
        obs = env.reset(seed=11)
        tracker = AgentTracker(agent_id=0)

        max_step_scalar = 0.0
        for _ in range(40):
            _, scalar = encode_obs(obs, agent_id=0, tracker=tracker)
            max_step_scalar = max(max_step_scalar, float(scalar[2]))
            actions = [1, 0, 0, 0]
            obs, term, trunc = env.step(actions)
            tracker.update(obs, actions[0])
            if term or trunc:
                break

        assert max_step_scalar > 0.0
        assert tracker.estimated_step > 0

    def test_without_tracker_defaults_to_zeros(self, real_obs):
        _, scalar = encode_obs(real_obs, agent_id=0, tracker=None)
        assert scalar.shape == (N_SCALAR,)
        assert scalar[4] == 0.0  # est_boxes
        assert scalar[5] == 0.0  # est_items
        assert scalar[6] == 0.0  # est_bombs_pl
        assert scalar[7] == 0.0  # est_kills
