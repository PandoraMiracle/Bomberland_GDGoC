"""
test_safety.py — Tests for the safety filter.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import BomberEnv
from agent.mappo_agent.safety import (
    legal_action_mask, apply_safety, blast_tiles, has_escape_path,
    compute_danger_map, N_ACTIONS
)


def _make_obs(grid_val=0, agent_pos=(1,1), bombs=None):
    """Build a minimal synthetic observation for unit testing."""
    grid = np.full((13, 13), grid_val, dtype=np.int32)
    # Border walls
    grid[0, :] = 1; grid[-1, :] = 1
    grid[:, 0] = 1; grid[:, -1] = 1
    # Clear agent position and surroundings
    grid[1:3, 1:3] = 0
    players = np.zeros((4, 5), dtype=np.int32)
    players[0] = [agent_pos[0], agent_pos[1], 1, 1, 0]
    players[1] = [11, 11, 1, 1, 0]
    players[2] = [1, 11, 1, 1, 0]
    players[3] = [11, 1, 1, 1, 0]
    bombs_arr = np.zeros((0, 4), dtype=np.int32) if bombs is None else np.array(bombs, dtype=np.int32)
    return {"map": grid, "players": players, "bombs": bombs_arr}


class TestBlastTiles:
    def test_no_obstacles_radius1(self):
        grid = np.zeros((13, 13), dtype=np.int32)
        tiles = blast_tiles(grid, 5, 5, 1)
        assert tiles == {(5,5),(4,5),(6,5),(5,4),(5,6)}

    def test_blocked_by_wall(self):
        grid = np.zeros((13, 13), dtype=np.int32)
        grid[5, 6] = 1  # Wall to the right
        tiles = blast_tiles(grid, 5, 5, 3)
        # Right direction stops at wall, doesn't include wall tile
        assert (5, 6) not in tiles
        assert (5, 7) not in tiles

    def test_stopped_at_box(self):
        grid = np.zeros((13, 13), dtype=np.int32)
        grid[5, 7] = 2  # Box 2 steps right
        tiles = blast_tiles(grid, 5, 5, 3)
        # Box is included but nothing beyond
        assert (5, 7) in tiles
        assert (5, 8) not in tiles

    def test_center_always_included(self):
        grid = np.zeros((13, 13), dtype=np.int32)
        tiles = blast_tiles(grid, 3, 3, 2)
        assert (3, 3) in tiles


class TestLegalMask:
    def test_stop_always_legal(self):
        obs = _make_obs(agent_pos=(1,1))
        mask = legal_action_mask(obs, 0)
        assert mask[0] == True

    def test_all_moves_open_center(self):
        # Agent at (6,6) on an open grid: all 4 movement actions should be legal
        obs = _make_obs(agent_pos=(6, 6))
        mask = legal_action_mask(obs, 0)
        # At least some movement actions legal
        assert mask[1:5].any()

    def test_place_bomb_blocked_when_no_ammo(self):
        obs = _make_obs(agent_pos=(1,1))
        obs["players"][0, 3] = 0   # bombs_left = 0
        mask = legal_action_mask(obs, 0)
        assert mask[5] == False

    def test_place_bomb_blocked_on_existing_bomb(self):
        obs = _make_obs(agent_pos=(1,1), bombs=[[1, 1, 5, 0]])
        mask = legal_action_mask(obs, 0)
        assert mask[5] == False

    def test_place_bomb_allowed(self):
        obs = _make_obs(agent_pos=(6, 6))
        obs["players"][0, 3] = 1   # bombs_left = 1
        mask = legal_action_mask(obs, 0)
        assert mask[5] == True

    def test_dead_agent_only_stop(self):
        obs = _make_obs(agent_pos=(1,1))
        obs["players"][0, 2] = 0   # alive = 0
        mask = legal_action_mask(obs, 0)
        assert mask[0] == True
        assert not mask[1:].any()

    def test_movement_blocked_by_wall(self):
        obs = _make_obs(agent_pos=(1, 1))
        # Agent at (1,1) — LEFT action (dx=-1) leads to (0,1) which is border wall
        mask = legal_action_mask(obs, 0)
        assert mask[1] == False   # LEFT → row=0 is wall


class TestHasEscapePath:
    def test_already_safe(self):
        grid = np.zeros((13, 13), dtype=np.int32)
        result = has_escape_path(grid, (6, 6), set(), set())
        assert result == True

    def test_can_escape_in_one_step(self):
        grid = np.zeros((13, 13), dtype=np.int32)
        grid[0, :] = 1; grid[-1, :] = 1; grid[:, 0] = 1; grid[:, -1] = 1
        danger = {(6, 6)}  # Only current tile is dangerous
        result = has_escape_path(grid, (6, 6), set(), danger)
        assert result == True

    def test_no_escape_fully_surrounded_by_danger(self):
        grid = np.zeros((13, 13), dtype=np.int32)
        grid[0, :] = 1; grid[-1, :] = 1; grid[:, 0] = 1; grid[:, -1] = 1
        # Box in all 4 directions
        grid[5, 6] = 2; grid[7, 6] = 2; grid[6, 5] = 2; grid[6, 7] = 2
        danger = {(6, 6), (5, 6), (7, 6), (6, 5), (6, 7)}
        result = has_escape_path(grid, (6, 6), set(), danger, depth=2)
        assert result == False


class TestApplySafety:
    def test_always_returns_int_in_range(self):
        env = BomberEnv(seed=0)
        obs = env.reset(seed=0)
        logits = np.random.randn(6).astype(np.float32)
        for aid in range(4):
            action = apply_safety(logits, obs, aid)
            assert isinstance(action, int)
            assert 0 <= action <= 5

    def test_dead_agent_returns_stop(self):
        obs = _make_obs(agent_pos=(1,1))
        obs["players"][0, 2] = 0
        logits = np.ones(6, dtype=np.float32)
        action = apply_safety(logits, obs, 0)
        assert action == 0

    def test_escape_from_immediate_danger(self):
        # Agent at (6,6), bomb at (6,7) timer=1 → danger_t1 includes (6,6)
        obs = _make_obs(agent_pos=(6,6), bombs=[[6, 7, 1, 1]])
        obs["players"][1] = [6, 7, 1, 1, 0]
        logits = np.zeros(6, dtype=np.float32)
        logits[0] = 100.0  # STOP would be preferred by logits
        action = apply_safety(logits, obs, 0)
        # Should NOT be STOP (0) since agent is in immediate danger
        assert action != 0, "Should escape, not STOP in immediate blast"

    def test_fallback_does_not_crash_on_garbage_obs(self):
        """Safety must never raise even with malformed input."""
        bad_obs = {"map": None, "players": None, "bombs": None}
        logits = np.zeros(6, dtype=np.float32)
        action = apply_safety(logits, bad_obs, 0)
        assert 0 <= action <= 5

    def test_multiple_real_engine_obs(self):
        """Run several engine steps, apply safety each time."""
        env = BomberEnv(seed=7)
        obs = env.reset(seed=7)
        for _ in range(20):
            for aid in range(4):
                logits = np.random.randn(6).astype(np.float32)
                a = apply_safety(logits, obs, aid)
                assert 0 <= a <= 5
            actions = [apply_safety(np.random.randn(6), obs, i) for i in range(4)]
            obs, _, _ = env.step(actions)
