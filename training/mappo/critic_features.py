"""
critic_features.py — Privileged global state for the centralized critic (training only).

The actor receives ego-centric encode_obs() output (28-dim scalar + 18 spatial channels).
The critic additionally receives a fixed-size privileged vector built from full
simulator state that is NOT fed to the actor at inference time.

Feature layout (PRIVILEGED_SCALAR_DIM = 64)
────────────────────────────────────────────
[0:4]   Global match context (critic-only true step from env)
  0  env_step / MAX_STEPS
  1  time_remaining = (MAX_STEPS - step) / MAX_STEPS
  2  n_alive / 4
  3  n_bombs_on_map / 10

[4:40]  Per-player block (4 players × 9 features) — absolute seat order 0..3
  For player i, base = 4 + i*9:
    +0 alive (0/1)
    +1 row / 12
    +2 col / 12
    +3 bombs_left / MAX_BOMBS
    +4 blast_radius / MAX_RADIUS
    +5 true kills / MAX_KILLS        ← from obs['stats'], not actor tracker
    +6 true boxes / MAX_BOXES
    +7 true items / MAX_ITEMS
    +8 true bombs_placed / MAX_BOMBS_PL

[40:44] Map aggregates (global counts)
  40 boxes_remaining / MAX_BOXES
  41 items_on_map / MAX_ITEMS
  42 passable_tiles / 121
  43 wall_tiles / 121

[44:50] Bomb aggregates (global, all owners)
  44 min_timer / BOMB_TIMER_MAX
  45 mean_timer / BOMB_TIMER_MAX
  46 imminent_bombs (timer<=1) / 10
  47 own_bombs (training seat) / 10
  48 enemy_bombs / 10
  49 max_bomb_radius_on_map / MAX_RADIUS

[50:58] Tie-break / ranking features for training seat (agent_id)
  50 ego kills / MAX_KILLS
  51 ego boxes / MAX_BOXES
  52 ego items / MAX_ITEMS
  53 ego bombs_placed / MAX_BOMBS_PL
  54 kill_lead vs best opponent / MAX_KILLS
  55 box_lead vs best opponent / MAX_BOXES
  56 item_lead vs best opponent / MAX_ITEMS
  57 bomb_lead vs best opponent / MAX_BOMBS_PL

[58:64] Ego-relative geometry to alive opponents (max 3 opponents)
  58 min manhattan dist to any alive opponent / MAX_DIST
  59 mean manhattan dist to alive opponents / MAX_DIST
  60-63 delta row/col to closest alive opponent / 12 (2 dims), second-closest (2 dims)

Actor does NOT receive: true env_step, other players' cumulative stats, global bomb
aggregates, tie-break margins, or opponent-relative geometry in scalar form.
"""

from __future__ import annotations

import numpy as np

PRIVILEGED_SCALAR_DIM = 64

MAX_STEPS = 500
MAX_BOMBS = 5
MAX_RADIUS = 5
MAX_KILLS = 3
MAX_BOXES = 40
MAX_ITEMS = 10
MAX_BOMBS_PL = 20
BOMB_TIMER_MAX = 7
MAX_DIST = 18.0
GRID_CELLS = 121  # 11x11 playable interior approx; normalization constant

# Map tile ids (engine/map.py)
_GRASS, _WALL, _BOX = 0, 1, 2
_ITEM_R, _ITEM_C = 3, 4


def _safe_players(obs: dict) -> np.ndarray:
    p = obs.get("players")
    if p is None:
        return np.zeros((4, 5), dtype=np.int32)
    return np.asarray(p, dtype=np.int32)


def _safe_bombs(obs: dict) -> np.ndarray:
    b = obs.get("bombs")
    if b is None:
        return np.zeros((0, 4), dtype=np.int32)
    arr = np.asarray(b, dtype=np.int32)
    if arr.ndim == 0 or arr.size == 0:
        return np.zeros((0, 4), dtype=np.int32)
    if arr.ndim == 1 and arr.shape[0] == 4:
        return arr.reshape(1, 4)
    return arr


def _safe_stats(obs: dict, player_id: int) -> dict[str, int]:
    stats = obs.get("stats", {})
    if isinstance(stats, dict):
        raw = stats.get(player_id, stats.get(str(player_id), {}))
        if isinstance(raw, dict):
            return {
                "kills": int(raw.get("kills", 0)),
                "boxes": int(raw.get("boxes", 0)),
                "items": int(raw.get("items", 0)),
                "bombs": int(raw.get("bombs", 0)),
            }
    return {"kills": 0, "boxes": 0, "items": 0, "bombs": 0}


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def build_privileged_critic_scalar(
    obs: dict,
    agent_id: int = 0,
) -> np.ndarray:
    """
    Build privileged critic vector from a full BomberEnv observation.

    Expects optional key ``env_step`` (injected by vec_env workers); falls back to 0.
    """
    vec = np.zeros(PRIVILEGED_SCALAR_DIM, dtype=np.float32)

    players = _safe_players(obs)
    bombs = _safe_bombs(obs)
    grid = np.asarray(obs.get("map", np.zeros((13, 13))), dtype=np.int32)

    step = int(obs.get("env_step", 0))
    n_players = min(4, players.shape[0])
    aid = int(agent_id)

    # ── [0:4] global context ─────────────────────────────────────────────────
    vec[0] = float(step) / float(MAX_STEPS)
    vec[1] = float(max(MAX_STEPS - step, 0)) / float(MAX_STEPS)
    if n_players > 0:
        vec[2] = float(np.sum(players[:n_players, 2])) / 4.0
    vec[3] = float(bombs.shape[0]) / 10.0

    # ── [4:40] per-player true stats (seat order) ────────────────────────────
    for i in range(4):
        base = 4 + i * 9
        if n_players <= i:
            continue
        st = _safe_stats(obs, i)
        vec[base]     = float(int(players[i, 2]))
        vec[base + 1] = float(int(players[i, 0])) / 12.0
        vec[base + 2] = float(int(players[i, 1])) / 12.0
        vec[base + 3] = float(int(players[i, 3])) / float(MAX_BOMBS)
        vec[base + 4] = float(1 + int(players[i, 4])) / float(MAX_RADIUS)
        vec[base + 5] = float(st["kills"]) / float(MAX_KILLS)
        vec[base + 6] = float(st["boxes"]) / float(MAX_BOXES)
        vec[base + 7] = float(st["items"]) / float(MAX_ITEMS)
        vec[base + 8] = float(st["bombs"]) / float(MAX_BOMBS_PL)

    # ── [40:44] map aggregates ─────────────────────────────────────────────────
    if grid.size > 0:
        vec[40] = float(np.sum(grid == _BOX)) / float(MAX_BOXES)
        vec[41] = float(np.sum((grid == _ITEM_R) | (grid == _ITEM_C))) / float(MAX_ITEMS)
        vec[42] = float(np.sum((grid == _GRASS) | (grid == _ITEM_R) | (grid == _ITEM_C))) / float(GRID_CELLS)
        vec[43] = float(np.sum(grid == _WALL)) / float(GRID_CELLS)

    # ── [44:50] bomb aggregates ──────────────────────────────────────────────
    if bombs.shape[0] > 0:
        timers = bombs[:, 2].astype(np.float32)
        vec[44] = float(np.min(timers)) / float(BOMB_TIMER_MAX)
        vec[45] = float(np.mean(timers)) / float(BOMB_TIMER_MAX)
        vec[46] = float(np.sum(timers <= 1)) / 10.0
        radii = []
        for b in bombs:
            oid = int(b[3])
            rad = 1 + (int(players[oid, 4]) if 0 <= oid < n_players else 0)
            radii.append(rad)
        vec[49] = float(max(radii)) / float(MAX_RADIUS)

    if bombs.shape[0] > 0 and n_players > aid:
        own = int(np.sum(bombs[:, 3] == aid))
        enemy = int(bombs.shape[0] - own)
        vec[47] = float(own) / 10.0
        vec[48] = float(enemy) / 10.0

    # ── [50:58] tie-break features for training seat ─────────────────────────
    if n_players > aid:
        ego = _safe_stats(obs, aid)
        opp_stats = [_safe_stats(obs, i) for i in range(n_players) if i != aid]

        vec[50] = float(ego["kills"]) / float(MAX_KILLS)
        vec[51] = float(ego["boxes"]) / float(MAX_BOXES)
        vec[52] = float(ego["items"]) / float(MAX_ITEMS)
        vec[53] = float(ego["bombs"]) / float(MAX_BOMBS_PL)

        if opp_stats:
            best_k = max(s["kills"] for s in opp_stats)
            best_b = max(s["boxes"] for s in opp_stats)
            best_i = max(s["items"] for s in opp_stats)
            best_p = max(s["bombs"] for s in opp_stats)
            vec[54] = float(ego["kills"] - best_k) / float(MAX_KILLS)
            vec[55] = float(ego["boxes"] - best_b) / float(MAX_BOXES)
            vec[56] = float(ego["items"] - best_i) / float(MAX_ITEMS)
            vec[57] = float(ego["bombs"] - best_p) / float(MAX_BOMBS_PL)

    # ── [58:64] ego-relative opponent geometry ─────────────────────────────
    if n_players > aid and int(players[aid, 2]) == 1:
        ego_pos = (int(players[aid, 0]), int(players[aid, 1]))
        opp_positions = [
            (int(players[i, 0]), int(players[i, 1]))
            for i in range(n_players)
            if i != aid and int(players[i, 2]) == 1
        ]
        if opp_positions:
            pairs = sorted(
                (_manhattan(ego_pos, p), p) for p in opp_positions
            )
            vec[58] = pairs[0][0] / MAX_DIST
            vec[59] = float(np.mean([d for d, _ in pairs])) / MAX_DIST
            closest = pairs[0][1]
            vec[60] = float(closest[0] - ego_pos[0]) / 12.0
            vec[61] = float(closest[1] - ego_pos[1]) / 12.0
            if len(pairs) > 1:
                second = pairs[1][1]
                vec[62] = float(second[0] - ego_pos[0]) / 12.0
                vec[63] = float(second[1] - ego_pos[1]) / 12.0

    assert vec.shape == (PRIVILEGED_SCALAR_DIM,)
    return vec


def enrich_obs_for_critic(obs: dict, env_step: int) -> dict:
    """Attach true simulator step for privileged critic encoding (training only)."""
    enriched = dict(obs)
    enriched["env_step"] = int(env_step)
    return enriched
