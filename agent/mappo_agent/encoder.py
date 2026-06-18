"""
encoder.py — Observation encoder for MAPPO agent.

Converts raw BomberEnv observation dict into:
  spatial : np.ndarray  shape (18, 13, 13)  float32
  scalar  : np.ndarray  shape (28,)          float32

Verified engine direction mappings (from engine/player.py, engine/game.py):
  Action 0  STOP       dx=0,  dy=0
  Action 1  LEFT       dx=-1, dy=0   (row decreases)
  Action 2  RIGHT      dx=+1, dy=0   (row increases)
  Action 3  UP         dx=0,  dy=-1  (col decreases)
  Action 4  DOWN       dx=0,  dy=+1  (col increases)
  Action 5  PLACE_BOMB

Coordinate convention: player.x = row, player.y = col → grid[row, col].

All encoding is safe for empty bomb arrays and variable shapes.
"""

from __future__ import annotations
from collections import deque
import numpy as np

from agent.mappo_agent.safety import legal_action_mask

# ── constants ────────────────────────────────────────────────────────────────
N_SPATIAL   = 18
N_SCALAR    = 28
SCALAR_IDX_LEGAL_START = 22   # legal_stop … legal_bomb (6 features)
GRID_H      = 13
GRID_W      = 13
BOMB_TIMER_MAX   = 7
MAX_BOMBS_LEFT   = 5
MAX_RADIUS       = 5
MAX_STEPS        = 500
MAX_DIST         = 18.0   # max Manhattan distance on 11×11 play area

# Action deltas: action_id → (dx, dy)   [verified against engine source]
ACTION_DELTAS = {
    0: (0,  0),   # STOP
    1: (-1, 0),   # LEFT
    2: ( 1, 0),   # RIGHT
    3: ( 0,-1),   # UP
    4: ( 0, 1),   # DOWN
    # 5 = PLACE_BOMB — no movement
}

# Map tile values
GRASS    = 0
WALL     = 1
BOX      = 2
ITEM_R   = 3   # Item: radius bonus
ITEM_C   = 4   # Item: capacity bonus


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe_grid(obs: dict) -> np.ndarray:
    g = obs.get("map")
    if g is None:
        return np.zeros((GRID_H, GRID_W), dtype=np.int32)
    return np.asarray(g, dtype=np.int32)


def _safe_players(obs: dict) -> np.ndarray:
    p = obs.get("players")
    if p is None:
        return np.zeros((4, 5), dtype=np.int32)
    return np.asarray(p, dtype=np.int32)


def _safe_bombs(obs: dict) -> np.ndarray:
    """Return (N,4) int32 array; safe for empty / 1-D / None input."""
    b = obs.get("bombs")
    if b is None:
        return np.zeros((0, 4), dtype=np.int32)
    arr = np.asarray(b, dtype=np.int32)
    if arr.ndim == 0 or arr.size == 0:
        return np.zeros((0, 4), dtype=np.int32)
    if arr.ndim == 1:
        if arr.shape[0] == 4:
            return arr.reshape(1, 4)
        return np.zeros((0, 4), dtype=np.int32)
    return arr  # (N,4)


def _blast_tiles(grid: np.ndarray, bx: int, by: int, radius: int) -> set[tuple[int,int]]:
    """Cross-shaped blast, blocked by WALL, stopped-but-included at BOX."""
    H, W = grid.shape
    tiles: set[tuple[int,int]] = {(bx, by)}
    for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
        for r in range(1, radius + 1):
            tx, ty = bx + dx*r, by + dy*r
            if not (0 <= tx < H and 0 <= ty < W):
                break
            cell = int(grid[tx, ty])
            if cell == WALL:
                break
            tiles.add((tx, ty))
            if cell == BOX:
                break
    return tiles


def _bfs_nearest(grid: np.ndarray, start: tuple[int,int],
                  targets: set[tuple[int,int]],
                  blocked: set[tuple[int,int]]) -> float:
    """BFS distance from start to nearest target; returns MAX_DIST if unreachable."""
    if not targets:
        return MAX_DIST
    if start in targets:
        return 0.0
    H, W = grid.shape
    q: deque[tuple[tuple[int,int], int]] = deque([(start, 0)])
    seen = {start}
    while q:
        pos, d = q.popleft()
        for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
            nx, ny = pos[0]+dx, pos[1]+dy
            np_ = (nx, ny)
            if np_ in seen:
                continue
            if not (0 < nx < H-1 and 0 < ny < W-1):
                continue
            cell = int(grid[nx, ny])
            if cell in (WALL, BOX) or np_ in blocked:
                continue
            if np_ in targets:
                return float(d + 1)
            seen.add(np_)
            q.append((np_, d + 1))
    return MAX_DIST


# ── main encoder ─────────────────────────────────────────────────────────────

def encode_obs(obs: dict, agent_id: int, tracker=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode a BomberEnv observation for agent `agent_id`.

    Parameters
    ----------
    obs      : dict with keys 'map', 'players', 'bombs'
    agent_id : int in [0, 3]
    tracker  : optional AgentTracker for scalar estimates

    Returns
    -------
    spatial : float32 array (N_SPATIAL=18, GRID_H, GRID_W)
    scalar  : float32 array (N_SCALAR=28,)
      Indices 22–27: legal_stop, legal_left, legal_right, legal_up, legal_down, legal_bomb
    """
    grid    = _safe_grid(obs)
    players = _safe_players(obs)
    bombs   = _safe_bombs(obs)

    H, W = GRID_H, GRID_W
    aid  = int(agent_id)
    n_players = players.shape[0]

    # ── my info ──────────────────────────────────────────────────────────────
    my_row  = int(players[aid, 0]) if n_players > aid else 1
    my_col  = int(players[aid, 1]) if n_players > aid else 1
    my_alive      = int(players[aid, 2]) if n_players > aid else 0
    my_bombs_left = int(players[aid, 3]) if n_players > aid else 0
    my_radius_bonus = int(players[aid, 4]) if n_players > aid else 0
    my_radius = 1 + my_radius_bonus

    # ── bomb position index ──────────────────────────────────────────────────
    bomb_pos_set: set[tuple[int,int]] = set()
    if bombs.shape[0] > 0:
        for b in bombs:
            bomb_pos_set.add((int(b[0]), int(b[1])))

    # ────────────────────────────────────────────────────────────────────────
    # SPATIAL CHANNELS (18)
    # ────────────────────────────────────────────────────────────────────────
    channels = []

    # 0–4: tile type one-hot
    for tile_val in (GRASS, WALL, BOX, ITEM_R, ITEM_C):
        channels.append((grid == tile_val).astype(np.float32))

    # 5: self position
    self_ch = np.zeros((H, W), dtype=np.float32)
    if my_alive:
        self_ch[my_row, my_col] = 1.0
    channels.append(self_ch)

    # 6–8: opponent positions in relative seat order
    for k in (1, 2, 3):
        opp_id = (aid + k) % 4
        opp_ch = np.zeros((H, W), dtype=np.float32)
        if n_players > opp_id and int(players[opp_id, 2]) == 1:
            opp_ch[int(players[opp_id, 0]), int(players[opp_id, 1])] = 1.0
        channels.append(opp_ch)

    # 9: bomb occupancy
    bomb_occ = np.zeros((H, W), dtype=np.float32)
    for bx, by in bomb_pos_set:
        bomb_occ[bx, by] = 1.0
    channels.append(bomb_occ)

    # 10: bomb timer / 7   (highest urgency = smallest timer → largest value)
    bomb_timer_ch = np.zeros((H, W), dtype=np.float32)
    if bombs.shape[0] > 0:
        for b in bombs:
            bx, by, timer = int(b[0]), int(b[1]), int(b[2])
            norm = float(timer) / float(BOMB_TIMER_MAX)
            bomb_timer_ch[bx, by] = max(bomb_timer_ch[bx, by], norm)
    channels.append(bomb_timer_ch)

    # 11: own bomb mask
    own_bomb_ch = np.zeros((H, W), dtype=np.float32)
    if bombs.shape[0] > 0:
        for b in bombs:
            if int(b[3]) == aid:
                own_bomb_ch[int(b[0]), int(b[1])] = 1.0
    channels.append(own_bomb_ch)

    # 12: enemy bomb mask
    enemy_bomb_ch = np.zeros((H, W), dtype=np.float32)
    if bombs.shape[0] > 0:
        for b in bombs:
            if int(b[3]) != aid:
                enemy_bomb_ch[int(b[0]), int(b[1])] = 1.0
    channels.append(enemy_bomb_ch)

    # 13–15: danger_t1 / _t2 / _t3  (blast tiles of bombs with timer ≤ 1/2/3)
    danger = {1: np.zeros((H, W), dtype=np.float32),
              2: np.zeros((H, W), dtype=np.float32),
              3: np.zeros((H, W), dtype=np.float32)}
    if bombs.shape[0] > 0:
        for b in bombs:
            bx, by, timer = int(b[0]), int(b[1]), int(b[2])
            owner_id = int(b[3])
            radius = 1
            if 0 <= owner_id < n_players:
                radius = 1 + int(players[owner_id, 4])
            if timer <= 0:
                continue
            blast = _blast_tiles(grid, bx, by, radius)
            for tx, ty in blast:
                if timer <= 3:
                    danger[3][tx, ty] = 1.0
                if timer <= 2:
                    danger[2][tx, ty] = 1.0
                if timer <= 1:
                    danger[1][tx, ty] = 1.0
    channels.extend([danger[1], danger[2], danger[3]])

    # 16: prospective self-blast if placing bomb now
    prosp_ch = np.zeros((H, W), dtype=np.float32)
    if my_alive and my_bombs_left > 0 and (my_row, my_col) not in bomb_pos_set:
        for tx, ty in _blast_tiles(grid, my_row, my_col, my_radius):
            prosp_ch[tx, ty] = 1.0
    channels.append(prosp_ch)

    # 17: passable mask (grass|item, no bomb)
    passable_ch = np.zeros((H, W), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) in (GRASS, ITEM_R, ITEM_C) and (r, c) not in bomb_pos_set:
                passable_ch[r, c] = 1.0
    channels.append(passable_ch)

    spatial = np.stack(channels, axis=0).astype(np.float32)  # (18, H, W)

    # ────────────────────────────────────────────────────────────────────────
    # SCALAR FEATURES (28)
    # ────────────────────────────────────────────────────────────────────────
    # Pull from tracker if available, else use defaults
    est_step       = float(tracker.estimated_step)   if tracker else 0.0
    est_boxes      = float(tracker.boxes_destroyed)  if tracker else 0.0
    est_items      = float(tracker.items_collected)  if tracker else 0.0
    est_bombs_pl   = float(tracker.bombs_placed)     if tracker else 0.0
    est_kills      = float(tracker.kills)            if tracker else 0.0
    last_action    = int(tracker.last_action)        if tracker else 0
    idle_streak    = float(tracker.idle_streak)      if tracker else 0.0

    alive_opponents = sum(
        1 for k in (1,2,3)
        if n_players > (aid+k)%4 and int(players[(aid+k)%4, 2]) == 1
    )
    opp1_alive = float(int(players[(aid+1)%4, 2]) == 1) if n_players > (aid+1)%4 else 0.0
    opp2_alive = float(int(players[(aid+2)%4, 2]) == 1) if n_players > (aid+2)%4 else 0.0

    on_danger_t1 = float(danger[1][my_row, my_col]) if my_alive else 0.0
    on_danger_t2 = float(danger[2][my_row, my_col]) if my_alive else 0.0
    can_place    = float(
        my_alive and my_bombs_left > 0 and (my_row, my_col) not in bomb_pos_set
    )

    # BFS to nearest item / enemy
    my_pos = (my_row, my_col)
    item_tiles = {
        (r, c)
        for r in range(H) for c in range(W)
        if int(grid[r, c]) in (ITEM_R, ITEM_C)
    }
    enemy_tiles = {
        (int(players[(aid+k)%4, 0]), int(players[(aid+k)%4, 1]))
        for k in (1,2,3)
        if n_players > (aid+k)%4 and int(players[(aid+k)%4, 2]) == 1
    }
    dist_item  = _bfs_nearest(grid, my_pos, item_tiles,  bomb_pos_set) / MAX_DIST
    dist_enemy = _bfs_nearest(grid, my_pos, enemy_tiles, bomb_pos_set) / MAX_DIST

    # One-hot last action (6)
    last_action_oh = np.zeros(6, dtype=np.float32)
    if 0 <= last_action <= 5:
        last_action_oh[last_action] = 1.0

    legal_feats = legal_action_mask(obs, aid).astype(np.float32)

    scalar = np.array([
        float(my_bombs_left)   / MAX_BOMBS_LEFT,   # 0
        float(my_radius)       / MAX_RADIUS,        # 1
        est_step               / MAX_STEPS,         # 2
        float(alive_opponents) / 3.0,               # 3
        est_boxes              / 20.0,              # 4
        est_items              / 10.0,              # 5
        est_bombs_pl           / 20.0,              # 6
        est_kills              / 3.0,               # 7
        *last_action_oh,                            # 8-13
        on_danger_t1,                               # 14
        on_danger_t2,                               # 15
        can_place,                                  # 16
        dist_item,                                  # 17
        dist_enemy,                                 # 18
        idle_streak            / 10.0,              # 19
        opp1_alive,                                 # 20
        opp2_alive,                                 # 21
        *legal_feats,                               # 22-27 legal mask
    ], dtype=np.float32)

    assert spatial.shape == (N_SPATIAL, GRID_H, GRID_W), f"spatial shape {spatial.shape}"
    assert scalar.shape  == (N_SCALAR,),                 f"scalar shape  {scalar.shape}"
    return spatial, scalar
