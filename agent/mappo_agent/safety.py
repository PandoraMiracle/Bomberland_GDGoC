"""
safety.py — Rule-based safety filter.

Used both during training (to mask illegal/suicidal actions before policy loss)
and in the final submitted agent (as last-mile guard before returning action).

Contract:
  apply_safety(logits, obs, agent_id, tracker=None) -> int in [0, 5]

Never prints or logs.  Never raises outside of its own internal try/except.
"""

from __future__ import annotations
from collections import deque
import numpy as np

# ── engine constants (must match engine/player.py) ───────────────────────────
GRASS   = 0
WALL    = 1
BOX     = 2
ITEM_R  = 3
ITEM_C  = 4

# Action → (dx, dy)  [verified from engine/game.py lines 56-59]
ACTION_DELTAS: dict[int, tuple[int, int]] = {
    0: ( 0,  0),   # STOP
    1: (-1,  0),   # LEFT  (row-1)
    2: ( 1,  0),   # RIGHT (row+1)
    3: ( 0, -1),   # UP    (col-1)
    4: ( 0,  1),   # DOWN  (col+1)
    # 5: PLACE_BOMB — special handled below
}
N_ACTIONS  = 6
BFS_DEPTH  = 8    # horizon for escape-path search


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe_grid(obs: dict) -> np.ndarray:
    g = obs.get("map")
    if g is None:
        return np.zeros((13, 13), dtype=np.int32)
    return np.asarray(g, dtype=np.int32)


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
    if arr.ndim == 1:
        return arr.reshape(1, 4) if arr.shape[0] == 4 else np.zeros((0, 4), dtype=np.int32)
    return arr


def blast_tiles(grid: np.ndarray, bx: int, by: int, radius: int) -> set[tuple[int, int]]:
    """
    Compute cross-shaped explosion tiles for a bomb at (bx, by) with given radius.
    Blocked by WALL; stops at (and includes) BOX.
    """
    H, W = grid.shape
    tiles: set[tuple[int, int]] = {(bx, by)}
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        for r in range(1, radius + 1):
            tx, ty = bx + dx * r, by + dy * r
            if not (0 <= tx < H and 0 <= ty < W):
                break
            cell = int(grid[tx, ty])
            if cell == WALL:
                break
            tiles.add((tx, ty))
            if cell == BOX:
                break
    return tiles


def compute_danger_map(obs: dict) -> dict[int, set[tuple[int, int]]]:
    """
    Returns {timer_threshold: set_of_dangerous_tiles} for t=1,2,3.
    Danger at threshold t means the tile will be in a blast within t steps.
    """
    grid    = _safe_grid(obs)
    players = _safe_players(obs)
    bombs   = _safe_bombs(obs)
    n_pl    = players.shape[0]

    danger: dict[int, set[tuple[int, int]]] = {1: set(), 2: set(), 3: set()}

    for b in bombs:
        bx, by, timer = int(b[0]), int(b[1]), int(b[2])
        owner_id = int(b[3])
        if timer <= 0:
            continue
        radius = 1
        if 0 <= owner_id < n_pl:
            radius = 1 + int(players[owner_id, 4])
        tiles = blast_tiles(grid, bx, by, radius)
        for t in (1, 2, 3):
            if timer <= t:
                danger[t] |= tiles
    return danger


def has_escape_path(
    grid: np.ndarray,
    start: tuple[int, int],
    bomb_pos: set[tuple[int, int]],
    danger_tiles: set[tuple[int, int]],
    depth: int = BFS_DEPTH,
) -> bool:
    """
    BFS search for a tile reachable within `depth` steps that is not in
    danger_tiles and not blocked.  Returns True if such a path exists.
    """
    H, W = grid.shape
    if start not in danger_tiles:
        return True  # already safe
    q: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
    seen = {start}
    while q:
        pos, d = q.popleft()
        if d >= depth:
            continue
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = pos[0] + dx, pos[1] + dy
            npos = (nx, ny)
            if npos in seen:
                continue
            if not (0 < nx < H - 1 and 0 < ny < W - 1):
                continue
            cell = int(grid[nx, ny])
            if cell in (WALL, BOX):
                continue
            if npos in bomb_pos:
                continue
            if npos not in danger_tiles:
                return True
            seen.add(npos)
            q.append((npos, d + 1))
    return False


def legal_action_mask(obs: dict, agent_id: int) -> np.ndarray:
    """
    Returns bool array of shape (6,) indicating which actions are physically legal.

    Rules:
    - Movement blocked by WALL, BOX, or existing bomb on that tile.
    - PLACE_BOMB blocked if bombs_left <= 0 or bomb already on current tile.
    - STOP is always legal.
    """
    grid    = _safe_grid(obs)
    players = _safe_players(obs)
    bombs   = _safe_bombs(obs)

    aid = int(agent_id)
    n_pl = players.shape[0]
    H, W = grid.shape

    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[0] = True  # STOP always legal

    if n_pl <= aid or int(players[aid, 2]) == 0:
        return mask  # dead agent — only STOP

    my_row = int(players[aid, 0])
    my_col = int(players[aid, 1])
    my_bombs_left = int(players[aid, 3])

    bomb_pos: set[tuple[int, int]] = set()
    for b in bombs:
        bomb_pos.add((int(b[0]), int(b[1])))

    # Movement actions 1–4
    for a, (dx, dy) in ACTION_DELTAS.items():
        if a == 0:
            continue
        nx, ny = my_row + dx, my_col + dy
        if not (0 < nx < H - 1 and 0 < ny < W - 1):
            continue
        cell = int(grid[nx, ny])
        if cell in (WALL, BOX):
            continue
        if (nx, ny) in bomb_pos:
            continue
        mask[a] = True

    # PLACE_BOMB (action 5)
    if my_bombs_left > 0 and (my_row, my_col) not in bomb_pos:
        mask[5] = True

    return mask


def apply_safety(
    logits: np.ndarray,
    obs: dict,
    agent_id: int,
    tracker=None,
) -> int:
    """
    Apply rule-based safety filter over raw policy logits.

    Algorithm:
    1. Compute legal mask (physical legality).
    2. If current tile is in danger_t1, force escape (ignore STOP/bomb).
    3. Block PLACE_BOMB if no escape path exists after placement.
    4. Mask out illegal actions from logits.
    5. Among safe actions, pick argmax of logits.
    6. If all actions masked, fall back to best legal action (argmax over legal).
    7. Always return int in [0, 5].
    """
    try:
        return _apply_safety_impl(logits, obs, agent_id, tracker)
    except Exception:
        # Last resort: return STOP
        return 0


def get_safe_mask(obs: dict, agent_id: int) -> np.ndarray:
    grid    = _safe_grid(obs)
    players = _safe_players(obs)
    bombs   = _safe_bombs(obs)

    aid  = int(agent_id)
    n_pl = players.shape[0]
    mask = np.zeros(N_ACTIONS, dtype=bool)

    if n_pl <= aid or int(players[aid, 2]) == 0:
        mask[0] = True
        return mask

    my_row        = int(players[aid, 0])
    my_col        = int(players[aid, 1])
    my_bombs_left = int(players[aid, 3])
    my_radius     = 1 + int(players[aid, 4])

    bomb_pos: set[tuple[int, int]] = set()
    for b in bombs:
        bomb_pos.add((int(b[0]), int(b[1])))

    danger = compute_danger_map(obs)
    danger_t1 = danger[1]
    danger_all = danger[1] | danger[2] | danger[3]

    legal = legal_action_mask(obs, aid)
    safe  = legal.copy()

    if safe[5]:
        prosp_blast = blast_tiles(grid, my_row, my_col, my_radius)
        combined_danger = danger_all | prosp_blast
        if not has_escape_path(grid, (my_row, my_col), bomb_pos, combined_danger):
            safe[5] = False

    if (my_row, my_col) in danger_t1:
        safe[0] = False
        safe[5] = False
        for a in range(1, 5):
            if not legal[a]:
                safe[a] = False
                continue
            dx, dy = ACTION_DELTAS[a]
            nx, ny = my_row + dx, my_col + dy
            if (nx, ny) in danger_t1:
                safe[a] = False

    if not np.any(safe):
        if np.any(legal):
            return legal
        mask[0] = True
        return mask

    return safe


def _apply_safety_impl(
    logits: np.ndarray,
    obs: dict,
    agent_id: int,
    tracker=None,
) -> int:
    safe = get_safe_mask(obs, agent_id)

    arr = np.asarray(logits, dtype=np.float32)
    if arr.shape[0] != N_ACTIONS:
        arr = np.zeros(N_ACTIONS, dtype=np.float32)

    # Choose best among safe actions
    safe_indices = np.where(safe)[0]
    if safe_indices.size > 0:
        best = safe_indices[np.argmax(arr[safe_indices])]
        return int(best)

    return 0

    return 0  # absolute last resort
