"""
reward_builder.py — Dense + terminal reward shaping for MAPPO training.

All reward values are configurable.  The dense component is scaled by
`dense_reward_coef` which is annealed during training phases.
"""

from __future__ import annotations
import numpy as np

# ── tile constants ────────────────────────────────────────────────────────────
GRASS  = 0
WALL   = 1
BOX    = 2
ITEM_R = 3
ITEM_C = 4

# ── terminal reward values ────────────────────────────────────────────────────
TERMINAL = {
    "rank_0_unique": +1.00,   # sole survivor / unique best rank
    "rank_0_shared": +0.70,   # survived, shared best rank (draw)
    "rank_1":        +0.15,   # second place (4-player)
    "rank_2":        -0.20,   # third place
    "rank_3":        -0.80,   # first eliminated
}

# ── dense reward values ───────────────────────────────────────────────────────
DENSE = {
    "kill":                 +0.25,
    "destroy_box":          +0.02,
    "collect_item":         +0.06,
    "place_useful_bomb":    +0.01,   # near box/enemy + escape exists
    "enter_danger":         -0.08,
    "escape_danger":        +0.05,
    "bomb_no_escape":       -0.10,
    "suicide_penalty":      -0.20,   # extra on top of rank_3
}


def _safe_players(obs: dict, n: int = 4) -> np.ndarray:
    p = obs.get("players")
    if p is None:
        return np.zeros((n, 5), dtype=np.int32)
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


def _blast_tiles(grid: np.ndarray, bx: int, by: int, radius: int) -> set:
    H, W = grid.shape
    tiles = {(bx, by)}
    for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
        for r in range(1, radius+1):
            tx, ty = bx+dx*r, by+dy*r
            if not (0<=tx<H and 0<=ty<W): break
            c = int(grid[tx, ty])
            if c == WALL: break
            tiles.add((tx, ty))
            if c == BOX: break
    return tiles


def _danger_set(obs: dict) -> set:
    """All tiles in any bomb's blast zone (any timer)."""
    grid    = np.asarray(obs.get("map", np.zeros((13,13))), dtype=np.int32)
    players = _safe_players(obs)
    bombs   = _safe_bombs(obs)
    tiles = set()
    for b in bombs:
        bx, by, timer = int(b[0]), int(b[1]), int(b[2])
        oid = int(b[3])
        if timer <= 0: continue
        radius = 1 + (int(players[oid, 4]) if 0 <= oid < len(players) else 0)
        tiles |= _blast_tiles(grid, bx, by, radius)
    return tiles


class RewardBuilder:
    """
    Computes step-level and terminal rewards.

    Usage:
        rb = RewardBuilder(dense_reward_coef=1.0)
        dense  = rb.compute_dense(prev_obs, curr_obs, agent_id, action)
        term   = rb.compute_terminal(ranks, agent_id)
    """

    def __init__(self, dense_reward_coef: float = 1.0):
        self.dense_reward_coef = dense_reward_coef

    # ── terminal ──────────────────────────────────────────────────────────────

    def compute_terminal(self, ranks: list[int], agent_id: int) -> float:
        """
        ranks: list of 4 ints (0 = best rank).
        Returns terminal reward for agent_id.
        """
        aid = int(agent_id)
        if aid >= len(ranks):
            return 0.0

        my_rank = ranks[aid]
        min_rank = min(ranks)

        if my_rank == min_rank:
            # Best rank: check if unique or shared
            winners = [i for i, r in enumerate(ranks) if r == min_rank]
            if len(winners) == 1:
                return TERMINAL["rank_0_unique"]
            else:
                return TERMINAL["rank_0_shared"]

        # Rank among the losers: map to rank_1/2/3 based on sorted rank
        sorted_ranks = sorted(set(ranks))
        position = sorted_ranks.index(my_rank)  # 0-based position from best
        key = f"rank_{min(position, 3)}"
        return TERMINAL.get(key, -0.5)

    # ── dense ─────────────────────────────────────────────────────────────────

    def compute_dense(
        self,
        prev_obs: dict | None,
        curr_obs:  dict,
        agent_id:  int,
        action:    int,
    ) -> float:
        if prev_obs is None:
            return 0.0

        try:
            return self._dense_impl(prev_obs, curr_obs, agent_id, action)
        except Exception:
            return 0.0

    def _dense_impl(
        self,
        prev_obs: dict,
        curr_obs:  dict,
        agent_id:  int,
        action:    int,
    ) -> float:
        aid   = int(agent_id)
        coef  = self.dense_reward_coef
        r     = 0.0

        prev_p = _safe_players(prev_obs)
        curr_p = _safe_players(curr_obs)
        n_pl   = min(prev_p.shape[0], curr_p.shape[0])

        if n_pl <= aid:
            return 0.0

        prev_alive = int(prev_p[aid, 2])
        curr_alive = int(curr_p[aid, 2])

        if prev_alive == 0:
            return 0.0   # already dead

        # Suicide: was alive, now dead
        if prev_alive == 1 and curr_alive == 0:
            r += DENSE["suicide_penalty"]
            return r * coef

        curr_row, curr_col = int(curr_p[aid, 0]), int(curr_p[aid, 1])
        prev_row, prev_col = int(prev_p[aid, 0]), int(prev_p[aid, 1])

        # ── true credit assignment via stats ──────────────────────────────────
        if "stats" in prev_obs and "stats" in curr_obs:
            prev_stats = prev_obs["stats"].get(aid, {})
            curr_stats = curr_obs["stats"].get(aid, {})
            
            kills = curr_stats.get("kills", 0) - prev_stats.get("kills", 0)
            boxes = curr_stats.get("boxes", 0) - prev_stats.get("boxes", 0)
            items = curr_stats.get("items", 0) - prev_stats.get("items", 0)
            
            r += DENSE["kill"] * kills
            r += DENSE["destroy_box"] * boxes
            r += DENSE["collect_item"] * items
        else:
            # Fallback if stats not available
            for oid in range(n_pl):
                if oid == aid:
                    continue
                if int(prev_p[oid, 2]) == 1 and int(curr_p[oid, 2]) == 0:
                    r += DENSE["kill"]

            prev_grid = np.asarray(prev_obs.get("map", np.zeros((13,13))), dtype=np.int32)
            curr_grid = np.asarray(curr_obs.get("map", np.zeros((13,13))), dtype=np.int32)
            if prev_grid.shape == curr_grid.shape:
                boxes_destroyed = int(np.sum((prev_grid == BOX) & (curr_grid != BOX)))
                r += DENSE["destroy_box"] * (boxes_destroyed / 4.0) # Assume shared
                
            prev_radius = int(prev_p[aid, 4])
            curr_radius = int(curr_p[aid, 4])
            if curr_radius > prev_radius:
                r += DENSE["collect_item"]
            prev_bl = int(prev_p[aid, 3])
            curr_bl = int(curr_p[aid, 3])
            if curr_bl > prev_bl and action != 5:
                r += DENSE["collect_item"]


        # ── danger events ─────────────────────────────────────────────────────
        prev_danger = _danger_set(prev_obs)
        curr_danger = _danger_set(curr_obs)
        prev_in  = (prev_row, prev_col) in prev_danger
        curr_in  = (curr_row, curr_col) in curr_danger

        if prev_in and not curr_in:
            r += DENSE["escape_danger"]
        elif not prev_in and curr_in:
            r += DENSE["enter_danger"]

        # ── useful bomb placement ─────────────────────────────────────────────
        if action == 5:
            # Did we place a bomb this step?
            if curr_bl < prev_bl:
                my_radius_now = 1 + int(curr_p[aid, 4])
                blast = _blast_tiles(prev_grid, prev_row, prev_col, my_radius_now)
                hits_box   = any(int(prev_grid[tx, ty]) == BOX for tx,ty in blast)
                enemies_now = [
                    (int(curr_p[oid, 0]), int(curr_p[oid, 1]))
                    for oid in range(n_pl)
                    if oid != aid and int(curr_p[oid, 2]) == 1
                ]
                hits_enemy = any(p in blast for p in enemies_now)
                if hits_box or hits_enemy:
                    r += DENSE["place_useful_bomb"]
                else:
                    r += DENSE["bomb_no_escape"]

        return r * coef
