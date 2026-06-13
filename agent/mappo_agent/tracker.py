"""
tracker.py — Lightweight per-agent state tracker.

Estimates internal statistics that are not directly observable from a single
obs dict (boxes destroyed, items collected, kills, etc.).  All fields default
to safe zero-values so the encoder never crashes even on the very first step.
"""

from __future__ import annotations
import numpy as np


class AgentTracker:
    """
    Maintained by the Agent across steps.  Call update() after each act()
    and reset() at the start of each episode.
    """

    __slots__ = (
        "agent_id",
        "prev_obs",
        "last_action",
        "estimated_step",
        "bombs_placed",
        "items_collected",
        "boxes_destroyed",
        "kills",
        "idle_streak",
    )

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.reset()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self.prev_obs:       dict | None = None
        self.last_action:    int         = 0
        self.estimated_step: int         = 0
        self.bombs_placed:   int         = 0
        self.items_collected: int        = 0
        self.boxes_destroyed: int        = 0
        self.kills:          int         = 0
        self.idle_streak:    int         = 0

    def update(self, obs: dict, action: int) -> None:
        """
        Call AFTER act() returns and AFTER env.step() is called — i.e. with
        the *next* observation so we can diff stats.
        """
        self.estimated_step += 1
        self.last_action = int(action)

        try:
            self._update_from_obs(obs, action)
        except Exception:
            pass  # never crash tracker; conservative estimates are fine

        self.prev_obs = obs

    # ── internal helpers ─────────────────────────────────────────────────────

    def _safe_players(self, obs: dict) -> np.ndarray | None:
        p = obs.get("players")
        if p is None:
            return None
        arr = np.asarray(p, dtype=np.int32)
        if arr.ndim < 2 or arr.shape[0] <= self.agent_id:
            return None
        return arr

    def _safe_bombs(self, obs: dict) -> np.ndarray:
        b = obs.get("bombs")
        if b is None:
            return np.zeros((0, 4), dtype=np.int32)
        arr = np.asarray(b, dtype=np.int32)
        if arr.ndim == 0 or arr.size == 0:
            return np.zeros((0, 4), dtype=np.int32)
        if arr.ndim == 1:
            return arr.reshape(1, 4) if arr.shape[0] == 4 else np.zeros((0, 4), dtype=np.int32)
        return arr

    def _update_from_obs(self, curr_obs: dict, action: int) -> None:
        aid = self.agent_id

        curr_p = self._safe_players(curr_obs)
        if curr_p is None:
            return

        curr_row = int(curr_p[aid, 0])
        curr_col = int(curr_p[aid, 1])
        curr_alive = int(curr_p[aid, 2])

        # ── idle streak ──────────────────────────────────────────────────────
        if self.prev_obs is not None:
            prev_p = self._safe_players(self.prev_obs)
            if prev_p is not None:
                prev_row = int(prev_p[aid, 0])
                prev_col = int(prev_p[aid, 1])
                if curr_row == prev_row and curr_col == prev_col and action in (0,):
                    self.idle_streak += 1
                else:
                    self.idle_streak = 0
            else:
                self.idle_streak = 0
        else:
            self.idle_streak = 0

        if not curr_alive:
            return

        # ── bombs placed ─────────────────────────────────────────────────────
        if action == 5:  # PLACE_BOMB
            self.bombs_placed += 1

        # ── items collected: detect radius/capacity increase ─────────────────
        if self.prev_obs is not None:
            prev_p = self._safe_players(self.prev_obs)
            if prev_p is not None:
                prev_radius = int(prev_p[aid, 4])
                prev_max_b  = int(prev_p[aid, 3]) + 1  # rough proxy for max_bombs
                curr_radius = int(curr_p[aid, 4])
                curr_bombs_left = int(curr_p[aid, 3])
                if curr_radius > prev_radius:
                    self.items_collected += 1
                # capacity item: bombs_left jumped in a step where no explosion
                # (rough heuristic — may double-count but conservative)
                prev_bl = int(prev_p[aid, 3])
                if curr_bombs_left > prev_bl and action != 5:
                    self.items_collected += 1

        # ── kills: count enemy alive-count drops ─────────────────────────────
        if self.prev_obs is not None:
            prev_p = self._safe_players(self.prev_obs)
            if prev_p is not None:
                n = min(curr_p.shape[0], prev_p.shape[0])
                for oid in range(n):
                    if oid == aid:
                        continue
                    if int(prev_p[oid, 2]) == 1 and int(curr_p[oid, 2]) == 0:
                        # An enemy just died — credit it to us heuristically
                        # (may over-count in multi-bomb scenarios; acceptable)
                        self.kills += 1

        # ── boxes destroyed: count box tiles that became grass ───────────────
        if self.prev_obs is not None:
            prev_grid = self.prev_obs.get("map")
            curr_grid = curr_obs.get("map")
            if prev_grid is not None and curr_grid is not None:
                pg = np.asarray(prev_grid, dtype=np.int32)
                cg = np.asarray(curr_grid, dtype=np.int32)
                if pg.shape == cg.shape:
                    destroyed = int(np.sum((pg == 2) & (cg != 2)))
                    self.boxes_destroyed += destroyed

    # ── inference helpers ────────────────────────────────────────────────────

    def sync_before_act(self, obs: dict, last_action: int | None) -> None:
        """
        For act(obs)-only call sites: apply the pending transition from the
        previous action once the new observation is available.
        """
        if last_action is not None:
            self.update(obs, last_action)

    def stats_dict(self) -> dict[str, int]:
        """Snapshot of tracker counters for logging."""
        return {
            "estimated_step":  self.estimated_step,
            "boxes_destroyed": self.boxes_destroyed,
            "items_collected": self.items_collected,
            "bombs_placed":    self.bombs_placed,
            "kills":           self.kills,
            "idle_streak":     self.idle_streak,
            "last_action":     self.last_action,
        }
