"""
league_manager.py — Opponent pool manager for self-play / league training.

Sampling strategy (configurable fractions):
  30% fixed baselines (rule-based agents)
  50% checkpoint pool (saved historical actor snapshots)
  20% random (any of the above)
"""

from __future__ import annotations
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── baseline registry ────────────────────────────────────────────────────────

def _load_baseline(agent_path: str, agent_id: int) -> Any:
    """Load a baseline rule-based agent by file path."""
    import importlib.util
    agent_dir = str(Path(agent_path).parent)
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)
    spec = importlib.util.spec_from_file_location("_baseline_mod", agent_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Find agent class
    for name in dir(mod):
        cls = getattr(mod, name)
        if isinstance(cls, type) and (name == "Agent" or name.endswith("Agent")):
            try:
                return cls(agent_id)
            except TypeError:
                return cls()
    raise RuntimeError(f"No Agent class found in {agent_path}")


BASELINE_PATHS = [
    str(ROOT / "agent" / "tactical_rule_agent.py"),
    str(ROOT / "agent" / "smarter_rule_agent.py"),
    str(ROOT / "agent" / "genius_rule_agent.py"),
    str(ROOT / "agent" / "random_agent.py"),
]


# ── checkpoint-based opponent ─────────────────────────────────────────────────

class _CheckpointAgent:
    """Wraps a saved actor checkpoint as an opponent agent."""

    def __init__(self, ckpt_path: str, agent_id: int):
        import torch
        from agent.mappo_agent.model import CNNActor
        from agent.mappo_agent.encoder import encode_obs
        from agent.mappo_agent.checkpoint_utils import load_actor_state_dict
        from agent.mappo_agent.tracker import AgentTracker
        from agent.mappo_agent.safety import apply_safety

        self.agent_id = int(agent_id)
        self._encode  = encode_obs
        self._safety  = apply_safety
        self._tracker = AgentTracker(agent_id)
        self._last_action: int | None = None

        actor = CNNActor()
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        load_actor_state_dict(actor, ckpt, map_location="cpu")
        actor.eval()
        self._actor  = actor
        self._device = torch.device("cpu")

    def reset(self) -> None:
        self._tracker.reset()
        self._last_action = None

    def act(self, obs: dict) -> int:
        import torch
        try:
            self._tracker.sync_before_act(obs, self._last_action)
            sp, sc = self._encode(obs, self.agent_id, self._tracker)
            with torch.inference_mode():
                logits = self._actor(
                    torch.from_numpy(sp).unsqueeze(0),
                    torch.from_numpy(sc).unsqueeze(0),
                ).squeeze(0).numpy()
            action = self._safety(logits, obs, self.agent_id, tracker=self._tracker)
            self._last_action = action
            return action
        except Exception:
            return 0


# ── league manager ────────────────────────────────────────────────────────────

class LeagueManager:
    """
    Maintains a pool of baseline and checkpoint opponents.
    Provides `sample_opponents(n, agent_id)` to fill the other 3 seats.
    """

    MAX_CKPTS = 20   # max checkpoints kept in pool

    def __init__(
        self,
        baseline_frac: float = 0.30,
        ckpt_frac:     float = 0.50,
        seed:          int   = 42,
    ):
        self.baseline_frac = baseline_frac
        self.ckpt_frac     = ckpt_frac
        self._ckpt_paths:  list[str] = []
        random.seed(seed)

        # Validate baseline paths exist
        self._valid_baselines = [p for p in BASELINE_PATHS if os.path.exists(p)]

    def add_checkpoint(self, path: str) -> None:
        """Add a new checkpoint to the pool (evict oldest if over limit)."""
        p = str(path)
        if p not in self._ckpt_paths:
            self._ckpt_paths.append(p)
        if len(self._ckpt_paths) > self.MAX_CKPTS:
            self._ckpt_paths.pop(0)

    def save_checkpoint(
        self,
        actor,
        critic,
        path:   str,
        update: int,
        extra:  dict | None = None,
    ) -> None:
        """Save a checkpoint and add it to the pool."""
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "actor_state_dict":  actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "update": update,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        self.add_checkpoint(path)

    def sample_opponents(self, n: int = 3, agent_id: int = 0) -> list[Any]:
        """
        Sample n opponent agents for seats OTHER than agent_id.
        Returns a list of agent instances with .act(obs) method.
        """
        opponents = []
        for seat in range(n):
            opp_id = (agent_id + seat + 1) % 4
            opp = self._sample_one(opp_id)
            opponents.append(opp)
        return opponents

    def _sample_one(self, agent_id: int) -> Any:
        roll = random.random()
        if roll < self.baseline_frac and self._valid_baselines:
            return self._load_baseline_safe(random.choice(self._valid_baselines), agent_id)
        elif roll < self.baseline_frac + self.ckpt_frac and self._ckpt_paths:
            return self._load_ckpt_safe(random.choice(self._ckpt_paths), agent_id)
        else:
            # Random: pick from whichever pool is available
            pool = self._valid_baselines + self._ckpt_paths
            if not pool:
                from agent.mappo_agent.agent import Agent
                return Agent(agent_id)
            choice = random.choice(pool)
            if choice in self._ckpt_paths:
                return self._load_ckpt_safe(choice, agent_id)
            return self._load_baseline_safe(choice, agent_id)

    @staticmethod
    def _load_baseline_safe(path: str, agent_id: int) -> Any:
        try:
            return _load_baseline(path, agent_id)
        except Exception:
            from agent.mappo_agent.agent import Agent
            return Agent(agent_id)

    @staticmethod
    def _load_ckpt_safe(path: str, agent_id: int) -> Any:
        try:
            return _CheckpointAgent(path, agent_id)
        except Exception:
            from agent.mappo_agent.agent import Agent
            return Agent(agent_id)
