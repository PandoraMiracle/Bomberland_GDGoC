"""
evaluate_agent.py — Evaluate a trained MAPPO actor checkpoint.

Runs matches against a fixed baseline suite, computes win/draw/loss rates,
average rank, TrueSkill estimate, and saves logs.

Match JSON mirrors the official server format:
  seed, team_ids, ranks, survival_steps, runtime_stats, history

Usage:
  python -m training.mappo.evaluate_agent [options]

Options:
  --checkpoint  PATH    actor checkpoint to evaluate (required)
  --matches     N       number of matches per seed suite (default: 50)
  --seed-suite  ID      fixed seed suite 0 | 1 | 2 (default: 0)
  --baselines   LIST    comma-separated names: tactical,genius,smarter,random
  --save-match-logs     save match JSONs to logs/mappo/matches/json/
  --save-gifs           also save GIF replays (slow)
  --log-dir     DIR     (default: logs/mappo)
  --update      N       training update number (for logging)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from engine.game import BomberEnv
from agent.mappo_agent.model import CNNActor
from agent.mappo_agent.encoder import encode_obs
from agent.mappo_agent.tracker import AgentTracker
from agent.mappo_agent.safety import apply_safety
from training.mappo.logger import TrainingLogger

# Fixed seed suites for reproducible comparisons
SEED_SUITES: dict[int, list[int]] = {
    0: list(range(1000, 1050)),
    1: list(range(2000, 2050)),
    2: list(range(3000, 3050)),
}

BASELINE_PATHS = {
    "tactical": str(ROOT / "agent" / "tactical_rule_agent.py"),
    "genius":   str(ROOT / "agent" / "genius_rule_agent.py"),
    "smarter":  str(ROOT / "agent" / "smarter_rule_agent.py"),
    "random":   str(ROOT / "agent" / "random_agent.py"),
}


def _resolve_device(device_str: str) -> str:
    """Use CPU when CUDA is requested but unavailable (common after Colab disconnect)."""
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("[Eval] CUDA unavailable — falling back to CPU")
        return "cpu"
    return device_str


def _load_baseline(name: str, agent_id: int):
    import importlib.util
    path = BASELINE_PATHS[name]
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(f"_eval_baseline_{name}_{agent_id}", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for attr in dir(mod):
        cls = getattr(mod, attr)
        if isinstance(cls, type) and (attr == "Agent" or attr.endswith("Agent")):
            try:    return cls(agent_id)
            except: return cls()
    raise RuntimeError(f"No Agent class in {path}")


class MAPPOEvalAgent:
    def __init__(self, ckpt_path: str, agent_id: int, device_str: str = "cpu"):
        self.agent_id = agent_id
        device_str = _resolve_device(device_str)
        dev = torch.device(device_str)
        actor = CNNActor()
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("actor_state_dict", ckpt.get("model_state_dict", ckpt))
        actor.load_state_dict(state, strict=True)
        actor.eval()
        self._actor  = actor.to(dev)
        self._device = dev
        self._tracker = AgentTracker(agent_id)
        self._last_action: int | None = None

    def reset(self) -> None:
        self._tracker.reset()
        self._last_action = None

    def act(self, obs: dict) -> int:
        try:
            self._tracker.sync_before_act(obs, self._last_action)
            sp, sc = encode_obs(obs, self.agent_id, self._tracker)
            with torch.inference_mode():
                logits = self._actor(
                    torch.from_numpy(sp).unsqueeze(0).to(self._device),
                    torch.from_numpy(sc).unsqueeze(0).to(self._device),
                ).squeeze(0).cpu().numpy()
            action = apply_safety(logits, obs, self.agent_id, tracker=self._tracker)
            self._last_action = action
            return action
        except Exception:
            return 0


def run_match(
    agents:    list,
    seed:      int,
    max_steps: int = 500,
    save_history: bool = False,
) -> dict[str, Any]:
    """Run one match, return result dict mirroring the server JSON format."""
    env = BomberEnv(seed=seed, max_steps=max_steps)
    obs = env.reset(seed=seed)

    for agent in agents:
        if hasattr(agent, "reset"):
            agent.reset()

    history = []
    death_order: list[list[int]] = []
    alive_mask = [True] * 4
    survival_steps = [0] * 4
    runtime_stats  = {str(i): {"timeouts":0,"errors":0,"invalid_actions":0,"fallback_uses":0}
                      for i in range(4)}

    if save_history:
        history.append({
            "step": 0, "actions": None,
            "alive": [True]*4,
            "map": obs["map"].tolist(),
            "players": obs["players"].tolist(),
            "bombs": obs["bombs"].tolist(),
        })

    terminated = truncated = False
    while not (terminated or truncated):
        actions = []
        t0 = time.perf_counter()
        for i, agent in enumerate(agents):
            if env.players[i].alive:
                try:
                    a = agent.act(obs)
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    if elapsed > 100.0:
                        runtime_stats[str(i)]["timeouts"] += 1
                        a = 0
                    if not isinstance(a, int) or not (0 <= a <= 5):
                        runtime_stats[str(i)]["invalid_actions"] += 1
                        a = 0
                except Exception:
                    runtime_stats[str(i)]["errors"] += 1
                    a = 0
                actions.append(a)
            else:
                actions.append(0)

        obs, terminated, truncated = env.step(actions)

        if save_history:
            history.append({
                "step":    env.current_step,
                "actions": actions,
                "alive":   [bool(p.alive) for p in env.players],
                "map":     obs["map"].tolist(),
                "players": obs["players"].tolist(),
                "bombs":   obs["bombs"].tolist(),
            })

        deaths = []
        for i, p in enumerate(env.players):
            if alive_mask[i] and not p.alive:
                alive_mask[i] = False
                survival_steps[i] = env.current_step
                deaths.append(i)
        if deaths:
            death_order.append(deaths)

    # Compute final ranks
    alives = [i for i, p in enumerate(env.players) if p.alive]
    if alives:
        def _stats(i):
            return (env.players[i].stats["kills"], env.players[i].stats["boxes"],
                    env.players[i].stats["items"], env.players[i].stats["bombs"])
        alives.sort(key=_stats, reverse=True)
        groups = []
        cur_g = [alives[0]]; cur_s = _stats(alives[0])
        for i in alives[1:]:
            if _stats(i) == cur_s: cur_g.append(i)
            else: groups.append(cur_g); cur_g = [i]; cur_s = _stats(i)
        groups.append(cur_g)
        death_order.extend(reversed(groups))
        for i in alives:
            survival_steps[i] = env.current_step

    ranks = [0] * 4
    for rank, group in enumerate(reversed(death_order)):
        for i in group:
            ranks[i] = rank

    return {
        "seed": seed, "ranks": ranks,
        "survival_steps": survival_steps,
        "runtime_stats":  runtime_stats,
        "history":        history,
        "player_stats":   [env.players[i].stats for i in range(4)],
    }


def evaluate(
    ckpt_path:     str,
    n_matches:     int   = 50,
    seed_suite_id: int   = 0,
    baseline_names: list = None,
    save_match_logs: bool = False,
    save_gifs:      bool  = False,
    log_dir:        str   = "logs/mappo",
    update:         int   = 0,
    device_str:     str   = "cpu",
) -> dict[str, Any]:
    device_str = _resolve_device(device_str)
    seeds = SEED_SUITES.get(seed_suite_id, SEED_SUITES[0])[:n_matches]
    if baseline_names is None:
        baseline_names = ["tactical", "genius", "smarter"]

    logger = TrainingLogger(log_dir, use_tensorboard=False)
    match_log_dir = Path(log_dir) / "matches" / "json"
    if save_match_logs:
        match_log_dir.mkdir(parents=True, exist_ok=True)

    AGENT_SEAT = 0
    mappo_agent = MAPPOEvalAgent(ckpt_path, AGENT_SEAT, device_str)
    baselines = []
    for k, name in enumerate(baseline_names[:3]):
        try:
            baselines.append(_load_baseline(name, (AGENT_SEAT + k + 1) % 4))
        except Exception:
            from agent.mappo_agent.agent import Agent
            baselines.append(Agent((AGENT_SEAT + k + 1) % 4))

    agents = [None] * 4
    agents[AGENT_SEAT] = mappo_agent
    for k, b in enumerate(baselines):
        agents[(AGENT_SEAT + k + 1) % 4] = b

    # ── run matches ───────────────────────────────────────────────────────────
    ranks_list:       list[int]   = []
    survival_list:    list[int]   = []
    kills_list:       list[int]   = []
    boxes_list:       list[int]   = []
    items_list:       list[int]   = []
    bombs_list:       list[int]   = []
    timeouts   = errors = invalid_actions = fallback_uses = 0
    wins = draws = losses = 0

    print(f"[Eval] checkpoint={Path(ckpt_path).name} matches={len(seeds)} suite={seed_suite_id}")

    for seed in seeds:
        result = run_match(agents, seed, save_history=save_match_logs)

        my_rank = result["ranks"][AGENT_SEAT]
        min_rank = min(result["ranks"])
        ranks_list.append(my_rank)
        survival_list.append(result["survival_steps"][AGENT_SEAT])
        stats = result["player_stats"][AGENT_SEAT]
        kills_list.append(stats.get("kills", 0))
        boxes_list.append(stats.get("boxes", 0))
        items_list.append(stats.get("items", 0))
        bombs_list.append(stats.get("bombs", 0))

        rs = result["runtime_stats"][str(AGENT_SEAT)]
        timeouts      += rs["timeouts"]
        errors        += rs["errors"]
        invalid_actions += rs["invalid_actions"]
        fallback_uses += rs["fallback_uses"]

        if my_rank == min_rank:
            winners = [i for i,r in enumerate(result["ranks"]) if r==min_rank]
            if len(winners) == 1: wins += 1
            else:                 draws += 1
        else:
            losses += 1

        if save_match_logs:
            team_ids = ["MAPPOAgent"] + baseline_names[:3]
            match_data = {
                "seed": seed,
                "team_ids": team_ids,
                "ranks": result["ranks"],
                "survival_steps": result["survival_steps"],
                "runtime_stats": result["runtime_stats"],
                "history": result["history"],
            }
            jpath = match_log_dir / f"match_{seed}.json"
            with open(jpath, "w") as f:
                json.dump(match_data, f)

    n = len(seeds)
    avg_rank     = float(np.mean(ranks_list))
    avg_survival = float(np.mean(survival_list))

    # TrueSkill estimate (simplified local version)
    try:
        import trueskill
        env_ts = trueskill.TrueSkill(mu=100.0, sigma=100/3, draw_probability=0.1)
        rating = env_ts.Rating()
        for rank in ranks_list:
            rating, = env_ts.rate_1vs1(rating, env_ts.Rating(), drawn=(rank==0))[0:1]
        mu, sigma = rating.mu, rating.sigma
    except Exception:
        mu, sigma = 25.0 + wins - losses, 25.0 / 3.0

    score = mu - 3 * sigma
    win_rate  = wins  / max(n, 1)
    draw_rate = draws / max(n, 1)
    loss_rate = losses / max(n, 1)

    metrics = {
        "checkpoint_path":       ckpt_path,
        "num_matches":           n,
        "fixed_seed_suite_id":   seed_suite_id,
        "win_rate":              round(win_rate,  4),
        "draw_rate":             round(draw_rate, 4),
        "loss_rate":             round(loss_rate, 4),
        "average_rank":          round(avg_rank, 4),
        "average_survival_steps": round(avg_survival, 2),
        "average_kills":         round(float(np.mean(kills_list)), 3),
        "average_boxes":         round(float(np.mean(boxes_list)), 3),
        "average_items":         round(float(np.mean(items_list)), 3),
        "average_bombs":         round(float(np.mean(bombs_list)), 3),
        "timeout_count":         timeouts,
        "error_count":           errors,
        "invalid_action_count":  invalid_actions,
        "fallback_uses":         fallback_uses,
        "estimated_mu":          round(mu,    3),
        "estimated_sigma":       round(sigma, 3),
        "estimated_score":       round(score, 3),
        "update":                update,
    }

    logger.log_eval(metrics)
    logger.close()

    print(
        f"[Eval] win={win_rate:.1%} draw={draw_rate:.1%} loss={loss_rate:.1%} "
        f"avg_rank={avg_rank:.2f} score={score:.1f} "
        f"(mu={mu:.1f} sigma={sigma:.1f})"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",      type=str, required=True)
    parser.add_argument("--matches",         type=int, default=50)
    parser.add_argument("--seed-suite",      type=int, default=0)
    parser.add_argument("--baselines",       type=str, default="tactical,genius,smarter")
    parser.add_argument("--save-match-logs", action="store_true")
    parser.add_argument("--save-gifs",       action="store_true")
    parser.add_argument("--log-dir",         type=str, default="logs/mappo")
    parser.add_argument("--update",          type=int, default=0)
    parser.add_argument("--device",          type=str, default="auto")
    args = parser.parse_args()

    evaluate(
        ckpt_path     = args.checkpoint,
        n_matches     = args.matches,
        seed_suite_id = args.seed_suite,
        baseline_names = args.baselines.split(","),
        save_match_logs = args.save_match_logs,
        save_gifs     = args.save_gifs,
        log_dir       = args.log_dir,
        update        = args.update,
        device_str    = args.device,
    )


if __name__ == "__main__":
    main()
