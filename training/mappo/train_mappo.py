"""
train_mappo.py — Main MAPPO training entry point.

Usage:
  python -m training.mappo.train_mappo [options]

Options:
  --config CONFIG       Path to MAPPOConfig JSON
  --resume CKPT         Resume from checkpoint path
  --phase PHASE         baseline | selfplay (default: baseline)
  --num-envs N          Override num_envs
  --total-steps N       Override total_env_steps
  --device DEVICE       cpu | cuda | auto
  --agent-id ID         Which seat to train (default: 0; all 4 share weights)
  --no-safety           Disable safety filter during collection
  --save-match-logs     Save match JSON logs to logs/mappo/matches/json/
  --save-gifs           Save GIF replays (slow, off by default)
  --seed N              Override random seed
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
import shutil
import dataclasses

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.optim as optim

from engine.game import BomberEnv
from agent.mappo_agent.model import CNNActor, CentralizedCritic
from agent.mappo_agent.encoder import encode_obs
from agent.mappo_agent.tracker import AgentTracker
from agent.mappo_agent.safety import apply_safety, legal_action_mask, get_safe_mask

from training.mappo.config import MAPPOConfig
from training.mappo.rollout_buffer import RolloutBuffer
from training.mappo.ppo_update import ppo_update
from training.mappo.league_manager import LeagueManager
from training.mappo.logger import TrainingLogger
from training.mappo.vec_env import VecEnv
from training.mappo.reward_builder import compute_annealed_dense_coef


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_global_scalar(obs: dict, n_players: int = 4) -> np.ndarray:
    """
    Build a fixed-dim (32,) global scalar for the centralized critic.
    Encodes all 4 players' stats + step + bomb count.
    """
    vec = np.zeros(32, dtype=np.float32)
    players = np.asarray(obs.get("players", np.zeros((4,5))), dtype=np.int32)
    bombs   = obs.get("bombs")
    n_bombs = 0 if (bombs is None or np.asarray(bombs).size == 0) else np.asarray(bombs).shape[0]

    for i in range(min(4, players.shape[0])):
        base = i * 5
        vec[base]     = float(players[i, 2])           # alive
        vec[base + 1] = float(players[i, 3]) / 5.0    # bombs_left
        vec[base + 2] = float(1 + players[i, 4]) / 5.0  # radius
        vec[base + 3] = float(players[i, 0]) / 12.0   # row
        vec[base + 4] = float(players[i, 1]) / 12.0   # col
    # slot 20: n_bombs / 10
    vec[20] = float(n_bombs) / 10.0
    # slots 21-31: reserved / zeros
    return vec


def load_checkpoint(
    actor:   CNNActor,
    critic:  CentralizedCritic,
    path:    str,
    device:  str = "cpu",
) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt.get("actor_state_dict", ckpt), strict=True)
    critic.load_state_dict(ckpt.get("critic_state_dict", {}), strict=False)
    return ckpt


def _resolve_dense_coef(cfg: MAPPOConfig, env_steps: int) -> tuple[float, str]:
    """Current dense reward coefficient and annealing phase label."""
    return compute_annealed_dense_coef(
        env_steps,
        cfg.total_env_steps,
        enabled=cfg.dense_reward_anneal,
        fixed_coef=cfg.dense_reward_coef,
        coef_early=cfg.dense_reward_coef_early,
        coef_mid=cfg.dense_reward_coef_mid,
        coef_late=cfg.dense_reward_coef_late,
        mid_start=cfg.dense_reward_anneal_mid_start,
        late_start=cfg.dense_reward_anneal_late_start,
    )


# ── training loop ─────────────────────────────────────────────────────────────

def train(cfg: MAPPOConfig, resume: str | None = None, use_safety: bool = True, reset_steps: bool = False) -> None:
    device_str = cfg.resolve_device()
    device     = torch.device(device_str)

    set_seed(cfg.seed)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ── model + optimisers ────────────────────────────────────────────────────
    actor  = CNNActor(cfg.n_spatial, cfg.n_scalar, cfg.n_actions).to(device)
    critic = CentralizedCritic(cfg.n_spatial, cfg.global_scalar_dim).to(device)
    opt_a  = optim.Adam(actor.parameters(),  lr=cfg.actor_lr)
    opt_c  = optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    start_update = 0
    env_steps    = 0

    if resume:
        ckpt = load_checkpoint(actor, critic, resume, device_str)
        if not reset_steps:
            start_update = int(ckpt.get("update", 0))
            env_steps    = int(ckpt.get("env_steps", 0))
        else:
            start_update = 0
            env_steps    = 0
        print(f"[Resume] update={start_update}, env_steps={env_steps} (reset_steps={reset_steps})")

    # ── league + logger ───────────────────────────────────────────────────────
    league  = LeagueManager(cfg.league_baseline_frac, cfg.league_ckpt_frac, cfg.seed)
    logger  = TrainingLogger(cfg.log_dir)

    if resume:
        league.add_checkpoint(resume)

    # ── rollout buffer ────────────────────────────────────────────────────────
    buffer = RolloutBuffer(
        cfg.rollout_length, cfg.num_envs,
        cfg.n_spatial, cfg.n_scalar, cfg.global_scalar_dim,
    )

    # ── environment pool ──────────────────────────────────────────────────────
    AGENT_ID = 0   # training seat; weights shared across all 4 seats
    init_coef, init_phase = _resolve_dense_coef(cfg, env_steps)
    vec_env = VecEnv(cfg.num_envs, AGENT_ID, cfg.seed, cfg.max_steps, init_coef)
    
    obs_list:  list[dict]  = []
    episode_returns: list[list[float]] = [[] for _ in range(cfg.num_envs)]
    episode_lens:    list[int]         = [0] * cfg.num_envs

    trackers: list[AgentTracker] = [AgentTracker(AGENT_ID) for _ in range(cfg.num_envs)]

    for e_i in range(cfg.num_envs):
        seed_i = cfg.seed + e_i
        obs = vec_env.reset_idx(e_i, seed_i, league._ckpt_paths)
        obs_list.append(obs)
        trackers[e_i].reset()

    # ── training loop ─────────────────────────────────────────────────────────
    update    = start_update
    total     = cfg.total_env_steps
    completed = 0
    actor.train(); critic.train()

    print(
        f"[Train] phase={cfg.phase} device={device_str} envs={cfg.num_envs} "
        f"total_steps={total:,} dense_anneal={cfg.dense_reward_anneal} "
        f"dense_coef={init_coef} ({init_phase})"
    )

    t_start = time.perf_counter()
    best_metric_value = float('-inf') if cfg.best_metric_mode == "max" else float('inf')
    numbered_checkpoints: list[Path] = []

    while env_steps < total:
        dense_coef, anneal_phase = _resolve_dense_coef(cfg, env_steps)
        vec_env.set_dense_reward_coef(dense_coef)
        update_episode_returns = []

        # ── rollout collection ─────────────────────────────────────────────────
        for t in range(cfg.rollout_length):
            # ── 1. Batch observations ──
            sp_list, sc_list, gsc_list, safe_list = [], [], [], []
            for e_i in range(cfg.num_envs):
                obs   = obs_list[e_i]
                sp, sc = encode_obs(obs, AGENT_ID, trackers[e_i])
                gsc    = make_global_scalar(obs)

                sp_list.append(sp)
                sc_list.append(sc)
                gsc_list.append(gsc)

                if use_safety:
                    safe_np = get_safe_mask(obs, AGENT_ID)
                else:
                    safe_np = np.ones(6, dtype=bool)
                safe_list.append(safe_np)

            # ── 2. Batched Inference ──
            sp_t  = torch.from_numpy(np.stack(sp_list)).to(device)
            sc_t  = torch.from_numpy(np.stack(sc_list)).to(device)
            gsc_t = torch.from_numpy(np.stack(gsc_list)).to(device)
            safe_t = torch.from_numpy(np.stack(safe_list)).to(device)

            with torch.no_grad():
                logits = actor(sp_t, sc_t)
                val    = critic(sp_t, gsc_t)
                
                if use_safety:
                    logits = logits.masked_fill(~safe_t, -1e9)
                    
                dist   = torch.distributions.Categorical(logits=logits)
                act_t  = dist.sample()
                lp_t   = dist.log_prob(act_t)

            # ── 3. Step Environments ──
            actions_list = [int(act_t[e_i].item()) for e_i in range(cfg.num_envs)]
            next_obs_list, reward_list, done_list = vec_env.step(actions_list)

            for e_i in range(cfg.num_envs):
                action = actions_list[e_i]
                sp_i = sp_list[e_i]
                sc_i = sc_list[e_i]
                gsc_i = gsc_list[e_i]
                safe_i = safe_list[e_i]

                next_obs = next_obs_list[e_i]
                reward = reward_list[e_i]
                done = bool(done_list[e_i])

                buffer.add(
                    t, e_i, sp_i, sc_i, gsc_i, action,
                    reward, float(done),
                    val[e_i].item(), lp_t[e_i].item(), safe_i
                )

                episode_returns[e_i].append(reward)
                episode_lens[e_i] += 1
                env_steps += 1

                trackers[e_i].update(next_obs, action)

                if done:
                    # Reset env with new opponents
                    seed_i = cfg.seed + e_i + update * cfg.num_envs
                    new_obs = vec_env.reset_idx(e_i, seed_i, league._ckpt_paths)
                    obs_list[e_i]       = new_obs
                    trackers[e_i].reset()
                    completed += 1
                    update_episode_returns.append(sum(episode_returns[e_i]))
                    episode_returns[e_i] = []
                    episode_lens[e_i]    = 0
                else:
                    obs_list[e_i] = next_obs

        # ── GAE + PPO update ──────────────────────────────────────────────────
        # Bootstrap last values
        sp_last_list, gsc_last_list = [], []
        for e_i in range(cfg.num_envs):
            sp, _ = encode_obs(obs_list[e_i], AGENT_ID, trackers[e_i])
            gsc = make_global_scalar(obs_list[e_i])
            sp_last_list.append(sp)
            gsc_last_list.append(gsc)

        sp_last_t = torch.from_numpy(np.stack(sp_last_list)).to(device)
        gsc_last_t = torch.from_numpy(np.stack(gsc_last_list)).to(device)

        with torch.no_grad():
            last_vals_t = critic(sp_last_t, gsc_last_t).squeeze(-1)
        last_vals = last_vals_t.cpu().numpy()

        buffer.compute_gae(last_vals, cfg.gamma, cfg.gae_lambda)
        flat = buffer.get_flat()

        metrics = ppo_update(
            actor, critic, opt_a, opt_c, flat,
            clip_eps    = cfg.clip_eps,
            ppo_epochs  = cfg.ppo_epochs,
            mini_batch  = cfg.mini_batch_size,
            entropy_coef = cfg.entropy_coef,
            value_coef  = cfg.value_coef,
            max_grad    = cfg.max_grad_norm,
            device      = device_str,
        )

        update += 1
        elapsed = time.perf_counter() - t_start
        fps     = env_steps / max(elapsed, 1)

        # All episode returns collected so far
        all_returns = [r for ep in episode_returns for r in ep]
        mean_return = float(np.mean(all_returns)) if all_returns else 0.0

        raw_rewards = buffer.rewards
        ep_ret_mean = float(np.mean(update_episode_returns)) if update_episode_returns else 0.0
        ep_ret_min = float(np.min(update_episode_returns)) if update_episode_returns else 0.0
        ep_ret_max = float(np.max(update_episode_returns)) if update_episode_returns else 0.0

        tracker_snapshots = [t.stats_dict() for t in trackers]
        log_row = {
            **metrics,
            "update":              update,
            "env_steps":           env_steps,
            "episodes_completed":  completed,
            "completed_episodes_this_update": len(update_episode_returns),
            "fps":                 round(fps, 1),
            "mean_return":         round(mean_return, 6),
            "raw_reward_mean":     round(float(raw_rewards.mean()), 6),
            "raw_reward_sum":      round(float(raw_rewards.sum()), 6),
            "raw_reward_min":      round(float(raw_rewards.min()), 6),
            "raw_reward_max":      round(float(raw_rewards.max()), 6),
            "episode_return_mean": round(ep_ret_mean, 6),
            "episode_return_min":  round(ep_ret_min, 6),
            "episode_return_max":  round(ep_ret_max, 6),
            "mean_episode_length": cfg.rollout_length,
            "dense_reward_coef":   round(dense_coef, 4),
            "dense_reward_anneal_phase": anneal_phase,
            "actor_lr":            cfg.actor_lr,
            "critic_lr":           cfg.critic_lr,
            "checkpoint_path":     "",
            "tracker_mean_step":   round(float(np.mean([s["estimated_step"] for s in tracker_snapshots])), 2),
            "tracker_mean_boxes":  round(float(np.mean([s["boxes_destroyed"] for s in tracker_snapshots])), 3),
            "tracker_mean_items":  round(float(np.mean([s["items_collected"] for s in tracker_snapshots])), 3),
            "tracker_mean_bombs":  round(float(np.mean([s["bombs_placed"] for s in tracker_snapshots])), 3),
            "tracker_mean_kills":  round(float(np.mean([s["kills"] for s in tracker_snapshots])), 3),
        }

        # ── checkpoint ────────────────────────────────────────────────────────
        ckpt_path = ""
        if update % cfg.checkpoint_every == 0:
            payload = {
                "update": update,
                "env_steps": env_steps,
                "config": dataclasses.asdict(cfg),
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "opt_actor_state_dict": opt_a.state_dict(),
                "opt_critic_state_dict": opt_c.state_dict(),
                "best_metric_value": best_metric_value,
                "phase": cfg.phase,
            }
            
            latest_path = Path(cfg.checkpoint_dir) / "latest.pth"
            numbered_path = Path(cfg.checkpoint_dir) / f"checkpoint_update_{update}.pth"
            
            # 1. Always save latest.pth
            torch.save(payload, str(latest_path))
            
            # 2. Save numbered checkpoint and handle retention
            shutil.copy2(latest_path, numbered_path)
            numbered_checkpoints.append(numbered_path)
            
            while len(numbered_checkpoints) > cfg.keep_last_checkpoints:
                oldest = numbered_checkpoints.pop(0)
                if oldest.exists():
                    oldest.unlink()
            
            # Add to league
            league.add_checkpoint(str(numbered_path))
            ckpt_path = str(numbered_path)
            log_row["checkpoint_path"] = ckpt_path
            
            # ── evaluation ────────────────────────────────────────────────────
            if update % cfg.eval_every == 0:
                from training.mappo.evaluate_agent import evaluate
                eval_metrics = evaluate(
                    ckpt_path=str(latest_path),
                    n_matches=cfg.eval_matches,
                    seed_suite_id=0,
                    log_dir=cfg.log_dir,
                    update=update,
                    device_str=device_str,
                )
                payload["eval_metrics"] = eval_metrics
                torch.save(payload, str(latest_path))
                
                current_score = eval_metrics.get(cfg.best_metric)
                if current_score is not None:
                    is_best = False
                    if cfg.best_metric_mode == "max":
                        is_best = current_score > best_metric_value
                    else:
                        is_best = current_score < best_metric_value
                        
                    if is_best:
                        best_metric_value = current_score
                        payload["best_metric_value"] = best_metric_value
                        torch.save(payload, str(latest_path))
                        
                        best_path = Path(cfg.checkpoint_dir) / "best.pth"
                        shutil.copy2(latest_path, best_path)
                        print(f"  [Eval] New best model saved! {cfg.best_metric}={best_metric_value}")

        logger.log_train(log_row)

        if update % 10 == 0:
            print(
                f"[{update:5d}] steps={env_steps:,} | fps={fps:.0f} | "
                f"a_loss={metrics['actor_loss']:.3f} c_loss={metrics['critic_loss']:.3f} "
                f"ent={metrics['entropy']:.3f} kl={metrics['approx_kl']:.4f} | "
                f"ret={mean_return:.6f} raw_r={float(raw_rewards.mean()):.6f} ep_ret={ep_ret_mean:.3f} | "
                f"adv_norm={metrics['advantage_norm_mean']:.6f} | "
                f"trk(step={log_row['tracker_mean_step']:.0f} "
                f"boxes={log_row['tracker_mean_boxes']:.1f} "
                f"items={log_row['tracker_mean_items']:.1f} "
                f"bombs={log_row['tracker_mean_bombs']:.1f} "
                f"kills={log_row['tracker_mean_kills']:.1f})"
                + (f" | ckpt={ckpt_path}" if ckpt_path else "")
            )

    print(f"[Train] Done. Total updates={update}, env_steps={env_steps:,}")
    logger.close()
    vec_env.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MAPPO trainer for Bomberland")
    parser.add_argument("--config",       type=str,   default=None)
    parser.add_argument("--resume",       type=str,   default=None)
    parser.add_argument("--phase",        type=str,   default="baseline")
    parser.add_argument("--num-envs",     type=int,   default=None)
    parser.add_argument("--total-steps",  type=int,   default=None)
    parser.add_argument("--device",       type=str,   default=None)
    parser.add_argument("--agent-id",     type=int,   default=0)
    parser.add_argument("--no-safety",    action="store_true")
    parser.add_argument("--save-match-logs", action="store_true")
    parser.add_argument("--save-gifs",    action="store_true")
    parser.add_argument("--seed",         type=int,   default=None)
    parser.add_argument("--no-dense-anneal", action="store_true",
                        help="Keep dense_reward_coef fixed (no annealing)")
    parser.add_argument("--reset-steps",  action="store_true",
                        help="Reset update and env_steps to 0 when resuming from a checkpoint")
    args = parser.parse_args()

    if args.config:
        cfg = MAPPOConfig.load(args.config)
    else:
        cfg = MAPPOConfig()

    if args.phase:        cfg.phase        = args.phase
    if args.num_envs:     cfg.num_envs     = args.num_envs
    if args.total_steps:  cfg.total_env_steps = args.total_steps
    if args.device:       cfg.device       = args.device
    if args.seed is not None: cfg.seed     = args.seed
    if args.no_dense_anneal:  cfg.dense_reward_anneal = False

    train(cfg, resume=args.resume, use_safety=not args.no_safety, reset_steps=args.reset_steps)


if __name__ == "__main__":
    main()
