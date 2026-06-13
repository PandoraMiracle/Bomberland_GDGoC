# TRAINING_README.md — MAPPO Training Guide

## Overview

This guide covers the full pipeline from setup to submission for the Bomberland GDGoC AI Challenge using a MAPPO/PPO-based reinforcement learning agent.

**Architecture:** CNN Actor (ResBlocks) + Centralized Critic + Rule-based Safety Layer  
**Input:** 18 spatial channels (13×13) + 22 scalar features  
**Training:** PPO with GAE, league self-play, behavior cloning warm-start option  
**Submission:** `agent.py` + `model.pth` in flat `.zip`

---

## File Structure

```
agent/mappo_agent/
  __init__.py
  encoder.py        — Observation encoder (18 spatial + 22 scalar)
  tracker.py        — AgentTracker (estimates kills/boxes/items/idle)
  safety.py         — Legal mask, blast tiles, escape path BFS, apply_safety
  model.py          — CNNActor, CentralizedCritic, ActorCritic
  agent.py          — Submission Agent class + MinimalTacticalFallback

training/mappo/
  config.py         — MAPPOConfig dataclass (all hyperparameters)
  rollout_buffer.py — Pre-allocated buffer + GAE computation
  reward_builder.py — Dense + terminal reward shaping
  ppo_update.py     — PPO clipped surrogate update step
  league_manager.py — Opponent pool (baselines + checkpoint pool)
  train_mappo.py    — Main training entry point
  bc_warmstart.py   — Behavior cloning from rule-based teacher
  evaluate_agent.py — Evaluation + TrueSkill estimation + CSV logging
  logger.py         — CSV + JSONL + optional TensorBoard logging

scripts/participant/
  export_submission.py  — Package agent into submission.zip with validation
```

---

## Recommended Training Pipeline

### Phase 0: Environment Setup (once)
See `SETUP.md`.

### Phase 1 (Optional): Behavior Cloning Warm-Start

Trains the actor to imitate a rule-based teacher agent before RL.  
Recommended to avoid poor early exploration.

```bash
# Collect 500 episodes from TacticalRuleAgent
python -m training.mappo.bc_warmstart --teacher tactical --episodes 500
# Output: checkpoints/mappo/bc_warmstart_tactical.pth

# Or use GeniusRuleAgent for higher-quality demonstrations
python -m training.mappo.bc_warmstart --teacher genius --episodes 300
# Output: checkpoints/mappo/bc_warmstart_genius.pth
```

Typical accuracy after 10 epochs: ~60-70% action match.

### Phase 2: PPO vs Baselines

Start from BC warm-start or random init. Train against rule-based baselines.

```bash
# Start from BC warm-start
python -m training.mappo.train_mappo \
  --phase baseline \
  --resume checkpoints/mappo/bc_warmstart_genius.pth \
  --num-envs 4 \
  --total-steps 2000000

# Or start from scratch
python -m training.mappo.train_mappo --phase baseline --num-envs 4
```

**Training logs:** `logs/mappo/train.csv` and `logs/mappo/train.jsonl`

Typical Phase 2 duration: ~2-4 hours on GPU, 12-24 hours on CPU for 2M steps.

### Phase 3: Self-Play (League)

Resume from a Phase 2 checkpoint. The league manager samples 50% historical checkpoints.

```bash
python -m training.mappo.train_mappo \
  --phase selfplay \
  --resume checkpoints/mappo/update_001000.pth \
  --num-envs 4 \
  --total-steps 5000000
```

### Phase 4: Evaluation

```bash
# Evaluate a checkpoint vs fixed seed suite
python -m training.mappo.evaluate_agent \
  --checkpoint checkpoints/mappo/update_002000.pth \
  --matches 50 \
  --seed-suite 0 \
  --baselines tactical,genius,smarter

# Save match JSON logs for debugging
python -m training.mappo.evaluate_agent \
  --checkpoint checkpoints/mappo/update_002000.pth \
  --matches 10 \
  --save-match-logs
```

**Evaluation logs:** `logs/mappo/eval.csv` and `logs/mappo/eval.jsonl`

### Checkpoint Selection & Retention

The training script automatically manages a checkpoint retention policy to avoid unbounded disk usage:
1. `latest.pth`: Always overwritten with the most recent checkpoint.
2. `best.pth`: Overwritten only when the internal evaluation metric improves (configured via `best_metric` in MAPPOConfig). This file is never deleted.
3. `checkpoint_update_{N}.pth`: Numbered historical checkpoints are saved every `checkpoint_every` steps. Only the most recent `keep_last_checkpoints` (default 3) are kept; older ones are automatically deleted.

**Never select checkpoints by training reward alone.** The script evaluates and tracks `best.pth` using `estimated_score` (μ − 3σ) by default. The export script will automatically use `best.pth` if you don't specify one.

If manually comparing metrics, use:

| Metric | Target |
|---|---|
| `average_rank` | Lower is better (best = 0) |
| `win_rate` | Higher is better |
| `estimated_score` = μ − 3σ | Higher is better |
| `timeout_count` + `error_count` | Must be 0 |

Compare candidates across the same `fixed_seed_suite_id` for fair comparison.

---

## Export Submission

```bash
# Auto-export (automatically selects best.pth or latest.pth)
python -m scripts.participant.export_submission

# Export specific checkpoint with model (NN agent)
python -m scripts.participant.export_submission \
  --checkpoint checkpoints/mappo/checkpoint_update_2000.pth \
  --output submission.zip

# Export fallback-only (MinimalTacticalFallback, no model needed)
python -m scripts.participant.export_submission \
  --fallback-only \
  --output submission_fallback.zip
```

Validation checks (auto-run):
- Exactly one `agent.py` at zip root
- `model.pth` present (unless `--fallback-only`)
- Zip ≤ 100 MB, extracted ≤ 300 MB, ≤ 20 files
- Smoke test: `Agent(0).act(obs)` returns valid action

---

## Configuration

Edit hyperparameters in `training/mappo/config.py` or pass a JSON config:

```bash
# Save config
python -c "
from training.mappo.config import MAPPOConfig
cfg = MAPPOConfig(num_envs=8, total_env_steps=10_000_000)
cfg.save('my_config.json')
"

# Use config
python -m training.mappo.train_mappo --config my_config.json
```

Key hyperparameters:

| Param | Default | Notes |
|---|---|---|
| `gamma` | 0.995 | Discount factor |
| `gae_lambda` | 0.95 | GAE smoothing |
| `clip_eps` | 0.15 | PPO clip |
| `actor_lr` | 3e-4 | Adam LR for actor |
| `critic_lr` | 5e-4 | Adam LR for critic |
| `entropy_coef` | 0.01 | Exploration bonus |
| `dense_reward_coef` | 1.0 | Anneal to 0.1 in Phase 4 |
| `num_envs` | 8 | Parallel envs (sequential sim) |
| `rollout_length` | 256 | Steps per rollout |

---

## Reward Shaping

**Terminal:**
| Outcome | Reward |
|---|---|
| Unique survivor | +1.00 |
| Shared best rank | +0.70 |
| Rank 2/4 | +0.15 |
| Rank 3/4 | -0.20 |
| First eliminated | -0.80 |

**Dense (×dense_reward_coef):**
| Event | Reward |
|---|---|
| Kill enemy | +0.25 |
| Destroy box | +0.02 |
| Collect item | +0.06 |
| Place useful bomb | +0.01 |
| Escape immediate danger | +0.05 |
| Enter immediate danger | -0.08 |
| Place bomb without escape | -0.10 |
| Suicide/self-blast | -0.20 |

---

## Submission Requirements (Competition)

- Single `agent.py` + `model.pth` in flat `.zip`
- Zip ≤ 100 MB, extracted ≤ 300 MB, ≤ 20 files
- Startup: ≤ 20s  
- `act()`: ≤ 100 ms  
- CPU-only (no CUDA on eval server)
- No network calls, no file writes inside `act()`
- Deadline: **21/6/2026**

---

## Local Match Testing

```bash
# Test MAPPO agent vs 3 baselines
python scripts/participant/run_local_match.py \
  --agent_paths \
    agent/mappo_agent \
    agent/tactical_rule_agent.py \
    agent/genius_rule_agent.py \
    agent/smarter_rule_agent.py
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ImportError: No module named torch` | Run `venv_mappo\Scripts\activate` first |
| Pygame fails to install | Safe to ignore — only affects GIF visualizer |
| `model.pth` not found | Agent falls back to `MinimalTacticalFallback` |
| Timeout in `act()` | Model too large or heavy ops in `__init__` |
| Poor win rate after training | Try BC warm-start first; check safety layer isn't over-blocking |
