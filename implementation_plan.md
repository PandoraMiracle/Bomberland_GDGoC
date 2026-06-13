# MAPPO/PPO Self-Play RL Pipeline — Updated Implementation Plan

## Background

4-player 13×13 Bomberland. All new files are **additive** under `agent/mappo_agent/` and `training/mappo/`. Zero existing files modified.

**Verified from engine source:**

| Fact | Detail |
|---|---|
| Actions | 0=STOP, 1=LEFT(`dx=-1,dy=0`), 2=RIGHT(`dx=+1,dy=0`), 3=UP(`dx=0,dy=-1`), 4=DOWN(`dx=0,dy=+1`), 5=PLACE_BOMB |
| Coord system | `player.x=row`, `player.y=col`; movement applies `new_x = x+dx, new_y = y+dy` |
| Bomb timer | Starts at 7, step() decrements, explodes when `timer <= 0` (so effective countdown is 7 ticks) |
| Block movement | Cannot enter WALL(1) or BOX(2) or tile with existing bomb (from previous steps) |
| Bomb placement | Blocked if `bombs_left <= 0` OR bomb already on tile from previous step |
| Startup timeout | `AgentProcessExecutor.start()` default `startup_timeout_s=2.0` (local); server uses env var `EVALUATION_STARTUP_TIMEOUT_S`, defaulting to `max(1.0, timeout_s*10)` = ~1s locally, 20s on server |
| Inference timeout | `act_with_timeout(..., timeout_s=0.1)` = 100ms |
| Match log format | JSON: `seed, team_ids, meta, ranks, survival_steps, runtime_stats, history[{step,actions,alive,map,players,bombs}]` |

---

## Decisions Incorporated

### 1. Actor Encoding — Self + 3 Opponents in Relative Seat Order
Encode all 4 players as:
- Channel self: `players[agent_id]`
- Channel opp_1: `players[(agent_id+1)%4]`
- Channel opp_2: `players[(agent_id+2)%4]`
- Channel opp_3: `players[(agent_id+3)%4]`

Keeps full information while making policy seat-invariant. **18 spatial channels + 22 scalar features retained.**

### 2. Logging — CSV+JSONL mandatory, TensorBoard optional

**A. Training logs** → `logs/mappo/train.csv` + `logs/mappo/train.jsonl`  
Per PPO update: `update, env_steps, episodes_completed, fps, actor_loss, critic_loss, entropy, approx_kl, clip_fraction, explained_variance, mean_return, mean_episode_length, dense_reward_coef, actor_lr, critic_lr, checkpoint_path`

**B. Evaluation logs** → `logs/mappo/eval.csv` + `logs/mappo/eval.jsonl`  
Per eval run: `checkpoint_path, num_matches, fixed_seed_suite_id, win_rate, draw_rate, loss_rate, average_rank, average_survival_steps, average_kills, average_boxes, average_items, average_bombs, timeout_count, error_count, invalid_action_count, fallback_uses, estimated_mu, estimated_sigma, estimated_score`

**C. Match debug logs** → `logs/mappo/matches/json/` (mirrors server format exactly), GIFs only with `--save-gifs`. No Drive upload from training scripts.

**D. TensorBoard** — `try: import tensorboard` guarded; silently omitted if unavailable.

**E. Submission agent** — `DEBUG = False` constant. No print/log in `__init__` or `act()`. All exceptions caught → safe fallback action.

### 3. Fallback — Embedded `MinimalTacticalFallback`
Fully inlined in `agent.py` (no import from baseline files). Uses only numpy + stdlib. Priority:
1. Escape immediate danger (BFS depth 8)
2. Pick nearby safe item
3. Place bomb near box/enemy if escape path exists
4. Move toward box/item/enemy via BFS
5. Safe legal move
6. STOP

### 4. Direction Mapping — Verified from Engine
`game.py` line 56-59:
```python
if action == Player.LEFT:  dx = -1   # x decreases (row up)
elif action == Player.RIGHT: dx = 1  # x increases (row down)
elif action == Player.UP:   dy = -1  # y decreases (col left)
elif action == Player.DOWN: dy = 1   # y increases (col right)
```
Safety/encoder uses identical deltas. Unit test validates our next-position logic against engine behavior on controlled map.

### 5. Checkpoint Selection Criteria
Never select by training reward alone. Use:
- Average rank (lower is better)
- Unique-best (win) rate
- Draw rate
- Timeout/error/invalid counts (lower is better)
- Estimated TrueSkill score = `mu - 3*sigma`

### 6. ZIP Validation
Export script validates: exactly one `agent.py`, model file present if configured, fallback works if model missing, zip ≤100MB, extracted ≤300MB, ≤20 files, no logs/checkpoints included.

### 7. Implementation Order (Per User Instruction)
```
Phase 1: encoder + safety + agent contract + tracker
Phase 2: tests (shapes, forward, safety, act contract, timing)
Phase 3: BC warm-start + evaluate_agent
Phase 4: PPO trainer (rollout buffer, reward builder, ppo_update)
Phase 5: league/self-play + train_mappo main script
Phase 6: export + TRAINING_README
```

---

## Proposed Changes — File Tree

```
agent/mappo_agent/
  __init__.py
  model.py          ← CNNActor, CentralizedCritic, ResBlock
  encoder.py        ← encode_obs_mappo() — 18 spatial + 22 scalar
  tracker.py        ← AgentTracker (step/bomb/item/box/kill estimates)
  safety.py         ← legal_mask(), apply_safety(), blast_tiles(), has_escape()
  agent.py          ← Agent class + MinimalTacticalFallback (submission-ready)

training/
  __init__.py
  mappo/
    __init__.py
    config.py         ← MAPPOConfig dataclass
    rollout_buffer.py ← RolloutBuffer + compute_gae()
    reward_builder.py ← RewardBuilder (dense + terminal)
    ppo_update.py     ← ppo_update() returning loss metrics dict
    league_manager.py ← LeagueManager (baseline pool + checkpoint pool)
    train_mappo.py    ← main training entry point
    bc_warmstart.py   ← behavior cloning from baselines
    evaluate_agent.py ← eval loop + TrueSkill sim + CSV/JSONL output
    logger.py         ← TrainingLogger (CSV+JSONL+optional TB)

scripts/participant/
  export_submission.py  ← packaging + validation

tests/
  __init__.py
  test_encoder.py       ← shape tests, direction unit test vs engine
  test_model.py         ← forward pass, device handling
  test_safety.py        ← legal mask, escape logic
  test_agent.py         ← act() contract, fallback, timing benchmark
  test_smoke.py         ← one match, smoke training (2 envs, 32 steps)

TRAINING_README.md
SETUP.md
```
**Total: 22 new files** (no existing files modified)

---

## Architecture Spec

### CNNActor (spatial=(18,13,13), scalar=22) → logits(6)
```
Conv2d(18, 64, 3, pad=1) → ReLU
ResBlock(64) × 2          [each: Conv→BN→ReLU→Conv→BN + skip]
Conv2d(64, 96, 3, pad=1) → ReLU → GlobalAvgPool → (96,)
ScalarMLP: 22→64→ReLU→64→ReLU  → (64,)
FusionMLP: 160→128→ReLU→64→ReLU → PolicyHead: 64→6
```

### CentralizedCritic (global_spatial=(18,13,13), global_scalar=~32) → value(1)
Same CNN trunk → global scalar MLP (32→128→64) → Fusion 160→128→1

### Encoder — 18 Spatial Channels
```
0:  grass mask
1:  wall mask
2:  box mask
3:  item_radius mask
4:  item_capacity mask
5:  self position
6:  opp1 position [(agent_id+1)%4]
7:  opp2 position [(agent_id+2)%4]
8:  opp3 position [(agent_id+3)%4]
9:  bomb occupancy
10: bomb timer / 7.0
11: own bomb mask (owner_id == agent_id)
12: enemy bomb mask
13: danger_t1 (blast tiles of bombs with timer==1)
14: danger_t2 (timer<=2)
15: danger_t3 (timer<=3)
16: prospective blast if placing bomb now
17: passable mask (grass|item, no bomb)
```

### Encoder — 22 Scalar Features
```
0:  bombs_left / 5.0
1:  bomb_radius / 5.0  (= 1 + radius_bonus)
2:  step / 500.0
3:  alive_opponents / 3.0
4:  est_boxes_destroyed / 20.0
5:  est_items_collected / 10.0
6:  est_bombs_placed / 20.0
7:  est_kills / 3.0
8-13: last_action one-hot (6)
14: on_danger_t1 flag
15: on_danger_t2 flag
16: can_place_bomb flag
17: dist_nearest_item / 18.0
18: dist_nearest_enemy / 18.0
19: idle_streak / 10.0
20: opp1_alive
21: opp2_alive  [opp3_alive derived from alive_opponents-opp1-opp2 but kept implicit]
```
*Total: 22 scalars*

---

## Hyperparameters

```python
gamma          = 0.995
gae_lambda     = 0.95
clip_eps       = 0.15
actor_lr       = 3e-4
critic_lr      = 5e-4
ppo_epochs     = 4
rollout_length = 256
num_envs       = 8      # configurable to 32/64
entropy_coef   = 0.01
value_coef     = 1.0
max_grad_norm  = 0.5
dense_reward_coef = 1.0  # annealed toward 0.1 by Phase 4
```

---

## Reward Spec

**Terminal** (applied at episode end):
- Unique best rank: `+1.0`
- Shared best rank: `+0.7`
- Rank 2 of 4: `+0.15`
- Rank 3 of 4: `-0.2`
- Rank 4 (first eliminated): `-0.8`

**Dense** (per step, scaled by `dense_reward_coef`):
- Kill: `+0.25`
- Destroy box: `+0.02`
- Collect item: `+0.06`
- Place useful bomb (near box/enemy + escape exists): `+0.01`
- Enter immediate danger: `-0.08`
- Escape immediate danger: `+0.05`
- Place bomb without escape: `-0.10`
- Suicide/self-blast: additional `-0.20`

---

## Verification Plan

### Commands (in order)
```bash
# Phase 1 — syntax
python -m compileall agent/mappo_agent training/mappo tests -q

# Phase 2 — unit tests
python -m pytest tests/test_encoder.py tests/test_model.py tests/test_safety.py tests/test_agent.py -v

# Phase 3 — one full match
python -m pytest tests/test_smoke.py::test_one_match -v

# Phase 4 — timing benchmark
python -m pytest tests/test_agent.py::test_inference_timing -v --tb=short

# Phase 5 — smoke training
python -m pytest tests/test_smoke.py::test_smoke_training -v

# Phase 6 — evaluate checkpoint
python -m training.mappo.evaluate_agent --checkpoint checkpoints/mappo/smoke.pth --matches 20

# Phase 7 — export
python -m scripts.participant.export_submission --checkpoint checkpoints/mappo/smoke.pth --output submission.zip

# Phase 8 — local match with agent
python -m scripts.participant.run_local_match \
  --agent_paths agent/mappo_agent None None None --num_episodes 1
```

### Checkpoint Selection
Compare candidates by: `avg_rank → win_rate → estimated_score (mu-3*sigma) → error_count`

---

## Open Questions (Resolved)

| # | Question | Decision |
|---|---|---|
| 1 | Encoding seats | Self + 3 opponents relative order `(agent_id+k)%4` |
| 2 | Logging | CSV+JSONL mandatory, TensorBoard optional |
| 3 | Fallback | Embedded `MinimalTacticalFallback`, no import from baseline files |
| 4 | Directions | Verified: LEFT=dx-1, RIGHT=dx+1, UP=dy-1, DOWN=dy+1 |
| 5 | Startup timeout | Server: 20s via env var; local tests: strict ~2s + faithful 20s |
| 6 | Training device | GPU if available; submitted agent forces CPU |
| 7 | Scalar count | 22 (indices 0-21 above) |
