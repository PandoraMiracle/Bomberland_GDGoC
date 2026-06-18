"""
agent.py — Submission-ready MAPPO Agent for Bomberland.

Submission zip must contain this file (flat, alongside model.pth).
Loads model once in __init__; uses torch.inference_mode() in act().
Falls back to MinimalTacticalFallback if model loading fails.

Constraints satisfied:
  ✓ No print/log in __init__ or act()  (DEBUG=False by default)
  ✓ No network calls, no file writes in act()
  ✓ act() always returns int in [0, 5]
  ✓ Startup ≤ 20s  (model is small ~3-5 MB)
  ✓ act() ≤ 100ms  (CPU inference, small model)
  ✓ No imports from baseline agent files
"""

from __future__ import annotations

import os
# Force single-threaded BLAS to match eval server environment
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
from collections import deque
import numpy as np

DEBUG = False  # Never set True in submission

# Prefer shared encoder/tracker when running from the repo package.
try:
    from agent.mappo_agent.tracker import AgentTracker as _AgentTracker
    from agent.mappo_agent.encoder import encode_obs as _encode_obs_shared
    _HAS_SHARED_MAPPO = True
except ImportError:
    try:
        from .tracker import AgentTracker as _AgentTracker
        from .encoder import encode_obs as _encode_obs_shared
        _HAS_SHARED_MAPPO = True
    except ImportError:
        _HAS_SHARED_MAPPO = False
        _AgentTracker = None
        _encode_obs_shared = None

# ── optional torch import ─────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Minimal Tactical Fallback (no external imports, stdlib+numpy only)
# ═══════════════════════════════════════════════════════════════════════════════

class _MinimalTacticalFallback:
    """
    Self-contained rule-based agent used when model loading fails.
    Implements: escape → item grab → safe bomb → move toward target → safe move.
    Never imports from baseline agent files.
    """

    # Action deltas verified against engine/game.py
    _DELTAS = {0:(0,0), 1:(-1,0), 2:(1,0), 3:(0,-1), 4:(0,1)}

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _in_bounds(grid: np.ndarray, x: int, y: int) -> bool:
        return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]

    @staticmethod
    def _passable(grid: np.ndarray, x: int, y: int) -> bool:
        if not (0 < x < grid.shape[0]-1 and 0 < y < grid.shape[1]-1):
            return False
        return int(grid[x, y]) in (0, 3, 4)

    def _blast_tiles(self, grid: np.ndarray, bx: int, by: int, radius: int) -> set:
        tiles = {(bx, by)}
        for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
            for r in range(1, radius+1):
                tx, ty = bx+dx*r, by+dy*r
                if not self._in_bounds(grid, tx, ty): break
                cell = int(grid[tx, ty])
                if cell == 1: break
                tiles.add((tx, ty))
                if cell == 2: break
        return tiles

    def _danger_tiles(self, grid: np.ndarray, bombs: np.ndarray,
                      players: np.ndarray) -> tuple[set, set]:
        soon, now = set(), set()
        for b in bombs:
            bx, by, timer = int(b[0]), int(b[1]), int(b[2])
            oid = int(b[3]) if len(b) > 3 else 0
            if timer <= 0: continue
            radius = 1 + int(players[oid, 4]) if 0 <= oid < len(players) else 2
            blast = self._blast_tiles(grid, bx, by, radius)
            soon |= blast
            if timer <= 1: now |= blast
        return soon, now

    def _legal_moves(self, grid: np.ndarray, pos: tuple,
                     bomb_pos: set) -> list[int]:
        actions = [0]
        for a in (1,2,3,4):
            dx, dy = self._DELTAS[a]
            nx, ny = pos[0]+dx, pos[1]+dy
            if self._passable(grid, nx, ny) and (nx, ny) not in bomb_pos:
                actions.append(a)
        return actions

    def _bfs_move(self, grid: np.ndarray, start: tuple, targets: set,
                  blocked: set, avoid: set) -> int | None:
        if not targets: return None
        q: deque = deque([(start, None)])
        seen = {start}
        while q:
            pos, first = q.popleft()
            if pos in targets and first is not None:
                return first
            for a in (1,2,3,4):
                dx, dy = self._DELTAS[a]
                nx, ny = pos[0]+dx, pos[1]+dy
                np_ = (nx, ny)
                if np_ in seen or not self._passable(grid, nx, ny): continue
                if np_ in blocked or np_ in avoid: continue
                seen.add(np_)
                q.append((np_, a if first is None else first))
        return None

    def _has_escape(self, grid: np.ndarray, pos: tuple, bomb_pos: set,
                    danger: set, depth: int = 8) -> bool:
        if pos not in danger: return True
        q: deque = deque([(pos, 0)])
        seen = {pos}
        H, W = grid.shape
        while q:
            p, d = q.popleft()
            if d >= depth: continue
            for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
                nx, ny = p[0]+dx, p[1]+dy
                np_ = (nx, ny)
                if np_ in seen: continue
                if not (0 < nx < H-1 and 0 < ny < W-1): continue
                if int(grid[nx, ny]) in (1, 2) or np_ in bomb_pos: continue
                if np_ not in danger: return True
                seen.add(np_); q.append((np_, d+1))
        return False

    # ── main decision ─────────────────────────────────────────────────────────

    def act(self, obs: dict) -> int:
        try:
            return self._decide(obs)
        except Exception:
            return 0

    def _decide(self, obs: dict) -> int:
        grid    = np.asarray(obs["map"], dtype=np.int32)
        players = np.asarray(obs["players"], dtype=np.int32)
        bombs_arr = obs.get("bombs")
        bombs = np.asarray(bombs_arr, dtype=np.int32) if (
            bombs_arr is not None and np.asarray(bombs_arr).size > 0
        ) else np.zeros((0, 4), dtype=np.int32)
        if bombs.ndim == 1 and bombs.size == 4:
            bombs = bombs.reshape(1, 4)

        aid = self.agent_id
        if aid >= len(players) or int(players[aid, 2]) != 1:
            return 0

        my_x, my_y = int(players[aid, 0]), int(players[aid, 1])
        my_pos = (my_x, my_y)
        bombs_left  = int(players[aid, 3])
        my_radius   = 1 + int(players[aid, 4])
        bomb_pos = {(int(b[0]), int(b[1])) for b in bombs}
        danger_soon, danger_now = self._danger_tiles(grid, bombs, players)
        legal = self._legal_moves(grid, my_pos, bomb_pos)
        enemies = [(int(players[i,0]), int(players[i,1]))
                   for i in range(len(players))
                   if i != aid and int(players[i,2]) == 1]

        # 1. Escape immediate danger
        if my_pos in danger_now or my_pos in danger_soon:
            for a in (1,2,3,4):
                if a not in legal: continue
                dx, dy = self._DELTAS[a]
                np_ = (my_x+dx, my_y+dy)
                if np_ not in danger_now and np_ not in danger_soon:
                    return a
            for a in (1,2,3,4):
                if a not in legal: continue
                dx, dy = self._DELTAS[a]
                np_ = (my_x+dx, my_y+dy)
                if np_ not in danger_now:
                    return a
            return 0

        # 2. Collect nearby safe item
        items = {(r,c) for r in range(grid.shape[0]) for c in range(grid.shape[1])
                 if int(grid[r,c]) in (3,4)}
        if items:
            mv = self._bfs_move(grid, my_pos, items, bomb_pos, danger_soon)
            if mv is not None:
                return mv

        # 3. Place bomb near box/enemy if escape exists
        if bombs_left > 0 and my_pos not in bomb_pos:
            my_blast = self._blast_tiles(grid, my_x, my_y, my_radius)
            hits_box   = any(int(grid[tx, ty]) == 2 for tx,ty in my_blast)
            hits_enemy = any(p in my_blast for p in enemies)
            if hits_box or hits_enemy:
                combined = danger_soon | my_blast
                if self._has_escape(grid, my_pos, bomb_pos, combined):
                    return 5

        # 4. Move toward box spots
        box_spots = {
            (x+dx, y+dy)
            for x in range(grid.shape[0]) for y in range(grid.shape[1])
            if int(grid[x,y]) == 2
            for dx,dy in ((-1,0),(1,0),(0,-1),(0,1))
            if self._passable(grid, x+dx, y+dy) and (x+dx,y+dy) not in bomb_pos
        }
        if box_spots:
            mv = self._bfs_move(grid, my_pos, box_spots, bomb_pos, danger_soon)
            if mv is not None: return mv

        # 5. Move toward enemy
        if enemies:
            mv = self._bfs_move(grid, my_pos, set(enemies), bomb_pos, danger_soon)
            if mv is not None: return mv

        # 6. Safe legal move
        safe_moves = [a for a in legal if a != 0 and
                      (my_x + self._DELTAS[a][0], my_y + self._DELTAS[a][1])
                      not in danger_soon]
        if safe_moves:
            return safe_moves[0]

        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Model definition (copy of agent/mappo_agent/model.py)
#             Inlined so the submission is self-contained.
# ═══════════════════════════════════════════════════════════════════════════════

if _TORCH_AVAILABLE:
    class _ResBlock(nn.Module):
        def __init__(self, ch: int):
            super().__init__()
            self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.gn1   = nn.GroupNorm(8, ch)
            self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.gn2   = nn.GroupNorm(8, ch)

        def forward(self, x):
            r = x
            x = F.relu(self.gn1(self.conv1(x)))
            x = self.gn2(self.conv2(x))
            return F.relu(x + r)

    class _CNNActor(nn.Module):
        def __init__(self, n_spatial=18, n_scalar=28, n_actions=6):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(n_spatial, 64, 3, padding=1), nn.ReLU(),
                _ResBlock(64), _ResBlock(64),
                nn.Conv2d(64, 96, 3, padding=1),         nn.ReLU(),
            )
            self.scalar_mlp = nn.Sequential(
                nn.Linear(n_scalar, 64), nn.ReLU(),
                nn.Linear(64, 64),       nn.ReLU(),
            )
            self.fusion = nn.Sequential(
                nn.Linear(160, 128), nn.ReLU(),
                nn.Linear(128, 64),  nn.ReLU(),
            )
            self.head = nn.Linear(64, n_actions)

        def forward(self, spatial, scalar):
            cnn_f = self.cnn(spatial).mean(dim=(-2, -1))
            s_f   = self.scalar_mlp(scalar)
            return self.head(self.fusion(torch.cat([cnn_f, s_f], dim=-1)))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Encoder (copy of agent/mappo_agent/encoder.py key function)
# ═══════════════════════════════════════════════════════════════════════════════

def _encode_obs_submission(obs: dict, agent_id: int, tracker=None) -> tuple[np.ndarray, np.ndarray]:
    """Inline encoder for submission. Returns (spatial (18,13,13), scalar (28,))."""
    if _HAS_SHARED_MAPPO and _encode_obs_shared is not None:
        return _encode_obs_shared(obs, agent_id, tracker)
    GRASS, WALL, BOX, ITEM_R, ITEM_C = 0, 1, 2, 3, 4
    H, W = 13, 13

    g = obs.get("map")
    grid = np.asarray(g, dtype=np.int32) if g is not None else np.zeros((H,W), dtype=np.int32)
    p = obs.get("players")
    players = np.asarray(p, dtype=np.int32) if p is not None else np.zeros((4,5), dtype=np.int32)
    b = obs.get("bombs")
    if b is None or np.asarray(b).size == 0:
        bombs = np.zeros((0,4), dtype=np.int32)
    else:
        bombs = np.asarray(b, dtype=np.int32)
        if bombs.ndim == 1 and bombs.size == 4: bombs = bombs.reshape(1,4)

    aid = int(agent_id)
    n_pl = players.shape[0]

    my_row  = int(players[aid,0]) if n_pl > aid else 1
    my_col  = int(players[aid,1]) if n_pl > aid else 1
    my_alive      = int(players[aid,2]) if n_pl > aid else 0
    my_bombs_left = int(players[aid,3]) if n_pl > aid else 0
    my_radius_bon = int(players[aid,4]) if n_pl > aid else 0
    my_radius = 1 + my_radius_bon

    bomb_pos_set = {(int(b_[0]),int(b_[1])) for b_ in bombs}

    # Danger maps
    def _blast(bx, by, rad):
        tiles = {(bx,by)}
        for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
            for r in range(1,rad+1):
                tx,ty = bx+dx*r, by+dy*r
                if not (0<=tx<H and 0<=ty<W): break
                cell = int(grid[tx,ty])
                if cell==WALL: break
                tiles.add((tx,ty))
                if cell==BOX: break
        return tiles

    d1,d2,d3 = np.zeros((H,W),dtype=np.float32), np.zeros((H,W),dtype=np.float32), np.zeros((H,W),dtype=np.float32)
    for b_ in bombs:
        bx,by,timer = int(b_[0]),int(b_[1]),int(b_[2])
        oid = int(b_[3]); timer_=int(timer)
        if timer_<=0: continue
        rad = 1+(int(players[oid,4]) if 0<=oid<n_pl else 0)
        for tx,ty in _blast(bx,by,rad):
            if timer_<=3: d3[tx,ty]=1.0
            if timer_<=2: d2[tx,ty]=1.0
            if timer_<=1: d1[tx,ty]=1.0

    channels = []
    for tv in (GRASS,WALL,BOX,ITEM_R,ITEM_C):
        channels.append((grid==tv).astype(np.float32))

    self_ch = np.zeros((H,W),dtype=np.float32)
    if my_alive: self_ch[my_row,my_col]=1.0
    channels.append(self_ch)

    for k in (1,2,3):
        oi = (aid+k)%4
        ch = np.zeros((H,W),dtype=np.float32)
        if n_pl>oi and int(players[oi,2])==1:
            ch[int(players[oi,0]),int(players[oi,1])]=1.0
        channels.append(ch)

    bomb_occ = np.zeros((H,W),dtype=np.float32)
    for bx,by in bomb_pos_set: bomb_occ[bx,by]=1.0
    channels.append(bomb_occ)

    btimer = np.zeros((H,W),dtype=np.float32)
    for b_ in bombs:
        bx,by,t=int(b_[0]),int(b_[1]),int(b_[2])
        btimer[bx,by]=max(btimer[bx,by],float(t)/7.0)
    channels.append(btimer)

    own_b = np.zeros((H,W),dtype=np.float32)
    emy_b = np.zeros((H,W),dtype=np.float32)
    for b_ in bombs:
        bx,by=int(b_[0]),int(b_[1])
        if int(b_[3])==aid: own_b[bx,by]=1.0
        else: emy_b[bx,by]=1.0
    channels.extend([own_b, emy_b, d1, d2, d3])

    prosp = np.zeros((H,W),dtype=np.float32)
    if my_alive and my_bombs_left>0 and (my_row,my_col) not in bomb_pos_set:
        for tx,ty in _blast(my_row,my_col,my_radius): prosp[tx,ty]=1.0
    channels.append(prosp)

    passable = np.zeros((H,W),dtype=np.float32)
    for r in range(H):
        for c in range(W):
            if int(grid[r,c]) in (0,3,4) and (r,c) not in bomb_pos_set:
                passable[r,c]=1.0
    channels.append(passable)

    spatial = np.stack(channels, axis=0).astype(np.float32)

    # Scalar (22)
    alive_opp = sum(1 for k in (1,2,3)
                    if n_pl>(aid+k)%4 and int(players[(aid+k)%4,2])==1)
    opp1 = float(int(players[(aid+1)%4,2])==1) if n_pl>(aid+1)%4 else 0.0
    opp2 = float(int(players[(aid+2)%4,2])==1) if n_pl>(aid+2)%4 else 0.0
    on_d1 = float(d1[my_row,my_col]) if my_alive else 0.0
    on_d2 = float(d2[my_row,my_col]) if my_alive else 0.0
    can_pl = float(my_alive and my_bombs_left>0 and (my_row,my_col) not in bomb_pos_set)
    last_action = int(tracker.last_action) if tracker else 0
    last_oh = np.zeros(6,dtype=np.float32)
    if 0 <= last_action <= 5:
        last_oh[last_action] = 1.0
    est_step    = float(tracker.estimated_step)  if tracker else 0.0
    est_boxes   = float(tracker.boxes_destroyed) if tracker else 0.0
    est_items   = float(tracker.items_collected) if tracker else 0.0
    est_bombs   = float(tracker.bombs_placed)    if tracker else 0.0
    est_kills   = float(tracker.kills)           if tracker else 0.0
    idle_streak = float(tracker.idle_streak)     if tracker else 0.0

    legal = np.zeros(6, dtype=np.float32)
    legal[0] = 1.0
    if my_alive:
        for a, (dx, dy) in ((1, (-1, 0)), (2, (1, 0)), (3, (0, -1)), (4, (0, 1))):
            nx, ny = my_row + dx, my_col + dy
            if not (0 < nx < H - 1 and 0 < ny < W - 1):
                continue
            if int(grid[nx, ny]) in (WALL, BOX) or (nx, ny) in bomb_pos_set:
                continue
            legal[a] = 1.0
        if my_bombs_left > 0 and (my_row, my_col) not in bomb_pos_set:
            legal[5] = 1.0

    scalar = np.array([
        float(my_bombs_left)/5.0, float(my_radius)/5.0,
        est_step/500.0, float(alive_opp)/3.0,
        est_boxes/20.0, est_items/10.0, est_bombs/20.0, est_kills/3.0,
        *last_oh,
        on_d1, on_d2, can_pl,
        1.0, 1.0,  # dist_item, dist_enemy (normalized max = unknown w/o BFS)
        idle_streak/10.0, opp1, opp2,
        *legal,
    ], dtype=np.float32)

    return spatial, scalar


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Safety (inline version for submission)
# ═══════════════════════════════════════════════════════════════════════════════

def _submission_apply_safety(logits: np.ndarray, obs: dict, agent_id: int) -> int:
    """Lightweight safety wrapper for submission agent."""
    try:
        g = obs.get("map"); p = obs.get("players"); b = obs.get("bombs")
        grid = np.asarray(g, dtype=np.int32) if g is not None else np.zeros((13,13),dtype=np.int32)
        players = np.asarray(p, dtype=np.int32) if p is not None else np.zeros((4,5),dtype=np.int32)
        bombs_raw = np.asarray(b, dtype=np.int32) if (b is not None and np.asarray(b).size>0) else np.zeros((0,4),dtype=np.int32)
        if bombs_raw.ndim==1 and bombs_raw.size==4: bombs_raw=bombs_raw.reshape(1,4)

        aid=int(agent_id); n_pl=players.shape[0]; H,W=grid.shape
        if n_pl<=aid or int(players[aid,2])==0: return 0

        mx,my_=int(players[aid,0]),int(players[aid,1])
        bbl=int(players[aid,3]); my_rad=1+int(players[aid,4])
        bomb_pos={(int(x[0]),int(x[1])) for x in bombs_raw}

        def _blast_s(bx,by,rad):
            tiles={(bx,by)}
            for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
                for r in range(1,rad+1):
                    tx,ty=bx+dx*r,by+dy*r
                    if not(0<=tx<H and 0<=ty<W): break
                    c=int(grid[tx,ty])
                    if c==1: break
                    tiles.add((tx,ty))
                    if c==2: break
            return tiles

        d1=set()
        for b_ in bombs_raw:
            bx,by,t=int(b_[0]),int(b_[1]),int(b_[2])
            if t<=0 or t>1: continue
            oid=int(b_[3]); rad=1+(int(players[oid,4]) if 0<=oid<n_pl else 0)
            d1|=_blast_s(bx,by,rad)

        deltas={0:(0,0),1:(-1,0),2:(1,0),3:(0,-1),4:(0,1)}
        legal=np.zeros(6,dtype=bool); legal[0]=True
        for a,(dx,dy) in deltas.items():
            if a==0: continue
            nx,ny=mx+dx,my_+dy
            if not(0<nx<H-1 and 0<ny<W-1): continue
            if int(grid[nx,ny]) in (1,2) or (nx,ny) in bomb_pos: continue
            legal[a]=True
        if bbl>0 and (mx,my_) not in bomb_pos: legal[5]=True

        safe=legal.copy()
        if (mx,my_) in d1:
            safe[0]=False; safe[5]=False
            for a in range(1,5):
                if not legal[a]: safe[a]=False; continue
                dx,dy=deltas[a]
                if (mx+dx,my_+dy) in d1: safe[a]=False

        arr=np.asarray(logits,dtype=np.float32)
        if arr.shape[0]!=6: arr=np.zeros(6,dtype=np.float32)
        si=np.where(safe)[0]
        if si.size>0: return int(si[np.argmax(arr[si])])
        li=np.where(legal)[0]
        if li.size>0: return int(li[np.argmax(arr[li])])
        return 0
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Main Agent class (submission interface)
# ═══════════════════════════════════════════════════════════════════════════════

class Agent:
    """
    Submission agent.  Must define __init__(agent_id) and act(obs) -> int.
    """

    team_id = "MAPPOAgent"

    # Model config — must match training
    _N_SPATIAL  = 18
    _N_SCALAR   = 28
    _N_ACTIONS  = 6
    _MODEL_FILE = "model.pth"   # expected next to agent.py in submission zip

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self._model   = None
        self._device  = torch.device("cpu") if _TORCH_AVAILABLE else None
        self._fallback = _MinimalTacticalFallback(agent_id)
        self._use_fallback = True
        self._tracker = _AgentTracker(agent_id) if _AgentTracker is not None else None
        self._last_action: int | None = None

        if _TORCH_AVAILABLE:
            self._try_load_model()

    def reset(self) -> None:
        if self._tracker is not None:
            self._tracker.reset()
        self._last_action = None

    def _try_load_model(self) -> None:
        model_path = Path(__file__).parent / self._MODEL_FILE
        if not model_path.exists():
            return
        try:
            actor = _CNNActor(self._N_SPATIAL, self._N_SCALAR, self._N_ACTIONS)
            ckpt  = torch.load(str(model_path), map_location="cpu", weights_only=False)
            if _HAS_SHARED_MAPPO:
                from agent.mappo_agent.checkpoint_utils import load_actor_state_dict
                load_actor_state_dict(actor, ckpt, map_location="cpu")
            else:
                state = ckpt.get("actor_state_dict", ckpt.get("model_state_dict", ckpt))
                actor.load_state_dict(state, strict=True)
            actor.eval()
            self._model = actor
            self._use_fallback = False
        except Exception:
            self._model = None
            self._use_fallback = True

    def act(self, obs: dict) -> int:
        try:
            if self._use_fallback or self._model is None:
                return self._fallback.act(obs)
            return self._act_nn(obs)
        except Exception:
            return 0

    def _act_nn(self, obs: dict) -> int:
        if self._tracker is not None:
            self._tracker.sync_before_act(obs, self._last_action)
        spatial, scalar = _encode_obs_submission(obs, self.agent_id, self._tracker)
        sp_t  = torch.from_numpy(spatial).unsqueeze(0)   # (1, 18, 13, 13)
        sc_t  = torch.from_numpy(scalar).unsqueeze(0)    # (1, 28)

        with torch.inference_mode():
            logits = self._model(sp_t, sc_t)              # (1, 6)
        logits_np = logits.squeeze(0).numpy()             # (6,)

        action = _submission_apply_safety(logits_np, obs, self.agent_id)
        self._last_action = action
        return action
