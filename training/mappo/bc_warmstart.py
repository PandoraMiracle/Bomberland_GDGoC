"""
bc_warmstart.py — Behavior cloning warm-start from rule-based teachers.

Collects (observation → action) pairs from teacher agents then trains
the actor with cross-entropy loss.

Usage:
  python -m training.mappo.bc_warmstart [options]

Options:
  --teacher    tactical | genius | smarter (default: tactical)
  --episodes   N        number of episodes to collect (default: 500)
  --output     PATH     output checkpoint path (default: checkpoints/mappo/bc_warmstart.pth)
  --epochs     N        training epochs over collected data (default: 10)
  --lr         LR       learning rate (default: 1e-3)
  --batch-size N        (default: 256)
  --device     DEVICE   cpu | cuda | auto
  --seed       N        (default: 42)
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from engine.game import BomberEnv
from agent.mappo_agent.model import CNNActor
from agent.mappo_agent.encoder import encode_obs
from agent.mappo_agent.tracker import AgentTracker

TEACHER_PATHS = {
    "tactical": str(ROOT / "agent" / "tactical_rule_agent.py"),
    "genius":   str(ROOT / "agent" / "genius_rule_agent.py"),
    "smarter":  str(ROOT / "agent" / "smarter_rule_agent.py"),
}


def _load_teacher(name: str, agent_id: int):
    import importlib.util
    path = TEACHER_PATHS[name]
    spec = importlib.util.spec_from_file_location(f"_teacher_{name}", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for attr in dir(mod):
        cls = getattr(mod, attr)
        if isinstance(cls, type) and (attr == "Agent" or attr.endswith("Agent")):
            try:    return cls(agent_id)
            except: return cls()
    raise RuntimeError(f"No Agent class in {path}")


def collect_data(
    teacher_name: str,
    n_episodes:   int,
    seed:         int,
    agent_id:     int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect spatial/scalar/action triples from teacher self-play.
    Returns arrays of shape (N, 18, 13, 13), (N, 22), (N,).
    """
    sp_list, sc_list, act_list = [], [], []

    teachers = [_load_teacher(teacher_name, i) for i in range(4)]
    print(f"[BC] Collecting {n_episodes} episodes with '{teacher_name}' teacher...")

    for ep in range(n_episodes):
        env = BomberEnv(seed=seed + ep, max_steps=500)
        obs = env.reset(seed=seed + ep)
        tracker = AgentTracker(agent_id)
        tracker.reset()
        done = False
        while not done:
            actions = []
            for i in range(4):
                try:    a = teachers[i].act(obs)
                except: a = 0
                if 0 <= a <= 5:
                    actions.append(a)
                else:
                    actions.append(0)
                if i == agent_id:
                    sp, sc = encode_obs(obs, agent_id, tracker)
                    sp_list.append(sp)
                    sc_list.append(sc)
                    act_list.append(a)
            obs, term, trunc = env.step(actions)
            tracker.update(obs, actions[agent_id])
            done = term or trunc

        if (ep + 1) % 50 == 0:
            print(f"  episodes collected: {ep+1}/{n_episodes}, samples: {len(act_list)}")

    return (
        np.stack(sp_list,  axis=0).astype(np.float32),
        np.stack(sc_list,  axis=0).astype(np.float32),
        np.array(act_list, dtype=np.int64),
    )


def train_bc(
    spatials:   np.ndarray,
    scalars:    np.ndarray,
    actions:    np.ndarray,
    epochs:     int   = 10,
    batch_size: int   = 256,
    lr:         float = 1e-3,
    device_str: str   = "cpu",
    output:     str   = "checkpoints/mappo/bc_warmstart.pth",
) -> None:
    device = torch.device(device_str)

    ds     = TensorDataset(
        torch.from_numpy(spatials),
        torch.from_numpy(scalars),
        torch.from_numpy(actions),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    actor  = CNNActor().to(device)
    opt    = optim.Adam(actor.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    actor.train()
    print(f"[BC] Training actor: {len(actions)} samples, {epochs} epochs, device={device_str}")

    for epoch in range(epochs):
        total_loss = 0.0
        total_acc  = 0
        n          = 0
        for sp_b, sc_b, act_b in loader:
            sp_b  = sp_b.to(device)
            sc_b  = sc_b.to(device)
            act_b = act_b.to(device)

            logits = actor(sp_b, sc_b)
            loss   = loss_fn(logits, act_b)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            preds = logits.argmax(dim=-1)
            total_acc  += int((preds == act_b).sum().item())
            total_loss += loss.item() * len(act_b)
            n          += len(act_b)

        acc = total_acc / max(n, 1)
        print(f"  Epoch {epoch+1:3d}/{epochs} | loss={total_loss/n:.4f} | acc={acc:.3f}")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor_state_dict": actor.state_dict(), "bc_epochs": epochs}, output)
    print(f"[BC] Saved warm-start checkpoint → {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavior cloning warm-start")
    parser.add_argument("--teacher",    type=str,   default="tactical",
                        choices=list(TEACHER_PATHS.keys()))
    parser.add_argument("--episodes",   type=int,   default=500)
    parser.add_argument("--output",     type=str,   default=None)
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--device",     type=str,   default="auto")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"checkpoints/mappo/bc_warmstart_{args.teacher}.pth"

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    spatials, scalars, actions = collect_data(
        args.teacher, args.episodes, args.seed
    )
    train_bc(
        spatials, scalars, actions,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device_str=device_str, output=args.output,
    )


if __name__ == "__main__":
    main()
