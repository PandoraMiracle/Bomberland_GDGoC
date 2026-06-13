"""
vec_env.py — Custom Multiprocessing Vectorized Environment for Bomberland.

Runs environments and rule-based opponents in separate processes.
"""
from __future__ import annotations

import multiprocessing as mp
import traceback
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from engine.game import BomberEnv
from training.mappo.league_manager import LeagueManager
from training.mappo.reward_builder import RewardBuilder


def env_worker(remote: mp.connection.Connection, parent_remote: mp.connection.Connection, 
               agent_id: int, seed: int, max_steps: int, dense_coef: float):
    """
    Worker process loop. Runs a single BomberEnv, samples opponents, 
    computes rewards, and manages step execution.
    """
    parent_remote.close()
    
    env = BomberEnv(seed=seed, max_steps=max_steps)
    league = LeagueManager(seed=seed)
    rb = RewardBuilder(dense_reward_coef=dense_coef)
    
    opponents = []
    prev_obs = None

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                action = data
                # Collect opponent actions
                all_actions = [0] * 4
                all_actions[agent_id] = action
                for k, opp in enumerate(opponents):
                    opp_id = (agent_id + k + 1) % 4
                    try:
                        all_actions[opp_id] = opp.act(prev_obs)
                    except Exception:
                        all_actions[opp_id] = 0

                next_obs, terminated, truncated = env.step(all_actions)
                done = terminated or truncated

                # Compute dense reward
                dense_r = rb.compute_dense(prev_obs, next_obs, agent_id, action)
                if done:
                    final_ranks = getattr(env, "ranks", None)
                    if final_ranks is not None:
                        term_r = rb.compute_terminal(final_ranks, agent_id)
                    else:
                        players = np.asarray(next_obs.get("players", []), dtype=np.int32)
                        alive = int(players[agent_id, 2]) if len(players) > agent_id else 0
                        term_r = 1.0 if alive else -0.5
                    reward = dense_r + term_r
                else:
                    reward = dense_r

                prev_obs = next_obs
                remote.send((next_obs, float(reward), float(done)))

            elif cmd == "reset":
                # data = (seed, ckpt_paths)
                seed_val, ckpt_paths = data
                league._ckpt_paths = ckpt_paths
                obs = env.reset(seed=seed_val)
                opponents = league.sample_opponents(3, agent_id)
                prev_obs = obs
                remote.send(obs)

            elif cmd == "close":
                remote.close()
                break
            else:
                raise NotImplementedError(f"Got invalid command {cmd}")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Worker Error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise e


class VecEnv:
    """
    Main thread interface for multiple environment workers.
    """
    def __init__(self, num_envs: int, agent_id: int, seed: int, max_steps: int, dense_coef: float):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        
        self.processes = []
        for i in range(num_envs):
            worker_seed = seed + i
            p = mp.Process(
                target=env_worker, 
                args=(self.work_remotes[i], self.remotes[i], agent_id, worker_seed, max_steps, dense_coef),
                daemon=True
            )
            p.start()
            self.processes.append(p)
            
        for remote in self.work_remotes:
            remote.close()

    def step_async(self, actions: list[int]):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))

    def step_wait(self) -> tuple[list[dict], list[float], list[float]]:
        results = [remote.recv() for remote in self.remotes]
        next_obs_list, reward_list, done_list = zip(*results)
        return list(next_obs_list), list(reward_list), list(done_list)

    def step(self, actions: list[int]) -> tuple[list[dict], list[float], list[float]]:
        self.step_async(actions)
        return self.step_wait()

    def reset_idx(self, idx: int, seed: int, ckpt_paths: list[str]) -> dict:
        self.remotes[idx].send(("reset", (seed, ckpt_paths)))
        return self.remotes[idx].recv()

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for p in self.processes:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
