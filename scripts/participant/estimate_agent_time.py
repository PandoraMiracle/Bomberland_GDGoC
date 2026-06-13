import os

# ── Sandbox simulation ────────────────────────────────────────────────────────
# The real evaluator enforces single-threaded execution on every agent worker
# to prevent CPU saturation attacks and to ensure a fair, consistent timeout.
# We must set these BEFORE importing numpy, torch, or any C-extension, because
# those libraries read the env-vars only once at import time.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import sys
import time
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from engine.game import BomberEnv
from scripts.participant.run_local_match import make_agents

def run_timing_benchmark(agent_path, opponents, num_matches=10, max_steps=500, seed=None):
    # Combine target agent and opponents
    all_agent_paths = [agent_path] + opponents
    
    # Check if exactly 4 agents
    if len(all_agent_paths) != 4:
        raise ValueError(f"Expected exactly 4 agents total (1 target + 3 opponents), got {len(all_agent_paths)}")
        
    print(f"Loading Target Agent: {agent_path}")
    print(f"Loading Opponents: {opponents}")
    
    env = BomberEnv(max_steps=max_steps, seed=seed)
    agents, names = make_agents(all_agent_paths, seed)
    
    target_name = names[0]
    
    print(f"\n--- Starting benchmark for {target_name} ---")
    
    global_total_ms = 0.0
    global_total_steps = 0
    global_max_ms = 0.0
    
    for episode in range(num_matches):
        episode_seed = None if seed is None else seed + episode
        obs = env.reset(seed=episode_seed)
        done = False
        step = 0
        
        match_total_ms = 0.0
        match_steps = 0
        
        while not done and step < max_steps:
            actions = []
            
            # Timing the target agent (Player 0)
            target_agent = agents[0]
            if env.players[0].alive:
                start_time = time.perf_counter()
                try:
                    action = target_agent.act(obs)
                except Exception as e:
                    print(f"Agent failed to act at step {step}: {e}")
                    action = 0
                end_time = time.perf_counter()
                
                elapsed_ms = (end_time - start_time) * 1000.0
                match_total_ms += elapsed_ms
                match_steps += 1
                
                if elapsed_ms > global_max_ms:
                    global_max_ms = elapsed_ms
            else:
                action = 0
            
            actions.append(action)
            
            # Other agents act normally without timing
            for i in range(1, 4):
                if env.players[i].alive:
                    try:
                        act_i = agents[i].act(obs)
                    except Exception:
                        act_i = 0
                else:
                    act_i = 0
                actions.append(act_i)
                
            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            step += 1
            
        if match_steps > 0:
            avg_ms = match_total_ms / match_steps
            print(f"Match {episode + 1}/{num_matches}: Alive for {match_steps} steps | Avg Time: {avg_ms:.2f} ms/step")
        else:
            print(f"Match {episode + 1}/{num_matches}: Died immediately or took 0 steps.")
            
        global_total_ms += match_total_ms
        global_total_steps += match_steps

    print("\n" + "="*40)
    print("           BENCHMARK SUMMARY            ")
    print("="*40)
    if global_total_steps > 0:
        overall_avg = global_total_ms / global_total_steps
        print(f"Target Agent       : {target_name}")
        print(f"Total Matches      : {num_matches}")
        print(f"Total Active Steps : {global_total_steps}")
        print(f"Max Spike Time     : {global_max_ms:.2f} ms")
        print(f"Global Average Time: {overall_avg:.2f} ms/step")
        
        if overall_avg >= 100.0:
            print("\nWARNING: Your average time is over the 100ms server limit! You will likely be disqualified or timeout during real matches.")
        elif global_max_ms >= 100.0:
            print("\nCAUTION: Your average is fine, but you have spikes over 100ms. Make sure this doesn't happen frequently.")
        else:
            print("\nSTATUS: Safe! Your agent is well within the 100ms timeout limit.")
    else:
        print("Agent did not survive long enough to record any steps.")
    print("="*40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate the average inference time of an agent.")
    parser.add_argument("agent", type=str, help="Path to the target agent (e.g., submissions/team_a/agent.py)")
    parser.add_argument("--opponents", nargs="+", default=["None", "None", "None"], help="Paths to 3 opponent agents. Use 'None' for random baselines.")
    parser.add_argument("--num_matches", type=int, default=10, help="Number of matches to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    
    # Ensure exactly 3 opponents are passed. If the user provided fewer, pad with 'None'
    opponents = args.opponents
    while len(opponents) < 3:
        opponents.append("None")
    
    run_timing_benchmark(
        agent_path=args.agent,
        opponents=opponents[:3],
        num_matches=args.num_matches,
        seed=args.seed
    )
