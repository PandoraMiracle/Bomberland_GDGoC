import time
from engine.game import BomberEnv
from agent.mappo_agent.model import CNNActor
from agent.tactical_rule_agent import Agent as Tactical
from agent.smarter_rule_agent import Agent as Smarter
from agent.genius_rule_agent import Agent as Genius
import torch
import numpy as np

def profile():
    env = BomberEnv(seed=42)
    obs = env.reset(seed=42)
    
    tactical = Tactical(1)
    smarter = Smarter(2)
    genius = Genius(3)
    
    actor = CNNActor(18, 22, 6)
    
    t0 = time.time()
    for _ in range(100):
        # Time agent 1
        t_start = time.time()
        tactical.act(obs)
        tactical_t = time.time() - t_start
        
        # Time agent 2
        t_start = time.time()
        smarter.act(obs)
        smarter_t = time.time() - t_start
        
        # Time agent 3
        t_start = time.time()
        genius.act(obs)
        genius_t = time.time() - t_start
        
        # Time env
        t_start = time.time()
        obs, _, _ = env.step([0, 0, 0, 0])
        env_t = time.time() - t_start
        
    t_end = time.time()
    print(f"Total time for 100 steps: {t_end - t0:.3f} s")
    
    # Batched vs unbatched torch
    # ...

profile()
