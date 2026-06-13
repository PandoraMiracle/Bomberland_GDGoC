"""
test_smoke.py — Smoke tests: one full local match, tiny training loop.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.game import BomberEnv
from agent.mappo_agent.agent import Agent


class TestOneMatch:
    def test_one_match_no_crash(self):
        """Run one complete match with MAPPO agent (seat 0) vs 3 fallback agents."""
        env = BomberEnv(seed=123, max_steps=100)
        obs = env.reset(seed=123)

        agents = [Agent(agent_id=i) for i in range(4)]

        done = False
        steps = 0
        while not done and steps < 100:
            actions = []
            for i in range(4):
                a = agents[i].act(obs)
                assert 0 <= a <= 5, f"Agent {i} returned invalid action {a}"
                actions.append(a)
            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            steps += 1

        assert steps > 0, "Match should have run at least 1 step"
        # Verify final observation is well-formed
        assert "map" in obs
        assert "players" in obs
        assert "bombs" in obs
        print(f"\n[SmokeMatch] Completed {steps} steps, terminated={terminated}, truncated={truncated}")


class TestSmokePPO:
    """
    Minimal smoke test for the PPO training pipeline.
    Uses 2 envs, 32 steps, 1 PPO update — just verifies no crash.
    """

    def test_smoke_training(self, tmp_path):
        try:
            import torch
            import torch.optim as optim
        except ImportError:
            pytest.skip("PyTorch not available")

        sys.path.insert(0, str(ROOT))
        from agent.mappo_agent.model import ActorCritic, N_SPATIAL, N_SCALAR
        from agent.mappo_agent.encoder import encode_obs
        from agent.mappo_agent.safety import apply_safety

        N_ENVS      = 2
        ROLLOUT_LEN = 32
        GAMMA       = 0.99
        GAE_LAM     = 0.95
        CLIP_EPS    = 0.2
        ENT_COEF    = 0.01
        VF_COEF     = 0.5
        AGENT_ID    = 0
        GLOBAL_SC   = 32
        DEVICE      = torch.device("cpu")

        ac = ActorCritic(global_scalar_dim=GLOBAL_SC).to(DEVICE)
        opt_a = optim.Adam(ac.actor.parameters(),  lr=3e-4)
        opt_c = optim.Adam(ac.critic.parameters(), lr=5e-4)

        # Rollout buffers
        sp_buf  = np.zeros((ROLLOUT_LEN, N_ENVS, N_SPATIAL, 13, 13), dtype=np.float32)
        sc_buf  = np.zeros((ROLLOUT_LEN, N_ENVS, N_SCALAR), dtype=np.float32)
        act_buf = np.zeros((ROLLOUT_LEN, N_ENVS), dtype=np.int64)
        rew_buf = np.zeros((ROLLOUT_LEN, N_ENVS), dtype=np.float32)
        don_buf = np.zeros((ROLLOUT_LEN, N_ENVS), dtype=np.float32)
        val_buf = np.zeros((ROLLOUT_LEN, N_ENVS), dtype=np.float32)
        lp_buf  = np.zeros((ROLLOUT_LEN, N_ENVS), dtype=np.float32)

        envs = [BomberEnv(seed=i, max_steps=50) for i in range(N_ENVS)]
        obs_list = [env.reset(seed=i) for i, env in enumerate(envs)]

        # ── collect rollout ──────────────────────────────────────────────────
        for t in range(ROLLOUT_LEN):
            for e_i, (env, obs) in enumerate(zip(envs, obs_list)):
                sp, sc = encode_obs(obs, AGENT_ID)
                sp_t = torch.from_numpy(sp).unsqueeze(0).to(DEVICE)
                sc_t = torch.from_numpy(sc).unsqueeze(0).to(DEVICE)
                gsc  = torch.zeros(1, GLOBAL_SC).to(DEVICE)

                with torch.no_grad():
                    act, lp, _ = ac.act(sp_t, sc_t)
                    val = ac.critic(sp_t, gsc)

                action = int(act.item())
                actions = [0, 0, 0, 0]
                actions[AGENT_ID] = action

                next_obs, term, trunc = env.step(actions)
                done = float(term or trunc)
                reward = 1.0 - done  # trivial reward for smoke test

                sp_buf[t, e_i]  = sp
                sc_buf[t, e_i]  = sc
                act_buf[t, e_i] = action
                rew_buf[t, e_i] = reward
                don_buf[t, e_i] = done
                val_buf[t, e_i] = val.item()
                lp_buf[t, e_i]  = lp.item()

                if done:
                    obs_list[e_i] = env.reset()
                else:
                    obs_list[e_i] = next_obs

        # ── GAE ──────────────────────────────────────────────────────────────
        adv_buf = np.zeros_like(rew_buf)
        ret_buf = np.zeros_like(rew_buf)
        for e_i in range(N_ENVS):
            last_gae = 0.0
            last_val = val_buf[-1, e_i]
            for t in reversed(range(ROLLOUT_LEN)):
                next_val = last_val if t == ROLLOUT_LEN - 1 else val_buf[t + 1, e_i]
                delta = rew_buf[t, e_i] + GAMMA * next_val * (1 - don_buf[t, e_i]) - val_buf[t, e_i]
                last_gae = delta + GAMMA * GAE_LAM * (1 - don_buf[t, e_i]) * last_gae
                adv_buf[t, e_i] = last_gae
            ret_buf[:, e_i] = adv_buf[:, e_i] + val_buf[:, e_i]

        # ── PPO update (1 epoch) ─────────────────────────────────────────────
        B = ROLLOUT_LEN * N_ENVS
        sp_t  = torch.from_numpy(sp_buf.reshape(B, N_SPATIAL, 13, 13)).to(DEVICE)
        sc_t  = torch.from_numpy(sc_buf.reshape(B, N_SCALAR)).to(DEVICE)
        act_t = torch.from_numpy(act_buf.reshape(B)).to(DEVICE)
        adv_t = torch.from_numpy(adv_buf.reshape(B)).float().to(DEVICE)
        ret_t = torch.from_numpy(ret_buf.reshape(B)).float().to(DEVICE)
        olp_t = torch.from_numpy(lp_buf.reshape(B)).float().to(DEVICE)
        gsc_t = torch.zeros(B, GLOBAL_SC).to(DEVICE)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        lp_new, ent, val_new = ac.evaluate(sp_t, sc_t, sp_t, gsc_t, act_t)
        ratio = torch.exp(lp_new - olp_t)
        pg_loss = -torch.min(
            ratio * adv_t,
            torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t
        ).mean()
        vf_loss = ((val_new - ret_t) ** 2).mean()
        ent_loss = -ent.mean()
        loss = pg_loss + VF_COEF * vf_loss + ENT_COEF * ent_loss

        opt_a.zero_grad(); opt_c.zero_grad()
        loss.backward()
        opt_a.step(); opt_c.step()

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

        # ── save checkpoint ──────────────────────────────────────────────────
        ckpt_path = tmp_path / "smoke.pth"
        torch.save({
            "actor_state_dict":  ac.actor.state_dict(),
            "critic_state_dict": ac.critic.state_dict(),
            "update": 1,
        }, str(ckpt_path))
        assert ckpt_path.exists()

        print(f"\n[SmokePPO] loss={loss.item():.4f} | checkpoint saved to {ckpt_path}")
