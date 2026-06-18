"""
test_checkpoint_compat.py — Actor checkpoint migration for scalar dim changes.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.mappo_agent.model import CNNActor, N_SCALAR
from agent.mappo_agent.encoder import encode_obs
from agent.mappo_agent.checkpoint_utils import (
    adapt_actor_scalar_weights,
    load_actor_state_dict,
)
from agent.mappo_agent.safety import legal_action_mask


class TestCheckpointCompat:
    def test_adapt_pads_old_scalar_weights(self):
        actor = CNNActor()
        old_state = actor.state_dict()
        old_state["scalar_mlp.0.weight"] = old_state["scalar_mlp.0.weight"][:, :22].clone()

        adapted = adapt_actor_scalar_weights(old_state, N_SCALAR)
        assert adapted["scalar_mlp.0.weight"].shape == (64, N_SCALAR)
        assert torch.allclose(adapted["scalar_mlp.0.weight"][:, :22], old_state["scalar_mlp.0.weight"])
        assert torch.all(adapted["scalar_mlp.0.weight"][:, 22:] == 0)

    def test_load_22_dim_checkpoint_into_28_dim_actor(self):
        new_actor = CNNActor()
        ckpt = {"actor_state_dict": new_actor.state_dict()}
        ckpt["actor_state_dict"]["scalar_mlp.0.weight"] = (
            ckpt["actor_state_dict"]["scalar_mlp.0.weight"][:, :22].clone()
        )

        loaded = CNNActor()
        load_actor_state_dict(loaded, ckpt)

        sp = torch.randn(1, 18, 13, 13)
        sc_old = torch.randn(1, 22)
        sc_new = torch.zeros(1, N_SCALAR)
        sc_new[:, :22] = sc_old

        with torch.no_grad():
            out_padded = loaded(sp, sc_new)
            ref = CNNActor()
            ref.load_state_dict(new_actor.state_dict())
            ref.scalar_mlp[0].weight.data[:, 22:] = 0
            out_ref = ref(sp, sc_new)
        assert torch.allclose(out_padded, out_ref, atol=1e-5)

    def test_encoder_legal_features_match_safety_mask(self):
        from engine.game import BomberEnv

        env = BomberEnv(seed=7)
        obs = env.reset(seed=7)
        for _ in range(15):
            _, scalar = encode_obs(obs, agent_id=0)
            legal = legal_action_mask(obs, 0).astype(np.float32)
            np.testing.assert_array_equal(
                scalar[22:28], legal,
                err_msg="encoder legal scalars must match legal_action_mask",
            )
            obs, _, term = env.step([1, 0, 0, 0])
            if term:
                break
