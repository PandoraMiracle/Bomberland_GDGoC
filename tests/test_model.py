"""
test_model.py — Tests for CNNActor and CentralizedCritic.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.mappo_agent.model import (
    CNNActor, CentralizedCritic, ActorCritic,
    N_SPATIAL, N_SCALAR, N_ACTIONS, GRID_H, GRID_W
)
from training.mappo.critic_features import PRIVILEGED_SCALAR_DIM


class TestCNNActor:
    @pytest.fixture
    def actor(self):
        return CNNActor().eval()

    def test_output_shape_single(self, actor):
        sp = torch.zeros(1, N_SPATIAL, GRID_H, GRID_W)
        sc = torch.zeros(1, N_SCALAR)
        with torch.no_grad():
            logits = actor(sp, sc)
        assert logits.shape == (1, N_ACTIONS)

    def test_output_shape_batch(self, actor):
        sp = torch.zeros(8, N_SPATIAL, GRID_H, GRID_W)
        sc = torch.zeros(8, N_SCALAR)
        with torch.no_grad():
            logits = actor(sp, sc)
        assert logits.shape == (8, N_ACTIONS)

    def test_output_finite(self, actor):
        sp = torch.randn(4, N_SPATIAL, GRID_H, GRID_W)
        sc = torch.randn(4, N_SCALAR)
        with torch.no_grad():
            logits = actor(sp, sc)
        assert torch.all(torch.isfinite(logits))

    def test_inference_mode(self, actor):
        sp = torch.zeros(1, N_SPATIAL, GRID_H, GRID_W)
        sc = torch.zeros(1, N_SCALAR)
        with torch.inference_mode():
            logits = actor(sp, sc)
        assert logits.shape == (1, N_ACTIONS)

    def test_parameter_count_reasonable(self, actor):
        n_params = sum(p.numel() for p in actor.parameters())
        # Should be between 200K and 5M
        assert 100_000 < n_params < 5_000_000, f"Parameter count {n_params} out of expected range"


class TestCentralizedCritic:
    @pytest.fixture
    def critic(self):
        return CentralizedCritic(global_scalar_dim=PRIVILEGED_SCALAR_DIM).eval()

    def test_output_shape(self, critic):
        sp = torch.zeros(1, N_SPATIAL, GRID_H, GRID_W)
        sc = torch.zeros(1, PRIVILEGED_SCALAR_DIM)
        with torch.no_grad():
            val = critic(sp, sc)
        assert val.shape == (1, 1)

    def test_output_finite(self, critic):
        sp = torch.randn(4, N_SPATIAL, GRID_H, GRID_W)
        sc = torch.randn(4, PRIVILEGED_SCALAR_DIM)
        with torch.no_grad():
            val = critic(sp, sc)
        assert torch.all(torch.isfinite(val))


class TestActorCritic:
    @pytest.fixture
    def ac(self):
        return ActorCritic().eval()

    def test_act_returns_three_tensors(self, ac):
        sp = torch.zeros(2, N_SPATIAL, GRID_H, GRID_W)
        sc = torch.zeros(2, N_SCALAR)
        action, log_prob, entropy = ac.act(sp, sc)
        assert action.shape   == (2,)
        assert log_prob.shape == (2,)
        assert entropy.shape  == (2,)

    def test_actions_in_range(self, ac):
        sp = torch.zeros(32, N_SPATIAL, GRID_H, GRID_W)
        sc = torch.zeros(32, N_SCALAR)
        action, _, _ = ac.act(sp, sc)
        assert torch.all(action >= 0)
        assert torch.all(action <= 5)

    def test_evaluate_shapes(self, ac):
        B = 4
        sp = torch.zeros(B, N_SPATIAL, GRID_H, GRID_W)
        sc = torch.zeros(B, N_SCALAR)
        gsp = torch.zeros(B, N_SPATIAL, GRID_H, GRID_W)
        gsc = torch.zeros(B, PRIVILEGED_SCALAR_DIM)
        acts = torch.randint(0, 6, (B,))
        log_prob, entropy, value = ac.evaluate(sp, sc, gsp, gsc, acts)
        assert log_prob.shape == (B,)
        assert entropy.shape  == (B,)
        assert value.shape    == (B,)

    def test_checkpoint_save_load(self, tmp_path, ac):
        path = tmp_path / "test_ckpt.pth"
        torch.save({"actor_state_dict": ac.actor.state_dict()}, str(path))
        loaded = CNNActor()
        from agent.mappo_agent.checkpoint_utils import load_actor_state_dict
        load_actor_state_dict(loaded, str(path))
        loaded.eval()
        # Verify forward pass is identical
        sp = torch.randn(1, N_SPATIAL, GRID_H, GRID_W)
        sc = torch.randn(1, N_SCALAR)
        with torch.no_grad():
            out1 = ac.actor(sp, sc)
            out2 = loaded(sp, sc)
        assert torch.allclose(out1, out2)
