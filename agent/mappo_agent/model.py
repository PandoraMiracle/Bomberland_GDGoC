"""
model.py — CNN Actor and Centralized Critic for MAPPO.

Architecture (Actor):
  Conv(18→64, 3×3, pad=1) → ReLU
  ResBlock(64) × 2
  Conv(64→96, 3×3, pad=1) → ReLU → GlobalAvgPool  → (96,)
  ScalarMLP: 22 → 64 → 64
  FusionMLP: 160 → 128 → 64 → PolicyHead: 64 → 6 logits

Architecture (CentralizedCritic):
  Same CNN trunk on global spatial (18, 13, 13)
  GlobalScalarMLP: global_scalar_dim → 128 → 64
  FusionMLP: 160 → 128 → 1 value scalar

Both networks use LayerNorm instead of BatchNorm to be stable with
small batches and single-env rollouts.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# Input dimensions (must match encoder.py)
N_SPATIAL    = 18
N_SCALAR     = 22
N_ACTIONS    = 6
GRID_H       = 13
GRID_W       = 13


# ── building blocks ──────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """2-conv residual block with LayerNorm and ReLU."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.ln1   = nn.GroupNorm(8, channels)   # GroupNorm works for any batch size
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.ln2   = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.ln1(self.conv1(x)))
        x = self.ln2(self.conv2(x))
        return F.relu(x + residual)


class CNNTrunk(nn.Module):
    """Shared CNN trunk used by both actor and critic."""

    def __init__(self, in_channels: int = N_SPATIAL):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.res1  = ResBlock(64)
        self.res2  = ResBlock(64)
        self.conv2 = nn.Conv2d(64, 96, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = self.res2(x)
        x = F.relu(self.conv2(x))
        # Global average pooling → (B, 96)
        return x.mean(dim=(-2, -1))


# ── actor ────────────────────────────────────────────────────────────────────

class CNNActor(nn.Module):
    """
    Policy network for one agent.

    Inputs:
        spatial : (B, 18, 13, 13) float32
        scalar  : (B, 22)          float32

    Output:
        logits  : (B, 6)           float32  (unnormalized log-probs)
    """

    def __init__(
        self,
        n_spatial: int = N_SPATIAL,
        n_scalar:  int = N_SCALAR,
        n_actions: int = N_ACTIONS,
    ):
        super().__init__()
        self.cnn_trunk = CNNTrunk(n_spatial)     # → (B, 96)

        self.scalar_mlp = nn.Sequential(
            nn.Linear(n_scalar, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(96 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(64, n_actions)

        # Orthogonal init for better early training stability
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Policy head: small init to encourage uniform distribution at start
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)

    def forward(
        self,
        spatial: torch.Tensor,
        scalar:  torch.Tensor,
    ) -> torch.Tensor:
        cnn_feat    = self.cnn_trunk(spatial)          # (B, 96)
        scalar_feat = self.scalar_mlp(scalar)          # (B, 64)
        fused       = self.fusion_mlp(
            torch.cat([cnn_feat, scalar_feat], dim=-1) # (B, 160)
        )                                              # (B, 64)
        return self.policy_head(fused)                 # (B, 6)


# ── centralized critic ───────────────────────────────────────────────────────

class CentralizedCritic(nn.Module):
    """
    Value network with access to privileged global state (training only).

    Inputs:
        global_spatial : (B, 18, 13, 13) — same channels as actor (full obs)
        global_scalar  : (B, global_scalar_dim)

    Output:
        value : (B, 1)
    """

    def __init__(
        self,
        n_spatial:           int = N_SPATIAL,
        global_scalar_dim:   int = 32,
    ):
        super().__init__()
        self.cnn_trunk = CNNTrunk(n_spatial)     # → (B, 96)

        self.global_scalar_mlp = nn.Sequential(
            nn.Linear(global_scalar_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(96 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.value_head = nn.Linear(64, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(
        self,
        global_spatial: torch.Tensor,
        global_scalar:  torch.Tensor,
    ) -> torch.Tensor:
        cnn_feat    = self.cnn_trunk(global_spatial)
        scalar_feat = self.global_scalar_mlp(global_scalar)
        fused       = self.fusion_mlp(
            torch.cat([cnn_feat, scalar_feat], dim=-1)
        )
        return self.value_head(fused)   # (B, 1)


# ── convenience wrapper ──────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Convenience wrapper holding actor + critic together.
    Used during training; only the actor is exported to the submission agent.
    """

    def __init__(
        self,
        n_spatial:         int = N_SPATIAL,
        n_scalar:          int = N_SCALAR,
        n_actions:         int = N_ACTIONS,
        global_scalar_dim: int = 32,
    ):
        super().__init__()
        self.actor  = CNNActor(n_spatial, n_scalar, n_actions)
        self.critic = CentralizedCritic(n_spatial, global_scalar_dim)

    def act(
        self,
        spatial: torch.Tensor,
        scalar:  torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (action, log_prob, entropy) for rollout collection.
        Does NOT compute value (call critic separately).
        """
        logits = self.actor(spatial, scalar)
        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate(
        self,
        spatial:        torch.Tensor,
        scalar:         torch.Tensor,
        global_spatial: torch.Tensor,
        global_scalar:  torch.Tensor,
        actions:        torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (log_prob, entropy, value) for PPO update step.
        """
        logits = self.actor(spatial, scalar)
        dist   = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        value    = self.critic(global_spatial, global_scalar).squeeze(-1)
        return log_prob, entropy, value
