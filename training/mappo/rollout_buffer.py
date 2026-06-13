"""
rollout_buffer.py — Pre-allocated rollout buffer with GAE computation.
"""

from __future__ import annotations
import numpy as np


class RolloutBuffer:
    """
    Pre-allocated circular storage for one PPO rollout.
    Stores transitions for a single agent across N_ENVS parallel environments.

    Layout: [rollout_length, num_envs, ...]
    """

    def __init__(
        self,
        rollout_length:  int,
        num_envs:        int,
        n_spatial:       int,
        n_scalar:        int,
        global_scalar_dim: int,
        grid_h:          int = 13,
        grid_w:          int = 13,
    ):
        self.rollout_length    = rollout_length
        self.num_envs          = num_envs
        T, E = rollout_length, num_envs

        # Observations
        self.spatials  = np.zeros((T, E, n_spatial, grid_h, grid_w), dtype=np.float32)
        self.scalars   = np.zeros((T, E, n_scalar),                  dtype=np.float32)
        # Global (critic)
        self.g_scalars = np.zeros((T, E, global_scalar_dim),         dtype=np.float32)
        # Transitions
        self.actions   = np.zeros((T, E),  dtype=np.int64)
        self.rewards   = np.zeros((T, E),  dtype=np.float32)
        self.dones     = np.zeros((T, E),  dtype=np.float32)
        self.values    = np.zeros((T, E),  dtype=np.float32)
        self.log_probs = np.zeros((T, E),  dtype=np.float32)
        self.action_masks = np.ones((T, E, 6), dtype=bool)
        # Computed after collection
        self.advantages = np.zeros((T, E), dtype=np.float32)
        self.returns    = np.zeros((T, E), dtype=np.float32)

        self._ptr = 0
        self._full = False

    def add(
        self,
        t:         int,
        env_idx:   int,
        spatial:   np.ndarray,
        scalar:    np.ndarray,
        g_scalar:  np.ndarray,
        action:    int,
        reward:    float,
        done:      float,
        value:     float,
        log_prob:  float,
        action_mask: np.ndarray,
    ) -> None:
        self.spatials[t,  env_idx] = spatial
        self.scalars[t,   env_idx] = scalar
        self.g_scalars[t, env_idx] = g_scalar
        self.actions[t,   env_idx] = action
        self.rewards[t,   env_idx] = reward
        self.dones[t,     env_idx] = done
        self.values[t,    env_idx] = value
        self.log_probs[t, env_idx] = log_prob
        self.action_masks[t, env_idx] = action_mask

    def compute_gae(
        self,
        last_values: np.ndarray,   # (num_envs,)
        gamma:       float,
        gae_lambda:  float,
    ) -> None:
        """Compute GAE advantages and returns in-place."""
        T, E = self.rollout_length, self.num_envs
        last_gae = np.zeros(E, dtype=np.float32)

        for t in reversed(range(T)):
            next_val = last_values if t == T - 1 else self.values[t + 1]
            next_non_term = 1.0 - self.dones[t]
            delta = (
                self.rewards[t]
                + gamma * next_val * next_non_term
                - self.values[t]
            )
            last_gae = delta + gamma * gae_lambda * next_non_term * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_flat(self) -> dict[str, np.ndarray]:
        """Flatten [T, E, ...] → [T*E, ...] for mini-batch sampling."""
        B = self.rollout_length * self.num_envs
        return {
            "spatials":   self.spatials.reshape(B, *self.spatials.shape[2:]),
            "scalars":    self.scalars.reshape(B,  *self.scalars.shape[2:]),
            "g_scalars":  self.g_scalars.reshape(B, *self.g_scalars.shape[2:]),
            "actions":    self.actions.reshape(B),
            "log_probs":  self.log_probs.reshape(B),
            "action_masks": self.action_masks.reshape(B, 6),
            "advantages": self.advantages.reshape(B),
            "returns":    self.returns.reshape(B),
            "values":     self.values.reshape(B),
        }
