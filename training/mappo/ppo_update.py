"""
ppo_update.py — PPO clipped surrogate loss update step.
"""

from __future__ import annotations
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def ppo_update(
    actor:       nn.Module,
    critic:      nn.Module,
    opt_actor:   optim.Optimizer,
    opt_critic:  optim.Optimizer,
    flat_data:   dict[str, np.ndarray],
    clip_eps:    float   = 0.15,
    ppo_epochs:  int     = 4,
    mini_batch:  int     = 256,
    entropy_coef: float  = 0.01,
    value_coef:  float   = 1.0,
    max_grad:    float   = 0.5,
    device:      str     = "cpu",
) -> dict[str, Any]:
    """
    Run `ppo_epochs` epochs of clipped PPO over the flat rollout data.

    Parameters
    ----------
    flat_data : output of RolloutBuffer.get_flat()

    Returns
    -------
    dict with keys:
        actor_loss, critic_loss, entropy, approx_kl,
        clip_fraction, explained_variance
    """
    dev = torch.device(device)

    # ── convert to tensors ────────────────────────────────────────────────────
    sp      = torch.from_numpy(flat_data["spatials"]).to(dev)
    sc      = torch.from_numpy(flat_data["scalars"]).to(dev)
    gsc     = torch.from_numpy(flat_data["g_scalars"]).to(dev)
    actions = torch.from_numpy(flat_data["actions"]).long().to(dev)
    old_lp  = torch.from_numpy(flat_data["log_probs"]).to(dev)
    returns = torch.from_numpy(flat_data["returns"]).to(dev)
    masks   = torch.from_numpy(flat_data["action_masks"]).to(dev)

    # Compute advantage stats before normalize
    adv = torch.from_numpy(flat_data["advantages"]).to(dev)
    adv_mean = adv.mean().item()
    adv_std = adv.std().item()

    # Normalise advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    adv_norm_mean = adv.mean().item()
    adv_norm_std = adv.std().item()

    B = sp.shape[0]

    # Accumulators
    total_actor_loss  = 0.0
    total_critic_loss = 0.0
    total_entropy     = 0.0
    total_kl          = 0.0
    total_clip_frac   = 0.0
    n_updates         = 0

    for _ in range(ppo_epochs):
        # Random mini-batch indices
        indices = np.random.permutation(B)
        for start in range(0, B, mini_batch):
            idx = torch.from_numpy(indices[start: start + mini_batch]).to(dev)

            mb_sp  = sp[idx];   mb_sc  = sc[idx];   mb_gsc = gsc[idx]
            mb_act = actions[idx]
            mb_olp = old_lp[idx]
            mb_adv = adv[idx]
            mb_ret = returns[idx]
            mb_msk = masks[idx]

            # ── actor forward ─────────────────────────────────────────────────
            logits = actor(mb_sp, mb_sc)
            logits = logits.masked_fill(~mb_msk, -1e9)
            dist   = torch.distributions.Categorical(logits=logits)
            new_lp = dist.log_prob(mb_act)
            entropy = dist.entropy().mean()

            ratio      = torch.exp(new_lp - mb_olp)
            pg_loss1   = -ratio * mb_adv
            pg_loss2   = -torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            actor_loss = torch.max(pg_loss1, pg_loss2).mean()

            clip_frac  = ((ratio - 1.0).abs() > clip_eps).float().mean()
            approx_kl  = ((mb_olp - new_lp).abs()).mean()

            # ── critic forward ────────────────────────────────────────────────
            value = critic(mb_sp, mb_gsc).squeeze(-1)
            critic_loss = ((value - mb_ret) ** 2).mean()

            # ── combined loss ─────────────────────────────────────────────────
            loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

            opt_actor.zero_grad(set_to_none=True)
            opt_critic.zero_grad(set_to_none=True)
            loss.backward()

            nn.utils.clip_grad_norm_(actor.parameters(),  max_grad)
            nn.utils.clip_grad_norm_(critic.parameters(), max_grad)

            opt_actor.step()
            opt_critic.step()

            total_actor_loss  += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy     += entropy.item()
            total_kl          += approx_kl.item()
            total_clip_frac   += clip_frac.item()
            n_updates         += 1

    n = max(n_updates, 1)

    # Explained variance
    var_y   = float(np.var(flat_data["returns"]))
    var_res = float(np.var(flat_data["returns"] - flat_data["values"]))
    ev = 1.0 - var_res / (var_y + 1e-8)

    return {
        "actor_loss":        total_actor_loss  / n,
        "critic_loss":       total_critic_loss / n,
        "entropy":           total_entropy     / n,
        "approx_kl":         total_kl          / n,
        "clip_fraction":     total_clip_frac   / n,
        "explained_variance": ev,
        "advantage_mean":    adv_mean,
        "advantage_std":     adv_std,
        "advantage_norm_mean": adv_norm_mean,
        "advantage_norm_std":  adv_norm_std,
        "return_mean":       float(np.mean(flat_data["returns"])),
        "return_std":        float(np.std(flat_data["returns"])),
        "return_min":        float(np.min(flat_data["returns"])),
        "return_max":        float(np.max(flat_data["returns"])),
        "value_mean":        float(np.mean(flat_data["values"])),
        "value_std":         float(np.std(flat_data["values"])),
    }
