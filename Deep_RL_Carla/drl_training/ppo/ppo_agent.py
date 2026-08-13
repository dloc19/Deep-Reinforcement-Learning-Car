"""Clipped-surrogate PPO update (Schulman et al., 2017) over a `RolloutBuffer`.

Standard recipe (matches common reference implementations, e.g. CleanRL's
`ppo_continuous_action.py` / Stable-Baselines3's `PPO`): advantage normalization, clipped
policy surrogate, clipped value loss, entropy bonus, gradient clipping, and an early stop
within an update if approximate KL grows too large (protects the IL warm-start from being
destroyed by one overly-aggressive update early in fine-tuning).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOAgent:
    def __init__(self, actor, critic, config, device):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.device = device
        self.cfg = config

        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config.get("learning_rate", 3e-4))

        self.clip_range = config.get("clip_range", 0.2)
        self.value_clip_range = config.get("value_clip_range", 0.2)
        self.entropy_coef = config.get("entropy_coef", 0.0)
        self.value_coef = config.get("value_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.epochs = config.get("epochs", 10)
        self.batch_size = config.get("batch_size", 256)
        self.target_kl = config.get("target_kl", 0.02)

    @torch.no_grad()
    def act(self, seg, scalar, deterministic=False):
        """Single-observation inference (env interaction, not training). Returns numpy."""
        seg_t = torch.as_tensor(seg, device=self.device).unsqueeze(0)
        scalar_t = torch.as_tensor(scalar, device=self.device).unsqueeze(0)
        env_action, raw_action, log_prob = self.actor.act(seg_t, scalar_t, deterministic=deterministic)
        value = self.critic(seg_t, scalar_t)
        return (
            env_action.squeeze(0).cpu().numpy(),
            raw_action.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    @torch.no_grad()
    def select_action(self, seg, scalar, deterministic=False):
        """Single-observation, action-only inference — used by `evaluate.py`, which needs
        the same call shape across PPO and SAC (see `sac.sac_agent.SACAgent.select_action`).
        `train_ppo.py`'s rollout loop uses the richer `.act()` above instead, since it also
        needs `log_prob`/`value` for the GAE/PPO update."""
        seg_t = torch.as_tensor(seg, device=self.device).unsqueeze(0)
        scalar_t = torch.as_tensor(scalar, device=self.device).unsqueeze(0)
        action = self.actor.select_action(seg_t, scalar_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def value_of(self, seg, scalar):
        seg_t = torch.as_tensor(seg, device=self.device).unsqueeze(0)
        scalar_t = torch.as_tensor(scalar, device=self.device).unsqueeze(0)
        return float(self.critic(seg_t, scalar_t).item())

    def update(self, buffer, advantages, returns):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_fraction": []}

        for _epoch in range(self.epochs):
            epoch_kls = []
            for batch in buffer.iter_minibatches(self.batch_size, advantages, returns):
                new_log_probs, entropy = self.actor.evaluate_actions(
                    batch["seg"], batch["scalar"], batch["actions"])
                ratio = torch.exp(new_log_probs - batch["old_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch["advantages"]
                policy_loss = -torch.min(surr1, surr2).mean()

                values = self.critic(batch["seg"], batch["scalar"])
                values_clipped = batch["old_values"] + torch.clamp(
                    values - batch["old_values"], -self.value_clip_range, self.value_clip_range)
                value_loss = 0.5 * torch.max(
                    F.mse_loss(values, batch["returns"], reduction="none"),
                    F.mse_loss(values_clipped, batch["returns"], reduction="none"),
                ).mean()

                entropy_loss = -entropy.mean()
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (batch["old_log_probs"] - new_log_probs).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                epoch_kls.append(approx_kl)
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.mean().item())
                stats["approx_kl"].append(approx_kl)
                stats["clip_fraction"].append(clip_fraction)

            if self.target_kl is not None and np.mean(epoch_kls) > 1.5 * self.target_kl:
                break  # policy moved too far this update — stop remaining epochs early

        return {key: float(np.mean(values)) if values else 0.0 for key, values in stats.items()}

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
