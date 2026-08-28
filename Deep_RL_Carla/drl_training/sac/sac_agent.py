"""SAC update rule (Haarnoja et al., 2018): twin-Q critics with Polyak-averaged target
networks, reparameterized policy gradient, automatic temperature (entropy coefficient)
tuning. Matches the standard reference recipe (e.g. CleanRL's `sac_continuous_action.py`).
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class SACAgent(object):
    def __init__(self, actor, critic, config, device):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.device = device
        self.cfg = config
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.get("actor_lr", 3e-4))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.get("critic_lr", 3e-4))

        # Auto temperature tuning (SAC v2, Haarnoja et al. "Soft Actor-Critic Algorithms and
        # Applications"): target entropy defaults to -action_dim, a heuristic that works well
        # across the continuous-control literature and needs no environment-specific tuning.
        target_entropy = config.get("target_entropy")
        self.target_entropy = float(target_entropy) if target_entropy is not None else -2.0
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.get("alpha_lr", 3e-4))

        # Xem docstring `policy.backbone.freeze_batchnorm`. SAC can dieu nay khong kem PPO:
        # `select_action()` chay tren batch = 1 con `update()` chay tren batch 128, va target
        # critic thi duoc cap nhat bang Polyak — de thong ke BatchNorm troi theo batch size
        # se lam target va online critic lech nhau mot cach khong the truy vet.
        from policy.backbone import freeze_batchnorm
        self.frozen_bn = freeze_batchnorm(self.actor, self.critic)
        if getattr(self, "critic_target", None) is not None:
            self.frozen_bn += freeze_batchnorm(self.critic_target)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, seg, scalar, deterministic=False):
        """Single-observation inference — used both by `train_sac.py`'s rollout loop and
        `evaluate.py`. Unlike PPO, SAC doesn't need `log_prob`/value alongside the acting
        call: the update step recomputes both fresh from replay-buffer samples, fully
        decoupled from how the action was originally chosen during collection."""
        seg_t = torch.as_tensor(seg, device=self.device).unsqueeze(0)
        scalar_t = torch.as_tensor(scalar, device=self.device).unsqueeze(0)
        action = self.actor.select_action(seg_t, scalar_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch):
        seg, scalar, action = batch["seg"], batch["scalar"], batch["action"]
        reward, done = batch["reward"], batch["done"]
        next_seg, next_scalar = batch["next_seg"], batch["next_scalar"]

        # --- critic ---
        with torch.no_grad():
            next_action, next_log_prob, _mean = self.actor.sample(next_seg, next_scalar)
            target_q1, target_q2 = self.critic_target(next_seg, next_scalar, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            y = reward + self.gamma * (1.0 - done) * target_q

        q1, q2 = self.critic(seg, scalar, action)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # --- actor (reparameterized policy gradient) ---
        new_action, log_prob, _mean = self.actor.sample(seg, scalar)
        q1_pi, q2_pi = self.critic(seg, scalar, new_action)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_prob - min_q_pi).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # --- temperature ---
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- Polyak-averaged target critic ---
        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.mul_(1.0 - self.tau).add_(param, alpha=self.tau)

        return {
            "critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(), "alpha": float(self.alpha.item()),
            "mean_q": float(min_q_pi.mean().item()), "entropy": float(-log_prob.mean().item()),
        }

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        if "critic_target" in state:
            self.critic_target.load_state_dict(state["critic_target"])
        else:
            self.critic_target = copy.deepcopy(self.critic)
            for param in self.critic_target.parameters():
                param.requires_grad = False
        if "actor_optimizer" in state:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        if "critic_optimizer" in state:
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if "log_alpha" in state:
            with torch.no_grad():
                self.log_alpha.copy_(state["log_alpha"].to(self.device))
        if "alpha_optimizer" in state:
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
