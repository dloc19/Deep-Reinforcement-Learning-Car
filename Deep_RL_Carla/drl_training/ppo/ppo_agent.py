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

        # Hai optimizer RIENG, khong phai mot. Ly do la §12 cua notebook IL: critic khoi
        # tao NGAU NHIEN, nen vai tram update dau no sinh advantage gan nhu nhieu trang.
        # Dung chung mot LR nghia la actor da duoc warm-start bang IL se bi chinh khoi
        # nhieu do voi cung mot toc do — tuc XOA trong so IL truoc khi critic kip hoc gi.
        # Tach ra cho phep: critic LR binh thuong (3e-4), actor LR nho (1e-5..3e-5), va
        # `critic_warmup_updates` dau tien DONG BANG han actor.
        self.actor_lr = config.get("actor_lr", config.get("learning_rate", 3e-5))
        self.critic_lr = config.get("critic_lr", config.get("learning_rate", 3e-4))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.critic_warmup_updates = int(config.get("critic_warmup_updates", 0))

        # Xem docstring `policy.backbone.freeze_batchnorm`. Phai goi SAU khi actor/critic da
        # .to(device) va TRUOC rollout dau tien.
        from policy.backbone import freeze_batchnorm
        self.frozen_bn = freeze_batchnorm(self.actor, self.critic)

        self.clip_range = config.get("clip_range", 0.2)
        # `None` = TAT clip value. Mac dinh cua Stable-Baselines3 (`clip_range_vf=None`)
        # cung la None, va ly do rat cu the: nguong nay o don vi TUYET DOI cua ham gia tri,
        # khong phai ti le. Reward cua env nay ~2-8.6 moi buoc, gamma 0.99 -> return co do
        # lon 200-800; kep buoc dich chuyen cua critic o +/-0.2 nghia la no can hang nghin
        # update moi bam kip. Do lai dung ham loss ben duoi voi return muc tieu 800, 50
        # update, critic_lr 3e-4:
        #     clip 0.2  -> V = 27.5  (3.4% muc tieu)
        #     clip 2.0  -> V = 127.3 (15.9%)
        #     clip 20.0 -> V = 318.6 (39.8%)
        #     None      -> V = 318.7 (39.8%)  <- tran that
        # Tuc clip 0.2 lam critic cham 12 lan. Trieu chung tren log that: value_loss dung im
        # hoac vot len (5696->9175->8741 tren runs/ppo_lane_keep, 3610->...->8175 o smoke
        # test) thay vi giam. Chi bat lai — voi gia tri cung do lon voi return — neu return
        # da duoc chuan hoa.
        self.value_clip_range = config.get("value_clip_range", None)
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

    def update(self, buffer, advantages, returns, freeze_actor=False):
        """`freeze_actor=True`: chi train critic trong update nay (giai doan warmup).

        Actor van chay forward de log approx_kl/entropy cho tien theo doi, nhung khong co
        gradient nao cham vao no. Day la khuyen nghi §12 cua notebook IL: de critic hoc
        xong ham gia tri quanh hanh vi IL TRUOC, roi moi cho phep policy dich chuyen.
        """
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_fraction": []}

        for _epoch in range(self.epochs):
            epoch_kls = []
            for batch in buffer.iter_minibatches(self.batch_size, advantages, returns):
                if freeze_actor:
                    with torch.no_grad():
                        new_log_probs, entropy = self.actor.evaluate_actions(
                            batch["seg"], batch["scalar"], batch["actions"])
                else:
                    new_log_probs, entropy = self.actor.evaluate_actions(
                        batch["seg"], batch["scalar"], batch["actions"])
                ratio = torch.exp(new_log_probs - batch["old_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch["advantages"]
                policy_loss = -torch.min(surr1, surr2).mean()

                values = self.critic(batch["seg"], batch["scalar"])
                if self.value_clip_range is None:
                    value_loss = 0.5 * F.mse_loss(values, batch["returns"])
                else:
                    values_clipped = batch["old_values"] + torch.clamp(
                        values - batch["old_values"], -self.value_clip_range, self.value_clip_range)
                    value_loss = 0.5 * torch.max(
                        F.mse_loss(values, batch["returns"], reduction="none"),
                        F.mse_loss(values_clipped, batch["returns"], reduction="none"),
                    ).mean()

                entropy_loss = -entropy.mean()

                self.critic_optimizer.zero_grad(set_to_none=True)
                if freeze_actor:
                    (self.value_coef * value_loss).backward()
                else:
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    (policy_loss + self.value_coef * value_loss
                     + self.entropy_coef * entropy_loss).backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = (batch["old_log_probs"] - new_log_probs).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                epoch_kls.append(approx_kl)
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.mean().item())
                stats["approx_kl"].append(approx_kl)
                stats["clip_fraction"].append(clip_fraction)

            if (not freeze_actor and self.target_kl is not None
                    and np.mean(epoch_kls) > 1.5 * self.target_kl):
                break  # policy moved too far this update — stop remaining epochs early

        return {key: float(np.mean(values)) if values else 0.0 for key, values in stats.items()}

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        # Checkpoint truoc ban tach optimizer chi co mot key "optimizer" gop ca actor lan
        # critic — khong the chia lai duoc, nen bo qua trang thai Adam (momentum) va chi
        # nap trong so mang. Mat vai chuc step de Adam ap lai moment, khong mat gi khac.
        if "actor_optimizer" in state:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        elif "optimizer" in state:
            print("[!] Checkpoint cu (mot optimizer gop). Da nap trong so actor/critic, bo "
                  "qua trang thai Adam.")
