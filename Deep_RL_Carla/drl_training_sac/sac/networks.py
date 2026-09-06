"""SAC actor (squashed Gaussian, reparameterized) and twin Q-critics (Haarnoja et al. 2018,
"Soft Actor-Critic Algorithms and Applications").

Reuses `policy.backbone.PolicyBackbone` + `policy.backbone.build_trunk_head` — the exact
same building blocks as PPO's `GaussianActor`/`ValueCritic` (`policy/actor_critic.py`) — so
warm-starting from the IL checkpoint uses the same `policy/il_compat.py` helpers on both
algorithms instead of two independently-maintained remap implementations.

Structural difference from PPO's actor (both are standard, algorithm-appropriate choices,
not one being "more correct" than the other):
* PPO: state-*independent* `log_std` (one learned vector, shared across all observations) —
  standard for PPO-continuous (OpenAI baselines, CleanRL).
* SAC: state-*dependent* `log_std` (its own small head) — SAC's canonical design, and part
  of why SAC needs the tanh-squash log-prob correction below (the action is always
  `tanh(sample)`, whereas PPO here samples-then-clips — see `GaussianActor.act` docstring
  for why PPO's simpler clip is an accepted approximation there).
"""

import torch
import torch.nn as nn

from policy.backbone import NUM_CLASSES, PolicyBackbone, build_trunk_head
from policy.il_compat import (
    load_matching, remap_backbone_state_dict, remap_final_layer_state_dict,
    remap_trunk_head_state_dict,
)

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
_EPS = 1e-6


class GaussianPolicy(nn.Module):
    def __init__(self, num_scalar_features, num_classes=NUM_CLASSES, log_std_init=-1.2):
        super(GaussianPolicy, self).__init__()
        self.backbone = PolicyBackbone(num_scalar_features, num_classes)
        self.trunk_head = build_trunk_head(self.backbone.out_features)
        self.mean_head = nn.Linear(32, 2)
        self.log_std_head = nn.Linear(32, 2)
        if log_std_init is None:
            log_std_init = -2.5
        # Small weights + constant bias so the log_std head starts near a fixed value
        # (~log_std_init) regardless of input, independent of the (warm-started) trunk —
        # keeps the actor's initial action noise modest instead of whatever a randomly
        # initialized head would produce, so early SAC rollouts stay close to IL behaviour.
        nn.init.uniform_(self.log_std_head.weight, -1e-3, 1e-3)
        # Bias THEO TUNG CHIEU, khong phai mot vo huong dung chung.
        #
        # `log_std_head` la Linear(32, 2) — hai chieu rieng biet [steer, longitudinal] — va
        # hai chieu do khong cung thang chut nao. `action_std` do tren tap train IL la
        # (0.078, 0.306): lech chuan cua lenh lai nho gap 4 lan lenh ga. Dat mot vo huong
        # chung (vd trung binh log = -1.866 -> std 0.155) lam nhieu lai LON GAP DOI muc can
        # va nhieu ga chi con MOT NUA.
        #
        # Do duoc truc tiep tren runs/sac_v2_smoke voi bias vo huong 0.155: lenh lai dien
        # hinh cua IL chi 0.005-0.03, nen nhieu 0.155 lon gap 5-30 lan tin hieu; xe lang
        # ngay va chet sau ~48 buoc (warm-start IL binh thuong di duoc 176-300 buoc), lech
        # lan 0.889 m, 85% episode va cham, va mean_ep_reward xau dan -33 -> -107.
        # PPO khong dinh loi nay vi `GaussianActor.log_std` von la vector 2 chieu.
        if isinstance(log_std_init, (list, tuple)):
            if len(log_std_init) != 2:
                raise ValueError("log_std_init dang danh sach phai co dung 2 phan tu "
                                 "[steer, longitudinal], nhan duoc %r" % (log_std_init,))
            with torch.no_grad():
                self.log_std_head.bias.copy_(torch.tensor([float(v) for v in log_std_init]))
        else:
            nn.init.constant_(self.log_std_head.bias, float(log_std_init))

    def forward(self, seg_map, scalar_features):
        feats = self.trunk_head(self.backbone(seg_map, scalar_features))
        mean = self.mean_head(feats)
        log_std = self.log_std_head(feats).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, seg_map, scalar_features):
        """Reparameterized tanh-squashed sample + corrected log-prob (SAC, Haarnoja et al.
        2018 appendix C — the `-log(1 - tanh(u)^2)` term accounts for the change of
        variables introduced by squashing through tanh).

        Returns (action, log_prob, mean_action) — `action`/`mean_action` are already in
        [-1, 1] (tanh-squashed), no separate clipping needed before `VehicleControl`.
        """
        mean, log_std = self.forward(seg_map, scalar_features)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        pre_tanh = normal.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = normal.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + _EPS)
        log_prob = log_prob.sum(-1)
        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action

    @torch.no_grad()
    def act(self, seg_map, scalar_features, deterministic=False):
        action, log_prob, mean_action = self.sample(seg_map, scalar_features)
        return (mean_action if deterministic else action), log_prob

    @torch.no_grad()
    def select_action(self, seg_map, scalar_features, deterministic=False):
        """Algorithm-agnostic single-return convenience used by `evaluate.py` (mirrors
        `policy.actor_critic.GaussianActor.select_action`)."""
        action, _log_prob = self.act(seg_map, scalar_features, deterministic)
        return action


class QNetwork(nn.Module):
    """State-action value network. No IL equivalent (IL never conditioned on an action) —
    always freshly initialized, never warm-started, consistent with standard BC+RL practice
    (e.g. AWAC, all warm-start-actor-only recipes)."""

    def __init__(self, num_scalar_features, action_dim=2, num_classes=NUM_CLASSES,
                 action_emb_dim=32):
        super(QNetwork, self).__init__()
        self.backbone = PolicyBackbone(num_scalar_features, num_classes)
        # Hanh dong duoc CHIEU LEN mot khong gian rong truoc khi ghep voi dac trung trang
        # thai, thay vi noi thang 2 chieu tho vao vector 96 chieu.
        #
        # Ban truoc: `Linear(96 + 2, 64)` — hanh dong chi la 2 trong 98 chieu dau vao, tuc
        # 2% be rong. Dac trung trang thai vua dong hon 48 lan vua co suc du doan return cao
        # hon han, nen gradient descent fit trang thai truoc va phan dong gop cua hanh dong
        # khong bao gio lon len. Do duoc tren runs/sac_a, sac_b, sac_c: quet lenh lai toan
        # dai -1..+1 chi lam Q doi 0.2-1.7% so voi bien thien giua cac trang thai — critic
        # hoc duoc Q(s,a) ~ V(s), va actor (chi hoc qua dQ/da) khong co gi de hoc.
        #
        # 32 chieu cho hanh dong dua ti le len 25% (32/128), du de mang khong the bo qua no.
        self.action_emb = nn.Sequential(nn.Linear(action_dim, action_emb_dim), nn.ELU())
        self.head = nn.Sequential(
            nn.Linear(self.backbone.out_features + action_emb_dim, 64), nn.ELU(),
            nn.Linear(64, 32), nn.ELU(),
            nn.Linear(32, 1),
        )

    def forward(self, seg_map, scalar_features, action):
        feats = self.backbone(seg_map, scalar_features)
        return self.head(torch.cat([feats, self.action_emb(action)], dim=1)).squeeze(-1)


class TwinQNetwork(nn.Module):
    """Two independent `QNetwork`s — SAC takes the min of both to fight overestimation
    bias (Fujimoto et al., "Addressing Function Approximation Error", the same trick TD3
    introduced and SAC adopted)."""

    def __init__(self, num_scalar_features, action_dim=2, num_classes=NUM_CLASSES):
        super(TwinQNetwork, self).__init__()
        self.q1 = QNetwork(num_scalar_features, action_dim, num_classes)
        self.q2 = QNetwork(num_scalar_features, action_dim, num_classes)

    def forward(self, seg_map, scalar_features, action):
        return self.q1(seg_map, scalar_features, action), self.q2(seg_map, scalar_features, action)


def load_il_actor_weights(actor, il_checkpoint):
    """Warm-start `actor.backbone` + `actor.trunk_head` + `actor.mean_head` from an IL
    checkpoint dict (`torch.load("best_il_model.pth")`). `actor.log_std_head` is left at its
    fresh init — IL has no notion of action-noise scale. Identical remap strategy to PPO's
    `policy.actor_critic.load_il_actor_weights`, via the same shared `policy/il_compat.py`
    helpers — see that module's docstring for why this guarantees the two algorithms can
    never silently diverge in how they interpret an IL checkpoint.
    """
    il_state_dict = il_checkpoint.get("model_state_dict", il_checkpoint)
    remapped = {}
    remapped.update(remap_backbone_state_dict(il_state_dict, "backbone"))
    remapped.update(remap_trunk_head_state_dict(il_state_dict, "trunk_head"))
    remapped.update(remap_final_layer_state_dict(il_state_dict, "mean_head"))
    return load_matching(actor, remapped, allow_missing=("log_std_head.weight", "log_std_head.bias"))
