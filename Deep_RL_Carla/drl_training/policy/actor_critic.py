"""PPO actor (Gaussian policy) and critic (state-value), plus the IL-checkpoint warm-start
loader for the actor.

Design notes
------------
* `GaussianActor` = `backbone` (`policy.backbone.PolicyBackbone`) + `trunk_head`
  (`policy.backbone.build_trunk_head`) + `mean_head` (final 32->2 Linear). This three-piece
  split (rather than one monolithic `head` Sequential) is what lets `policy/il_compat.py`
  remap IL's `head.0`/`head.3`/`head.6` onto this network — and onto SAC's `GaussianPolicy`
  (`sac/networks.py`) — with the exact same helper functions instead of duplicated,
  drift-prone remap logic per algorithm.
* No `nn.Dropout`: PPO recomputes `log_prob(action)` for the *same* stored action at both
  rollout time and update time to form the importance-sampling ratio
  `exp(new_log_prob - old_log_prob)`. If dropout made those two forward passes
  stochastically different networks, the ratio would be biased by dropout noise instead of
  only reflecting the policy update — standard practice (any PPO reference implementation)
  keeps the acting/updated network deterministic given (obs, action).
* `log_std` is a free parameter (state-independent, per-action-dim) with no IL equivalent —
  IL never had to output an exploration scale. This is standard for PPO-continuous (e.g.
  OpenAI baselines, CleanRL's `ppo_continuous_action.py`), unlike SAC which canonically uses
  a *state-dependent* log_std (see `sac/networks.py`). Initialized to a modest std (~0.3) so
  early PPO rollouts stay close to the warm-started IL behaviour instead of exploring wildly.
* The critic is a *separate* network (own backbone + trunk_head + value_head), not a head
  bolted onto the actor's trunk. Common PPO-continuous practice (e.g. CleanRL): sharing a
  trunk lets value-loss gradients distort the actor's warm-started features, exactly what
  warm-starting is trying to protect. The extra compute is a non-issue next to a CARLA step.
"""

import torch
import torch.nn as nn

from .backbone import NUM_CLASSES, PolicyBackbone, build_trunk_head
from .il_compat import (
    load_matching, remap_backbone_state_dict, remap_final_layer_state_dict,
    remap_trunk_head_state_dict,
)


class GaussianActor(nn.Module):
    def __init__(self, num_scalar_features, num_classes=NUM_CLASSES,
                 log_std_init=-1.2, log_std_min=-5.0, log_std_max=0.5):
        super(GaussianActor, self).__init__()
        self.backbone = PolicyBackbone(num_scalar_features, num_classes)
        self.trunk_head = build_trunk_head(self.backbone.out_features)
        self.mean_head = nn.Linear(32, 2)
        self.log_std = nn.Parameter(torch.full((2,), float(log_std_init)))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def mean_action(self, seg_map, scalar_features):
        feats = self.trunk_head(self.backbone(seg_map, scalar_features))
        return torch.tanh(self.mean_head(feats))

    def distribution(self, seg_map, scalar_features):
        mean = self.mean_action(seg_map, scalar_features)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def evaluate_actions(self, seg_map, scalar_features, action):
        """Used by the PPO update: log_prob/entropy of a *stored* (already-sampled)
        action under the *current* policy parameters."""
        dist = self.distribution(seg_map, scalar_features)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy

    @torch.no_grad()
    def act(self, seg_map, scalar_features, deterministic=False):
        """Returns (env_action, raw_action, log_prob).

        `env_action` is clipped to [-1, 1] for `VehicleControl`; `raw_action` (unclipped) is
        what must be stored in the rollout buffer, because `log_prob` was computed for the
        raw Normal sample. Recomputing log_prob against the clipped action at update time
        would silently bias every advantage estimate near the action bounds. This
        store-raw/apply-clipped split is the standard, widely-used approximation for
        continuous PPO with bounded action spaces (e.g. Stable-Baselines3).
        """
        dist = self.distribution(seg_map, scalar_features)
        raw_action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(raw_action).sum(-1)
        env_action = torch.clamp(raw_action, -1.0, 1.0)
        return env_action, raw_action, log_prob

    @torch.no_grad()
    def select_action(self, seg_map, scalar_features, deterministic=False):
        """Algorithm-agnostic single-return convenience used by `evaluate.py` (shared
        across PPO/SAC — see `sac/networks.py::GaussianPolicy.select_action`)."""
        env_action, _raw_action, _log_prob = self.act(seg_map, scalar_features, deterministic)
        return env_action


class ValueCritic(nn.Module):
    def __init__(self, num_scalar_features, num_classes=NUM_CLASSES):
        super(ValueCritic, self).__init__()
        self.backbone = PolicyBackbone(num_scalar_features, num_classes)
        self.trunk_head = build_trunk_head(self.backbone.out_features)
        self.value_head = nn.Linear(32, 1)

    def forward(self, seg_map, scalar_features):
        feats = self.trunk_head(self.backbone(seg_map, scalar_features))
        return self.value_head(feats).squeeze(-1)


def load_il_actor_weights(actor, il_checkpoint):
    """Warm-start `actor.backbone` + `actor.trunk_head` + `actor.mean_head` from an IL
    checkpoint dict (as saved by `train_il_kaggle.ipynb`, i.e.
    `torch.load("best_il_model.pth")`).

    `actor.log_std` is left at its fresh init — IL has no notion of action-noise scale.
    """
    il_state_dict = il_checkpoint.get("model_state_dict", il_checkpoint)
    remapped = {}
    remapped.update(remap_backbone_state_dict(il_state_dict, "backbone"))
    remapped.update(remap_trunk_head_state_dict(il_state_dict, "trunk_head"))
    remapped.update(remap_final_layer_state_dict(il_state_dict, "mean_head"))
    return load_matching(actor, remapped, allow_missing=("log_std",))
