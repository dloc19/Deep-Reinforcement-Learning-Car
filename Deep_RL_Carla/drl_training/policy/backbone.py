"""Shared CNN(segmentation) + MLP(scalar) trunk, plus the shared "trunk head" used by every
DRL algorithm in this package (PPO's `GaussianActor`/`ValueCritic`, SAC's `GaussianPolicy`/
`QNetwork` — see `policy/actor_critic.py` and `sac/networks.py`).

This mirrors `SteeringNet` in `behavior_cloning/train_il_kaggle.ipynb` layer-for-layer,
including attribute names (`conv`, `pool`, `cnn_fc`, `scalar_mlp`) and the first two layers
of its `head`. That is not a style choice: `policy/il_compat.py` copies tensors by
state_dict key name from the IL checkpoint into these modules, so the definitions must stay
in sync. If you change the IL notebook's `SteeringNet` architecture, mirror the change here
(and vice versa) or warm-start loading will fail with a shape-mismatch error (by design —
see `checkpoint_io.py`/`il_compat.py`, which fail loudly rather than silently skipping
mismatched tensors).

Deliberately resolution-agnostic: `nn.AdaptiveAvgPool2d((1, 1))` after the conv stack means
this backbone accepts any (H, W), so the DRL env can run at a lower resolution than the IL
training resolution (384x480) to keep the PPO rollout buffer small, while still being able
to load IL conv weights directly. Fine-tuning will adapt the pooled-feature statistics to
the new resolution; no architectural change is required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 13  # CARLA semantic segmentation classes used across the whole project


class PolicyBackbone(nn.Module):
    def __init__(self, num_scalar_features, num_classes=NUM_CLASSES):
        super(PolicyBackbone, self).__init__()
        self.num_classes = num_classes
        self.conv = nn.Sequential(
            nn.Conv2d(num_classes, 24, 5, 2, 2), nn.BatchNorm2d(24), nn.ELU(),
            nn.Conv2d(24, 36, 5, 2, 2), nn.BatchNorm2d(36), nn.ELU(),
            nn.Conv2d(36, 48, 5, 2, 2), nn.BatchNorm2d(48), nn.ELU(),
            nn.Conv2d(48, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ELU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn_fc = nn.Sequential(nn.Linear(64, 64), nn.ELU())
        self.scalar_mlp = nn.Sequential(
            nn.Linear(num_scalar_features, 32), nn.ELU(),
            nn.Linear(32, 32), nn.ELU(),
        )

    @property
    def out_features(self):
        return 64 + 32

    def forward(self, seg_map, scalar_features):
        """seg_map: (N, H, W) int64 class-id map, or (N, num_classes, H, W) one-hot float.

        Accepting the raw class-id map lets callers (the rollout buffer, the env) store
        1 byte/pixel instead of `num_classes` floats/pixel; one-hot happens here, on the
        already-batched GPU tensor, right before the conv stack.
        """
        if seg_map.dim() == 3:
            x_img = F.one_hot(seg_map.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        else:
            x_img = seg_map
        x_img = self.cnn_fc(self.pool(self.conv(x_img)).flatten(1))
        x_sca = self.scalar_mlp(scalar_features)
        return torch.cat([x_img, x_sca], dim=1)


def build_trunk_head(in_features, hidden1=64, hidden2=32):
    """The shared post-backbone layers every algorithm-specific head branches off of.

    Matches `SteeringNet.head`'s first two Linear+ELU layers 1:1 (IL's `head.0` and
    `head.3` once its two `nn.Dropout` layers are excluded — dropout has no place in an
    RL actor/critic that gets re-evaluated on the same stored observation multiple times
    during an update, see `policy/actor_critic.py` docstring). `policy/il_compat.py` relies
    on this exact shape (`in_features -> hidden1 -> hidden2`, two Linear+ELU pairs) to remap
    IL's `head.0`/`head.3` weights onto whatever this is attached to as `self.trunk_head`.
    """
    return nn.Sequential(
        nn.Linear(in_features, hidden1), nn.ELU(),
        nn.Linear(hidden1, hidden2), nn.ELU(),
    )
