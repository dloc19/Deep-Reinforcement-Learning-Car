"""Shared CNN(segmentation) + MLP(scalar) trunk, plus the shared "trunk head" used by every
DRL algorithm in this package (PPO's `GaussianActor`/`ValueCritic`, SAC's `GaussianPolicy`/
`QNetwork` — see `../drl_training/policy/actor_critic.py` and `sac/networks.py`).

This mirrors `SteeringNet` in `behavior_cloning/train_il_v9.ipynb` layer-for-layer,
including attribute names (`conv`, `pool`, `cnn_fc`, `scalar_mlp`) and the first two layers
of its `head`. That is not a style choice: `policy/il_compat.py` copies tensors by
state_dict key name from the IL checkpoint into these modules, so the definitions must stay
in sync. If you change the IL notebook's `SteeringNet` architecture, mirror the change here
(and vice versa) or warm-start loading will fail with a shape-mismatch error (by design —
see `checkpoint_io.py`/`il_compat.py`, which fail loudly rather than silently skipping
mismatched tensors).

Deliberately resolution-agnostic: `nn.AdaptiveAvgPool2d(POOL_GRID)` after the conv stack
means this backbone accepts any (H, W), so the DRL env can run at a different resolution from
the one IL trained on, while still loading IL conv weights directly. Three numbers are
involved and they are NOT the same number:
  - 480x384  what the collector records (`collector_config.json`'s `camera.width/height`)
    and what the DRL env's camera runs at (`sac_config.json`).
  - 240x192  what `SteeringNet` actually saw during IL training (`IMAGE_WIDTH`/
    `IMAGE_HEIGHT` in train_il_v9.ipynb — masks are downscaled with `downscale_labels`).
  - 160x128  this file's fallback when a config omits the field.
Nothing breaks when they differ, but a warm-started actor does see a different scale of
pooled features than IL did; fine-tuning adapts those statistics. Set the env's
`obs_width`/`obs_height` to 240x192 if you want the observation to match IL exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .observation import NUM_SEG_CLASSES

# Segmentation classes shared by the whole project: Background, Road, RoadLine, Sidewalk.
# Source of truth is `data_collection/carla_collector/schema.py::SEG_CLASS_NAMES`; it is
# imported (not re-typed) so this file cannot silently drift away from the collector and
# from `policy/observation.py::resize_class_map`, which produces the class ids fed here.
# A mismatch is the worst kind of bug in this pipeline: if the count still matches, nothing
# raises and the model just reads the wrong channels.
NUM_CLASSES = NUM_SEG_CLASSES

# Luoi pooling sau conv stack. (1, 1) — ban cu — la TRUNG BINH TOAN ANH cua tung kenh: no
# bop ban do dac trung 12x15 thanh mot so moi kenh, nen phan bo TRAI-PHAI cua lan duong (thu
# duy nhat noi cho xe biet no dang lech ve ben nao) phai di vong qua tuong quan giua cac kenh
# thay vi duoc bieu dien truc tiep. (4, 6) giu lai bo cuc tho: 4 hang (gan -> xa) x 6 cot
# (trai -> phai), tuc `cnn_fc` nhan 64*24 = 1536 chieu thay vi 64.
#
# PHAI KHOP `POOL_GRID` trong behavior_cloning/train_il_v9.ipynb §8. Doi mot ben ma quen ben
# kia thi `il_compat.load_matching` bao loi shape ngay o `cnn_fc.0.weight` — co y de nhu vay,
# vi day chinh xac la loai lech phai chet to chu khong duoc chay im lang.
POOL_GRID = (4, 6)


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
        self.pool = nn.AdaptiveAvgPool2d(POOL_GRID)
        self.cnn_fc = nn.Sequential(nn.Linear(64 * POOL_GRID[0] * POOL_GRID[1], 64), nn.ELU())
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


def freeze_batchnorm(*modules):
    """Dat MOI lop BatchNorm sang eval mode: dung running stats, va khong cap nhat chung.

    Bat buoc cho ca PPO lan SAC, vi hai ly do doc lap nhau — va ca hai deu la loi IM LANG:

    1. NHAT QUAN GIUA ROLLOUT VA UPDATE. `agent.act()` chay actor tren MOT quan sat
       (batch = 1), con `update()` chay lai chinh actor do tren minibatch 128. O train mode,
       BatchNorm chuan hoa theo thong ke CUA BATCH, nen hai lan forward do la hai mang KHAC
       NHAU. Ti so importance sampling `exp(new_log_prob - old_log_prob)` cua PPO gia dinh
       chung la mot; neu khong, moi advantage deu bi lech boi nhieu cua BatchNorm thay vi chi
       phan anh buoc cap nhat policy. Day dung la ly do PPO (`../drl_training/`) cung co y
       khong dung Dropout — BatchNorm o train mode gay ra cung mot van de, con nang hon.

    2. GIU TRONG SO WARM-START. `running_mean`/`running_var` cua actor den tu checkpoint IL
       va la mot phan cua hanh vi da hoc. O train mode, moi tick rollout lai cap nhat chung
       bang thong ke cua DUNG MOT khung hinh (momentum 0.1), nen chi vai tram buoc la bo
       thong ke IL bi ghi de bang nhieu — warm-start bi xoa dan ma khong co dau hieu gi.

    Trong so affine (`weight`/`bias`) cua BatchNorm VAN duoc hoc binh thuong; chi phan thong
    ke chay bi dong bang.
    """
    frozen = 0
    for module in modules:
        for sub in module.modules():
            if isinstance(sub, nn.modules.batchnorm._BatchNorm):
                sub.eval()
                frozen += 1
    return frozen


def build_trunk_head(in_features, hidden1=64, hidden2=32):
    """The shared post-backbone layers every algorithm-specific head branches off of.

    Matches `SteeringNet.head`'s first two Linear+ELU layers 1:1 (IL's `head.0` and
    `head.3` once its two `nn.Dropout` layers are excluded — dropout has no place in an
    RL actor/critic that gets re-evaluated on the same stored observation multiple times
    during an update, see `sac/networks.py` docstring). `policy/il_compat.py` relies
    on this exact shape (`in_features -> hidden1 -> hidden2`, two Linear+ELU pairs) to remap
    IL's `head.0`/`head.3` weights onto whatever this is attached to as `self.trunk_head`.
    """
    return nn.Sequential(
        nn.Linear(in_features, hidden1), nn.ELU(),
        nn.Linear(hidden1, hidden2), nn.ELU(),
    )
