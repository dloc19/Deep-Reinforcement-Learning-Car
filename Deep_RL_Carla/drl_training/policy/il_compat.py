"""Generic IL-checkpoint -> DRL-module state_dict remapping, shared by PPO's
`GaussianActor` and SAC's `GaussianPolicy` (see `policy/actor_critic.py` / `sac/networks.py`).

IL's `SteeringNet` (in `behavior_cloning/train_il.ipynb`) has:
    conv.*, pool (no params), cnn_fc.*, scalar_mlp.*      -> the shared backbone
    head.0 (Linear 96->64), head.3 (Linear 64->32)          -> the shared "trunk head"
    head.6 (Linear 32->2, tanh applied outside the Sequential) -> the final action layer

Every actor in this package reuses `policy.backbone.PolicyBackbone` as `self.backbone` and
`policy.backbone.build_trunk_head(...)` as `self.trunk_head`, so the first two remap
functions below are algorithm-agnostic. Only the *final* layer differs per algorithm (PPO:
a single `mean_head`; SAC: `mean_head` warm-started the same way, `log_std_head` untouched —
see each module's own loader for how these are combined).
"""

# IL SteeringNet submodules with no algorithm-specific meaning - always map onto `backbone`.
_BACKBONE_PREFIXES = ("conv.", "pool.", "cnn_fc.", "scalar_mlp.")
# IL `head` Sequential indices: 0=Linear(96,64), 1=ELU, 2=Dropout, 3=Linear(64,32), 4=ELU,
# 5=Dropout, 6=Linear(32,2). Only the Linear layers carry weights.
_TRUNK_HEAD_PREFIX_MAP = {"head.0": ".0", "head.3": ".2"}
_FINAL_LAYER_PREFIX = "head.6"


def remap_backbone_state_dict(il_state_dict, backbone_attr="backbone"):
    return {
        backbone_attr + "." + key: tensor
        for key, tensor in il_state_dict.items()
        if key.startswith(_BACKBONE_PREFIXES)
    }


def remap_trunk_head_state_dict(il_state_dict, trunk_head_attr="trunk_head"):
    remapped = {}
    for key, tensor in il_state_dict.items():
        prefix, _, suffix = key.rpartition(".")
        if prefix in _TRUNK_HEAD_PREFIX_MAP:
            remapped[trunk_head_attr + _TRUNK_HEAD_PREFIX_MAP[prefix] + "." + suffix] = tensor
    return remapped


def remap_final_layer_state_dict(il_state_dict, target_attr):
    """`target_attr`: the attribute name of the DRL module's final 32->2 Linear layer
    (e.g. `"mean_head"`)."""
    remapped = {}
    for key, tensor in il_state_dict.items():
        prefix, _, suffix = key.rpartition(".")
        if prefix == _FINAL_LAYER_PREFIX:
            remapped[target_attr + "." + suffix] = tensor
    return remapped


def load_matching(module, remapped, allow_missing=()):
    """Copies `remapped` tensors into `module` by exact key name, then calls
    `module.load_state_dict(...)` once. Fails loudly (RuntimeError) on any key present in
    `module.state_dict()` but absent from `remapped` (unless explicitly allowed via
    `allow_missing` — e.g. SAC's `log_std_head`, which has no IL equivalent), and on any
    shape mismatch. A silent skip here means part of the network quietly starts from random
    weights, which defeats the entire point of warm-starting and is very easy to miss.
    """
    own_state = module.state_dict()
    missing = [key for key in own_state if key not in remapped and key not in allow_missing]
    if missing:
        raise RuntimeError(
            "Khong tim thay trong so IL cho: %s. Kien truc mang va SteeringNet co the da "
            "lech nhau (xem docstring policy/backbone.py va policy/il_compat.py)." % missing)

    for key, tensor in remapped.items():
        if key not in own_state:
            raise RuntimeError("IL checkpoint co tensor '%s' nhung module khong co key nay." % key)
        if own_state[key].shape != tensor.shape:
            raise RuntimeError(
                "Shape lech o '%s': module=%s, checkpoint=%s. Kiem tra scalar_feature_dim / "
                "NUM_CLASSES giua IL notebook va DRL co khop nhau khong." %
                (key, tuple(own_state[key].shape), tuple(tensor.shape)))
        own_state[key] = tensor

    module.load_state_dict(own_state)
    return module
