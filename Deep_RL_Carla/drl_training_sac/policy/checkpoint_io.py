"""Load and validate an IL checkpoint produced by
`behavior_cloning/train_il_v9.ipynb`.
"""

import torch

REQUIRED_KEYS = (
    "model_state_dict", "num_classes", "scalar_feature_dim", "continuous_cols",
    "raw_action_cols", "norm_stats", "traffic_light_vocab",
)


def load_il_checkpoint(path, map_location="cpu"):
    """Loads the .pth dict and checks it has everything `ObservationContract` and
    `load_il_actor_weights` need. Fails fast with a clear message rather than letting a
    stale/incompatible checkpoint (e.g. saved before the observation-leakage fix) crash
    deep inside a training loop with a cryptic KeyError.
    """
    checkpoint = torch.load(path, map_location=map_location)
    missing = [key for key in REQUIRED_KEYS if key not in checkpoint]
    if missing:
        raise ValueError(
            "Checkpoint IL '%s' thieu truong: %s. Neu checkpoint nay duoc luu truoc khi "
            "sua bug ro ri quan sat (lane_offset_m/heading_error_rad/is_junction trong "
            "input), hay train lai bang notebook IL da sua." % (path, missing))
    return checkpoint
