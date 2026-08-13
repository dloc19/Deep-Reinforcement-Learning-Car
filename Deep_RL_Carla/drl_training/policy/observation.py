"""Turns a raw per-step CARLA state dict into the model's (seg_map, scalar_vector) input,
using the exact normalization the IL checkpoint was trained with.

The `ObservationContract` reads `continuous_cols` / `raw_action_cols` / `norm_stats` /
`traffic_light_vocab` straight out of the IL checkpoint instead of hardcoding them here a
second time. The Kaggle IL notebook and this local package run in different Python
environments (one on Kaggle, one on your machine against a live CARLA server) and can't
share a Python import, so encoding the contract *in the checkpoint's own data* is what
keeps the two from silently drifting apart if the feature set ever changes — the checkpoint
carries its own contract, not a hardcoded copy of it.
"""

import cv2
import numpy as np


class ObservationContract(object):
    def __init__(self, checkpoint):
        self.num_classes = checkpoint["num_classes"]
        self.image_height = checkpoint.get("image_height")
        self.image_width = checkpoint.get("image_width")
        self.scalar_feature_dim = checkpoint["scalar_feature_dim"]
        self.continuous_cols = list(checkpoint["continuous_cols"])
        self.raw_action_cols = list(checkpoint["raw_action_cols"])
        self.norm_stats = checkpoint["norm_stats"]
        self.traffic_light_vocab = list(checkpoint["traffic_light_vocab"])

        expected_dim = len(self.continuous_cols) + len(self.raw_action_cols) + len(self.traffic_light_vocab)
        if expected_dim != self.scalar_feature_dim:
            raise ValueError(
                "scalar_feature_dim (%d) trong checkpoint khong khop so cot suy ra tu "
                "continuous_cols+raw_action_cols+traffic_light_vocab (%d). Checkpoint co "
                "the tu mot phien ban notebook khac hoac bi chinh sua tay." %
                (self.scalar_feature_dim, expected_dim))

        missing_stats = [c for c in self.continuous_cols if c not in self.norm_stats]
        if missing_stats:
            raise ValueError("norm_stats trong checkpoint thieu cot: %s" % missing_stats)

    def normalize_traffic_light(self, value):
        v = str(value).strip().lower()
        return v if v in self.traffic_light_vocab else "unknown"

    def build_scalar_vector(self, state):
        """`state`: dict with (at least) the keys in `continuous_cols` + `raw_action_cols`
        + "traffic_light_state" (raw string label, any case — normalized here the same way
        the IL notebook normalized it at training time).
        """
        cont_vals = []
        for col in self.continuous_cols:
            mean, std = self.norm_stats[col]
            cont_vals.append((float(state[col]) - mean) / std)
        raw_vals = [float(state[col]) for col in self.raw_action_cols]
        tl = self.normalize_traffic_light(state.get("traffic_light_state", "unknown"))
        tl_onehot = [1.0 if tl == vocab_value else 0.0 for vocab_value in self.traffic_light_vocab]
        return np.array(cont_vals + raw_vals + tl_onehot, dtype=np.float32)


def resize_class_map(class_map, height, width):
    """Nearest-neighbour resize for a (H, W) uint8 class-ID map — never use linear/cubic
    interpolation on class IDs (see the segmentation notebook's own warning: interpolating
    discrete class IDs invents nonsense intermediate classes)."""
    if class_map.shape[0] == height and class_map.shape[1] == width:
        return class_map
    return cv2.resize(class_map, (width, height), interpolation=cv2.INTER_NEAREST)
