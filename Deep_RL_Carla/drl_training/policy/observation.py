"""Turns a raw per-step CARLA state dict into the model's (seg_map, scalar_vector) input,
using the exact normalization the IL checkpoint was trained with.

The `ObservationContract` reads `continuous_cols` / `raw_action_cols` / `norm_stats` /
`traffic_light_vocab` straight out of the IL checkpoint instead of hardcoding them here a
second time. The Kaggle IL notebook and this local package run in different Python
environments (one on Kaggle, one on your machine against a live CARLA server) and can't
share a Python import, so encoding the contract *in the checkpoint's own data* is what
keeps the two from silently drifting apart if the feature set ever changes — the checkpoint
carries its own contract, not a hardcoded copy of it.

`ObservationContract` and `resize_class_map()` only read plain dicts/checkpoints/arrays —
neither needs the `carla` package. `carla` is only required by `traffic_light_label()` and
`build_vehicle_state()` (both talk to a live vehicle/world), so importing it is deferred to
those two functions via `_require_carla()` below — a caller that only wants
`ObservationContract` (e.g. to read a checkpoint's contract for reporting) can import this
module on a machine without the CARLA Python API installed.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]  # policy/ -> drl_training/ -> Deep_RL_Carla/
_DATA_COLLECTION_DIR = _REPO_ROOT / "data_collection"
if str(_DATA_COLLECTION_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_COLLECTION_DIR))

try:
    # Dung lai dung cong thuc voi carla_collector (data_collection) va router_plan thay vi
    # viet lai inline: normalize_angle/magnitude phai cho DUNG MOT ket qua o moi noi, vi
    # cung mot dai luong vat ly (heading_error, speed) nay duoc dua vao chung 1 khong gian
    # input cho ca IL, DRL observation va A* router. Cung ly do, resize_class_map() ben duoi
    # dung lai dung bang remap raw->lop huan luyen voi schema.py va cac notebook train,
    # thay vi tu dinh nghia rieng mot ban thu ba.
    from carla_collector.geometry import magnitude, normalize_angle
    from carla_collector.schema import (
        NUM_SEG_CLASSES, RAW_TO_TRAIN_LANE_LUT, SEG_CLASS_NAMES,
        THIN_COVER_THRESH, THIN_SEG_CLASS_IDS)
except ImportError as exc:
    raise ImportError(
        "Khong import duoc carla_collector tu '%s'. Dam bao thu muc 'data_collection' van "
        "nam canh 'drl_training' trong repo (khong doi ten/di chuyen thu muc goc)." %
        _DATA_COLLECTION_DIR
    ) from exc


def _require_carla():
    """Lazy `import carla`, dung boi traffic_light_label()/build_vehicle_state() — day la 2
    ham DUY NHAT trong module nay thuc su can noi voi vehicle/world CARLA song."""
    try:
        import carla
    except ImportError as exc:
        raise ImportError(
            "Khong import duoc module 'carla'. Ham nay chi chay duoc voi CARLA 0.9.10 "
            "Python API da cai (xem docs/manual_thu_thap_du_lieu.md muc 2)."
        ) from exc
    return carla


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

        # Chot chan cho loi IM LANG nguy hiem nhat cua pipeline nay: checkpoint IL cu (vd
        # ban 6 lop co Vehicle/Sky) van nap duoc vi train_ppo.py dung
        # `contract.num_classes` de dung backbone. Shape khop het, khong gi bao loi, nhung
        # resize_class_map() o duoi lai sinh chi so theo bang 4 lop hien tai -> actor nhan
        # sai kenh va lai sai. Fail ngay tai day, voi thong bao noi ro phai train lai gi.
        if self.num_classes != NUM_SEG_CLASSES:
            raise ValueError(
                "Checkpoint IL dung %d lop segmentation, nhung pipeline hien tai dung %d "
                "lop (%s). Day KHONG phai loi shape - no se chay im lang va lai sai. Train "
                "lai theo thu tu: carla_seg.ipynb -> train_il.ipynb -> DRL, hoac quay ve "
                "dung checkpoint IL cung phien ban bang nhan." %
                (self.num_classes, NUM_SEG_CLASSES, SEG_CLASS_NAMES))

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
    """Remaps a raw CARLA semantic-segmentation frame (raw tags 0-22) onto the project's
    4-class lane-keeping scheme (`RAW_TO_TRAIN_LANE_LUT` — Background/Road/RoadLine/Sidewalk,
    the same table as `carla_seg.ipynb`'s `LABEL_LUT` / `train_il.ipynb`'s `SEG_LABEL_LUT`),
    then nearest-neighbour resizes it.

    This is the single choke point every *live* caller — this env's `_make_observation` and
    the Carla Console Bridge Server's `learned_autopilot.py` — pushes a fresh camera frame
    through before handing it to `PolicyBackbone`'s `F.one_hot(..., num_classes=NUM_CLASSES)`.
    Without the remap, a raw tag above the class count (the sky alone guarantees this on
    nearly every frame) throws an index/CUDA "device-side assert triggered" error the moment
    it reaches `F.one_hot`; where it does not crash, the raw tag silently lines up with the
    wrong training class — e.g. raw 1 (Building) would be read as class 1 (Road).

    Never use linear/cubic interpolation on class IDs either — interpolating discrete class
    IDs invents nonsense intermediate classes, hence INTER_NEAREST for the resize.

    Downscaling is NOT plain INTER_NEAREST, though. Nearest sampling keeps a pixel only if
    the sample point happens to land on it, so it erases classes thinner than the sampling
    step — and `RoadLine` is exactly that: 2-3 px wide near the car, 1 px near the horizon.
    Measured on a perspective lane mask, 384x480 -> 192x240 with plain nearest keeps only
    69% of the RoadLine pixels that the coverage-preserving pass keeps (and 71% in the far
    half of the road, where the marking decides how early the car starts a turn). That is a
    silent train/inference mismatch: `train_il.ipynb` downscales its training masks with
    `downscale_labels`, which restores any output cell whose *area coverage* by a thin class
    exceeds `THIN_COVER_THRESH`. This function has to do the identical thing, or a
    warm-started policy sees a systematically thinner lane marking online than it was
    trained on. Same table, same threshold, same order — all three live in `schema.py`.
    """
    remapped = RAW_TO_TRAIN_LANE_LUT[class_map]
    if remapped.shape[0] == height and remapped.shape[1] == width:
        return remapped

    resized = cv2.resize(remapped, (width, height), interpolation=cv2.INTER_NEAREST)
    if height >= remapped.shape[0] and width >= remapped.shape[1]:
        return resized  # upscaling cannot drop a class; nearest is enough

    for class_id in THIN_SEG_CLASS_IDS:
        mask = (remapped == class_id)
        if not mask.any():
            continue
        # INTER_AREA on the 0/1 mask = fraction of each output cell covered by this class.
        coverage = cv2.resize(mask.astype(np.float32), (width, height),
                              interpolation=cv2.INTER_AREA)
        resized[coverage > THIN_COVER_THRESH] = class_id
    return resized


_TRAFFIC_LIGHT_MAP = None  # built lazily — needs `carla` already imported


def traffic_light_label(vehicle):
    """Vehicle's current traffic-light state as one of the raw string labels
    `ObservationContract.normalize_traffic_light` understands ("red"/"yellow"/"green"/
    "unknown")."""
    carla = _require_carla()
    global _TRAFFIC_LIGHT_MAP
    if not vehicle.is_at_traffic_light():
        return "unknown"
    if _TRAFFIC_LIGHT_MAP is None:
        _TRAFFIC_LIGHT_MAP = {
            carla.TrafficLightState.Red: "red",
            carla.TrafficLightState.Yellow: "yellow",
            carla.TrafficLightState.Green: "green",
        }
    return _TRAFFIC_LIGHT_MAP.get(vehicle.get_traffic_light_state(), "unknown")


def build_vehicle_state(vehicle, world_map, seg, previous_steer, previous_longitudinal):
    """Turns one live CARLA tick into the driving-state dict `ObservationContract` and a
    reward/termination function need — lane offset/heading error relative to the nearest
    driving waypoint, forward speed, yaw rate, traffic light label.

    Shared by the DRL training env (`envs/carla_lane_keep_env.py`, which adds its own
    collision/lane-invasion bookkeeping on top for the reward) and the Carla Console Bridge
    Server's IL/DRL Autopilot modes (`CarlaConsole/bridge_server/bridge/modes/
    learned_autopilot.py`, which only needs the state dict itself to drive) — the lane
    offset/heading math is the one part that's easy to get subtly wrong twice, so it exists
    exactly once here.
    """
    carla = _require_carla()
    transform = vehicle.get_transform()
    location = transform.location
    velocity = vehicle.get_velocity()
    angular = vehicle.get_angular_velocity()
    yaw = np.radians(transform.rotation.yaw)
    forward_speed = velocity.x * np.cos(yaw) + velocity.y * np.sin(yaw)

    waypoint = world_map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if waypoint is None:
        lane_offset, heading_error, off_lane = 0.0, 0.0, 1
    else:
        wp_tf = waypoint.transform
        dx = location.x - wp_tf.location.x
        dy = location.y - wp_tf.location.y
        wp_yaw = np.radians(wp_tf.rotation.yaw)
        lane_offset = dx * (-np.sin(wp_yaw)) + dy * np.cos(wp_yaw)
        half_width = max(float(waypoint.lane_width) * 0.5, 1e-6)
        off_lane = int(abs(lane_offset) > half_width)
        heading_error_deg = normalize_angle(transform.rotation.yaw - wp_tf.rotation.yaw)
        heading_error = np.radians(heading_error_deg)

    return {
        "seg": seg,
        "speed_mps": float(magnitude(velocity)),
        "forward_speed_mps": float(forward_speed),
        "yaw_rate_rps": float(np.radians(angular.z)),
        "previous_steer": previous_steer,
        "previous_longitudinal": previous_longitudinal,
        "speed_limit_kmh": vehicle.get_speed_limit(),
        "traffic_light_state": traffic_light_label(vehicle),
        "lane_offset_m": float(lane_offset),
        "heading_error_rad": float(heading_error),
        "off_lane": off_lane,
    }
