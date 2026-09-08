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


# Cac cot KHONG duoc phep nam trong observation vi chung la HE QUA cua hanh dong dang can
# du doan, do cung mot thoi diem — dua vao la ro ri nhan (causal leak), khong phai dac trung.
#
#   yaw_rate_rps[t]  : xe dang quay CHINH VI vo-lang dang quay. Do tren checkpoint v8/v4 cho
#                      d(steer)/d(yaw_rate) = +0.32 / +0.44, tuc bien do steer do yaw_rate
#                      gay ra lon gap 114 lan (v8) va 290 lan (v4) so voi do ANH gay ra.
#                      Model chi doc lai dap an. Trong vong kin, yaw_rate khoi tao bang 0 ->
#                      steer 0 -> xe di thang -> yaw_rate van 0: mot vong lap tu duy tri lam
#                      xe lao thang cho toi khi ra khoi duong.
#   previous_steer   : ro ri yeu hon (bien do 0.175) nhung cung co che.
#   previous_longitudinal : gay sup do ve phanh — xe dung yen thi model tiep tuc phanh.
#
# `speed_mps` / `speed_limit_kmh` KHONG nam day: toc do la he qua cua ga TRONG QUA KHU, va
# la thong tin bat buoc de quyet dinh ga hien tai.
LEAKY_OBSERVATION_COLS = ("yaw_rate_rps", "previous_steer", "previous_longitudinal")


class ObservationContract(object):
    def __init__(self, checkpoint):
        self.num_classes = checkpoint["num_classes"]
        self.image_height = checkpoint.get("image_height")
        self.image_width = checkpoint.get("image_width")
        self.scalar_feature_dim = checkpoint["scalar_feature_dim"]
        # Nhip ra quyet dinh luc train IL (giay). `previous_steer`/`previous_longitudinal`
        # nghia la "lenh cua control_dt truoc" — env DRL va bridge phai chay dung nhip nay
        # (xem CarlaLaneKeepEnv._check_control_rate / LearnedAutopilotMode.action_repeat).
        self.control_dt = checkpoint.get("control_dt")
        self.continuous_cols = list(checkpoint["continuous_cols"])
        self.raw_action_cols = list(checkpoint["raw_action_cols"])
        self.norm_stats = checkpoint["norm_stats"]
        self.traffic_light_vocab = list(checkpoint["traffic_light_vocab"])
        # Do lech chuan THUC TE cua hanh dong trong tap train IL, [steer, longitudinal].
        # Dung lam `log_std` khoi tao cho actor DRL: no la thang nhieu tu nhien cua chinh
        # hanh vi dang duoc fine-tune, chinh xac hon bat ky uoc luong tay nao. None neu
        # checkpoint cu khong ghi truong nay.
        action_std = checkpoint.get("action_std")
        self.action_std = [float(v) for v in action_std] if action_std is not None else None

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
        # ban 6 lop co Vehicle/Sky) van nap duoc vi train_sac.py dung
        # `contract.num_classes` de dung backbone. Shape khop het, khong gi bao loi, nhung
        # resize_class_map() o duoi lai sinh chi so theo bang 4 lop hien tai -> actor nhan
        # sai kenh va lai sai. Fail ngay tai day, voi thong bao noi ro phai train lai gi.
        if self.num_classes != NUM_SEG_CLASSES:
            raise ValueError(
                "Checkpoint IL dung %d lop segmentation, nhung pipeline hien tai dung %d "
                "lop (%s). Day KHONG phai loi shape - no se chay im lang va lai sai. Train "
                "lai theo thu tu: train-segment-lane.ipynb -> train_il.ipynb -> DRL, hoac quay ve "
                "dung checkpoint IL cung phien ban bang nhan." %
                (self.num_classes, NUM_SEG_CLASSES, SEG_CLASS_NAMES))

        leaks = [c for c in (self.continuous_cols + self.raw_action_cols)
                 if c in LEAKY_OBSERVATION_COLS]
        if leaks:
            print("[!] CANH BAO RO RI QUAN SAT: checkpoint IL nay dua %s vao observation.\n"
                  "    Day la he qua cua chinh hanh dong can du doan, nen model se hoc doc\n"
                  "    lai dap an thay vi nhin anh segmentation. MAE offline se rat dep va\n"
                  "    xe se KHONG lai duoc trong vong kin. Train lai voi\n"
                  "    behavior_cloning/train_il.ipynb (da bo cac cot nay)." % (leaks,))

    def default_log_std(self, fallback=(-3.0, -1.5)):
        """`log(action_std)` tu checkpoint, hoac `fallback` neu checkpoint khong ghi.

        Ket qua duoc kep duoi -1.0: `action_std` do tren TOAN tap train, nen no gom ca bien
        thien giua cac tinh huong khac nhau (vao cua vs di thang), khong chi nhieu quanh
        mot trang thai. Dung nguyen si o chieu longitudinal (std 0.306 -> log -1.19) se cho
        mot policy tham do rong hon muc can thiet ngay tu rollout dau.
        """
        import math
        if not self.action_std:
            return tuple(fallback)
        return tuple(min(math.log(max(v, 1e-6)), -1.0) for v in self.action_std)

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
    the same table as `train-segment-lane.ipynb`'s `LABEL_LUT` / `train_il.ipynb`'s `SEG_LABEL_LUT`),
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
        lane_offset, heading_error, off_lane, is_junction = 0.0, 0.0, 1, 0
        half_width = 1.75
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
        # TRONG NGA TU, ba dai luong tren KHONG con y nghia. `get_waypoint(project_to_road)`
        # bam vao lan gan nhat, ma trong nga tu cac nhanh cat nhau nen "lan gan nhat" doi
        # sang nhanh vuong goc/nguoc chieu chi sau vai met. Do tren runs/il_demo_v9:
        # heading_error nhay 175 do trong MOT buoc 0.2s o 7.4 m/s = yaw rate 874 do/s, tuc
        # bat kha thi voi xe that — do la tham chieu nhay, khong phai xe quay.
        # Ai dung `lane_offset_m`/`heading_error_rad`/`off_lane` de tinh reward hay de danh
        # gia PHAI kiem co nay truoc (xem envs/carla_lane_keep_env.py::_compute_reward).
        # Ten cot giong `is_junction` cua collector (data_collection/.../schema.py) va cua
        # AUX_COLS trong notebook IL — co y de ba noi goi cung mot thu bang cung mot ten.
        is_junction = int(waypoint.is_junction)

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
        "is_junction": is_junction,
        # Nua be rong lan THUC TE tai waypoint nay. Duoc dung lam thang chuan hoa cho
        # `lane_offset_m` trong reward (xem CarlaLaneKeepEnv._compute_reward): lan cao toc
        # Town04 rong hon lan pho Town01, nen "lech 0.5 m" khong cung mot muc do nguy hiem
        # o hai noi. Cung la nguong ma `off_lane` dung, nen hai dai luong nhat quan.
        "lane_half_width_m": float(half_width),
    }
