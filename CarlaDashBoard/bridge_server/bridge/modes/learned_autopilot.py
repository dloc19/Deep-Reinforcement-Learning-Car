"""IL / DRL Autopilot — design doc §07, Phase 4: drive using a trained model instead of a
route or the Traffic Manager. Both modes are lane-keeping only (no destination, no A* route —
that's what ASTAR_AUTOPILOT is for), so the two share every bit of the tick loop; only *how*
one (seg, scalar) observation turns into an action differs, which is exactly what `predict`
(built by `bridge/il_drl_bridge.py::build_il_predictor`/`build_drl_predictor`) captures.

Mirrors `astar_autopilot.py`'s shape: `sim_loop.py` builds the model (and caches it — loading
a checkpoint from disk on every SetMode would be wasteful) and constructs this mode directly,
the same way it builds `AstarAutopilotMode` with an already-resolved `RouteContext`.
"""

import numpy as np
import carla

from .base import ModeRuntime


class LearnedAutopilotMode(ModeRuntime):
    def __init__(self, mode_name, drl_training, contract, predict):
        self.name = mode_name
        self.drl_training = drl_training
        self.contract = contract
        self.predict = predict
        self.session = None
        self.previous_steer = 0.0
        self.previous_longitudinal = 0.0
        self.last_state = {}

    def start(self, session):
        self.session = session
        self.previous_steer = 0.0
        self.previous_longitudinal = 0.0
        self.last_state = {}

    def tick(self, snapshot):
        vehicle = self.session.ego
        seg = self.session.last_seg_class_map
        if vehicle is None or not vehicle.is_alive or seg is None:
            return None  # xe chua san sang hoac chua nhan duoc frame segmentation nao

        height = self.contract.image_height or seg.shape[0]
        width = self.contract.image_width or seg.shape[1]
        seg_model = self.drl_training.resize_class_map(seg, height, width)

        state = self.drl_training.build_vehicle_state(
            vehicle, self.session.map, seg_model, self.previous_steer, self.previous_longitudinal)
        scalar = self.contract.build_scalar_vector(state)
        action = self.predict(seg_model, scalar)

        steer = float(np.clip(action[0], -1.0, 1.0))
        longitudinal = float(np.clip(action[1], -1.0, 1.0))
        throttle, brake = (longitudinal, 0.0) if longitudinal >= 0.0 else (0.0, -longitudinal)

        self.previous_steer, self.previous_longitudinal = steer, longitudinal
        self.last_state = state
        return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

    def stop(self):
        self.session = None
        self.last_state = {}

    def status_extra(self):
        if not self.last_state:
            return {}
        return {
            "lane_offset_m": round(self.last_state.get("lane_offset_m", 0.0), 2),
            "heading_error_rad": round(self.last_state.get("heading_error_rad", 0.0), 3),
            "off_lane": bool(self.last_state.get("off_lane")),
        }
