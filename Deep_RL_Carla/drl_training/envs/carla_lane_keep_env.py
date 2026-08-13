"""Active-client CARLA lane-keeping environment for DRL fine-tuning (PPO or SAC).

Unlike `data_collection/` (a *passive* client that attaches to a vehicle already spawned by
`automatic_control.py`), this environment is an *active* client: it spawns its own ego
vehicle, drives the simulation in synchronous mode, and owns the world tick. Do NOT run
this alongside the passive data-collection pipeline or `automatic_control.py` against the
same CARLA server/world — both would fight over vehicle control and world settings.

Observation, reward and termination follow `docs/csv_fields_by_task.md` ("DRL" section) and
`docs/manual_thu_thap_du_lieu.md` §9.2, scoped to lane-keeping only: no A* route fields,
since `router_plan/Global_Route_Planner.py` is not implemented yet and the thesis itself
targets lane-keeping control, not full route navigation.
"""

import queue
import random
import sys
from pathlib import Path

import numpy as np

_ENV_FILE = Path(__file__).resolve()
_DRL_TRAINING_DIR = _ENV_FILE.parents[1]          # "drl_training/"
_REPO_ROOT = _ENV_FILE.parents[2]                  # "Deep_RL_Carla/"
_DATA_COLLECTION_DIR = _REPO_ROOT / "data_collection"
for _path in (str(_DRL_TRAINING_DIR), str(_DATA_COLLECTION_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from carla_collector.events import EventCounters
    from carla_collector.geometry import magnitude, normalize_angle
except ImportError as exc:
    raise ImportError(
        "Khong import duoc carla_collector tu '%s'. Dam bao thu muc 'data_collection' van "
        "nam canh 'drl_training' trong repo (khong doi ten/di chuyen thu muc goc)." %
        _DATA_COLLECTION_DIR
    ) from exc

from policy.observation import resize_class_map

try:
    import carla
except ImportError as exc:
    raise ImportError(
        "Khong import duoc module 'carla'. Cai CARLA 0.9.10 Python API truoc khi chay "
        "module DRL (xem docs/manual_thu_thap_du_lieu.md muc 2)."
    ) from exc


_TRAFFIC_LIGHT_MAP = None  # built lazily — needs `carla` already imported


def _traffic_light_label(vehicle):
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


class CarlaLaneKeepEnv(object):
    """Duck-typed Gym-like env: `reset() -> (obs, info)`, `step(action) -> (obs, reward,
    terminated, truncated, info)` — the modern Gymnasium 5-tuple `step()` contract.

    Deliberately NOT a `gymnasium.Env` subclass: this project targets the CARLA 0.9.10
    Python 3.7 client API, and recent `gymnasium` releases have dropped 3.7 support. The
    interface matches Gymnasium closely enough that wrapping this in a real `gymnasium.Env`
    later (on a newer Python + a rebuilt CARLA wheel) is a thin shim, not a rewrite.

    `observation_contract` (a `policy.observation.ObservationContract`, built once from the
    IL checkpoint in `train_ppo.py`) is required: it is the single source of truth for how
    raw CARLA state turns into the scalar feature vector, shared with the actor/critic that
    consume this env's observations.
    """

    def __init__(self, config, observation_contract):
        self.cfg = config
        self.contract = observation_contract
        self.client = carla.Client(config["host"], config.get("port", 2000))
        self.client.set_timeout(config.get("timeout", 20.0))
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self._apply_synchronous_mode()

        self.spawn_points = self.map.get_spawn_points()
        if not self.spawn_points:
            raise RuntimeError("Map hien tai (%s) khong co spawn point nao." % self.map.name)

        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.events = None
        self._seg_queue = None

        self.previous_steer = 0.0
        self.previous_longitudinal = 0.0
        self.previous_collisions = 0
        self.previous_lane_invasions = 0
        self.off_lane_streak = 0
        self.step_count = 0
        self._rng = random.Random(config.get("seed"))

    # ------------------------------------------------------------------ setup / teardown
    def _apply_synchronous_mode(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / float(self.cfg.get("fps", 10.0))
        settings.no_rendering_mode = bool(self.cfg.get("no_rendering", False))
        self.world.apply_settings(settings)

    def _camera_blueprint(self):
        bp = self.world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
        bp.set_attribute("image_size_x", str(self.cfg.get("width", 160)))
        bp.set_attribute("image_size_y", str(self.cfg.get("height", 128)))
        bp.set_attribute("fov", str(self.cfg.get("fov", 90.0)))
        return bp

    def _on_camera_image(self, image):
        # Same decode as `data_collection/carla_collector/writer.py`: raw class ID lives in
        # the red channel of the semantic segmentation camera's BGRA output.
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        labels = array[:, :, 2].copy()
        if self._seg_queue.full():
            try:
                self._seg_queue.get_nowait()
            except queue.Empty:
                pass
        self._seg_queue.put_nowait(labels)

    def _spawn_actors(self):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(self.cfg.get("vehicle_filter", "vehicle.lincoln.mkz2017"))[0]
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "drl_ego")

        spawn_points = list(self.spawn_points)
        self._rng.shuffle(spawn_points)
        vehicle = None
        for transform in spawn_points:
            vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
            if vehicle is not None:
                break
        if vehicle is None:
            raise RuntimeError("Khong spawn duoc xe o bat ky spawn point nao (co the dang bi chiem).")
        self.vehicle = vehicle

        camera_tf = carla.Transform(
            carla.Location(
                x=self.cfg.get("camera_x", 1.5),
                y=self.cfg.get("camera_y", 0.0),
                z=self.cfg.get("camera_z", 2.4)),
            carla.Rotation(pitch=self.cfg.get("camera_pitch", -5.0)),
        )
        self._seg_queue = queue.Queue(maxsize=1)
        self.camera = self.world.spawn_actor(
            self._camera_blueprint(), camera_tf, attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid)
        self.camera.listen(self._on_camera_image)

        self.events = EventCounters()
        self.collision_sensor = self.world.spawn_actor(
            blueprint_library.find("sensor.other.collision"), carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(self.events.collision)
        self.lane_invasion_sensor = self.world.spawn_actor(
            blueprint_library.find("sensor.other.lane_invasion"), carla.Transform(), attach_to=self.vehicle)
        self.lane_invasion_sensor.listen(self.events.lane_invasion)

    def _destroy_actors(self):
        actors = (self.camera, self.collision_sensor, self.lane_invasion_sensor, self.vehicle)
        for actor in actors:
            if actor is not None and hasattr(actor, "stop"):
                try:
                    actor.stop()
                except RuntimeError:
                    pass
        for actor in actors:
            if actor is not None:
                try:
                    if actor.is_alive:
                        actor.destroy()
                except RuntimeError:
                    pass
        self.vehicle = self.camera = self.collision_sensor = self.lane_invasion_sensor = None
        self._seg_queue = None

    def close(self):
        self._destroy_actors()
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        except RuntimeError:
            pass

    # ------------------------------------------------------------------------- Gym API
    def reset(self):
        self._destroy_actors()
        self._spawn_actors()
        self.previous_steer = 0.0
        self.previous_longitudinal = 0.0
        self.previous_collisions = 0
        self.previous_lane_invasions = 0
        self.off_lane_streak = 0
        self.step_count = 0

        seg = None
        for _ in range(self.cfg.get("warmup_ticks", 4)):
            self.world.tick()
            seg = self._get_seg_frame()
        state = self._build_state(seg)
        return self._make_observation(state), {"state": state}

    def step(self, action):
        steer = float(np.clip(action[0], -1.0, 1.0))
        longitudinal = float(np.clip(action[1], -1.0, 1.0))
        throttle, brake = (longitudinal, 0.0) if longitudinal >= 0.0 else (0.0, -longitudinal)
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
        self.world.tick()
        seg = self._get_seg_frame()
        self.step_count += 1

        state = self._build_state(seg)
        reward, terminated, info = self._compute_reward(state, steer, longitudinal)
        truncated = self.step_count >= self.cfg.get("max_episode_steps", 1000)

        self.previous_steer = steer
        self.previous_longitudinal = longitudinal
        info["state"] = state
        return self._make_observation(state), reward, terminated, truncated, info

    # ------------------------------------------------------------------------- internals
    def _get_seg_frame(self):
        timeout = self.cfg.get("frame_timeout", 5.0)
        try:
            return self._seg_queue.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(
                "Khong nhan duoc frame camera segmentation trong %.1fs — kiem tra CARLA "
                "server con chay, fps cau hinh, va khong co client passive nao khac dang "
                "giu synchronous_mode." % timeout)

    def _build_state(self, seg):
        transform = self.vehicle.get_transform()
        location = transform.location
        velocity = self.vehicle.get_velocity()
        angular = self.vehicle.get_angular_velocity()
        yaw = np.radians(transform.rotation.yaw)
        forward_speed = velocity.x * np.cos(yaw) + velocity.y * np.sin(yaw)

        waypoint = self.map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
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
            heading_error = np.radians(normalize_angle(transform.rotation.yaw - wp_tf.rotation.yaw))

        collisions, lane_invasions = self.events.snapshot()
        new_collision = collisions > self.previous_collisions
        new_lane_invasions = max(0, lane_invasions - self.previous_lane_invasions)
        self.previous_collisions, self.previous_lane_invasions = collisions, lane_invasions

        return {
            "seg": seg,
            "speed_mps": magnitude(velocity),
            "forward_speed_mps": float(forward_speed),
            "yaw_rate_rps": float(np.radians(angular.z)),
            "previous_steer": self.previous_steer,
            "previous_longitudinal": self.previous_longitudinal,
            "speed_limit_kmh": self.vehicle.get_speed_limit(),
            "traffic_light_state": _traffic_light_label(self.vehicle),
            "lane_offset_m": float(lane_offset),
            "heading_error_rad": float(heading_error),
            "off_lane": off_lane,
            "collisions": collisions,
            "lane_invasions": lane_invasions,
            "new_collision": bool(new_collision),
            "new_lane_invasions": int(new_lane_invasions),
        }

    def _compute_reward(self, state, steer, longitudinal):
        cfg = self.cfg
        speed_limit_mps = max(state["speed_limit_kmh"] / 3.6, 1e-6)
        speed_term = float(np.clip(state["forward_speed_mps"], 0.0, speed_limit_mps))
        steer_delta = steer - self.previous_steer
        long_delta = longitudinal - self.previous_longitudinal

        reward = (
            cfg.get("w_speed", 1.0) * speed_term
            - cfg.get("w_lane_offset", 1.0) * abs(state["lane_offset_m"])
            - cfg.get("w_heading", 0.5) * abs(state["heading_error_rad"])
            - cfg.get("w_steer_delta", 1.0) * steer_delta ** 2
            - cfg.get("w_long_delta", 0.5) * long_delta ** 2
            - cfg.get("w_yaw_rate", 0.1) * state["yaw_rate_rps"] ** 2
        )

        if state["off_lane"]:
            reward -= cfg.get("off_lane_penalty", 5.0)
            self.off_lane_streak += 1
        else:
            self.off_lane_streak = 0

        if state["new_lane_invasions"]:
            reward -= cfg.get("lane_invasion_penalty", 1.0) * state["new_lane_invasions"]

        terminated = False
        info = {}
        if state["new_collision"]:
            reward -= cfg.get("collision_penalty", 50.0)
            terminated = True
            info["terminate_reason"] = "collision"
        elif self.off_lane_streak >= cfg.get("off_lane_patience_steps", 20):
            terminated = True
            info["terminate_reason"] = "off_lane"

        return float(reward), terminated, info

    def _make_observation(self, state):
        seg = resize_class_map(
            state["seg"],
            self.cfg.get("obs_height", self.cfg.get("height", 128)),
            self.cfg.get("obs_width", self.cfg.get("width", 160)),
        )
        scalar = self.contract.build_scalar_vector(state)
        return {"seg": seg.astype(np.uint8), "scalar": scalar}
