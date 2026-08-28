"""Owns the single active CARLA client connection: world, ego vehicle, the rgb+seg camera pair.

Everything here runs on the sim thread (see sim_loop.py) — CarlaSession itself does not touch
the network/asyncio side, it only calls back into `hub.publish_stream_binary(...)` (thread-safe)
whenever a camera frame is ready.
"""

import logging
import random
import time

import carla
import numpy as np

from . import cameras

logger = logging.getLogger("bridge.carla_session")


class CarlaSession:
    def __init__(self, cfg, hub):
        self.cfg = cfg
        self.hub = hub
        self.client = None
        self.world = None
        self.map = None
        self.ego = None
        self.rgb_sensor = None
        self.seg_sensor = None
        self._original_settings = None
        self.last_rgb_jpeg = None
        self.last_seg_jpeg = None
        self.last_seg_class_map = None   # (H, W) uint8 raw class-id map — IL/DRL Autopilot input
        self._last_seg_publish = 0.0

    # ---------------------------------------------------------------- connect
    def connect(self):
        self.client = carla.Client(self.cfg.carla_host, self.cfg.carla_port)
        self.client.set_timeout(self.cfg.carla_timeout_s)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        if self.cfg.town and not self.map.name.endswith(self.cfg.town):
            self.load_town(self.cfg.town)
        else:
            self._enable_synchronous_mode()
        logger.info("Connected to CARLA %s:%d | map=%s",
                    self.cfg.carla_host, self.cfg.carla_port, self.map.name)

    def _enable_synchronous_mode(self):
        settings = self.world.get_settings()
        if self._original_settings is None:
            # `or 0.0`: khi world dang chay bat dong bo (mac dinh), CARLA 0.9.10 tra ve
            # fixed_delta_seconds = None, nhung constructor WorldSettings cua no chi nhan
            # `double` — truyen None vao la Boost.Python.ArgumentError ngay luc connect().
            # 0.0 chinh la gia tri "buoc thoi gian bien thien", tuc dung y nghia can khoi
            # phuc luc shutdown().
            self._original_settings = carla.WorldSettings(
                synchronous_mode=settings.synchronous_mode,
                fixed_delta_seconds=settings.fixed_delta_seconds or 0.0)
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.cfg.sim_fps
        self.world.apply_settings(settings)

    def shutdown(self):
        self.destroy_cameras()
        self.destroy_ego()
        if self.world is not None and self._original_settings is not None:
            try:
                self.world.apply_settings(self._original_settings)
            except RuntimeError:
                pass

    # ---------------------------------------------------------------- world / town / weather
    def available_towns(self):
        return sorted({name.split("/")[-1] for name in self.client.get_available_maps()})

    def load_town(self, town_name):
        self.destroy_cameras()
        self.destroy_ego()
        self.world = self.client.load_world(town_name)
        self.map = self.world.get_map()
        self._enable_synchronous_mode()
        logger.info("Loaded town %s", town_name)

    def set_weather(self, preset_name):
        presets = {
            name: getattr(carla.WeatherParameters, name)
            for name in dir(carla.WeatherParameters) if name and name[0].isupper()
        }
        if preset_name not in presets:
            raise ValueError("Weather preset khong hop le: %s" % preset_name)
        self.world.set_weather(presets[preset_name])

    def current_town_short(self):
        return self.map.name.split("/")[-1] if self.map else ""

    # ---------------------------------------------------------------- ego vehicle
    def spawn_ego(self, spawn_index=-1):
        self.destroy_cameras()
        self.destroy_ego()
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(self.cfg.vehicle_filter)[0]
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", self.cfg.role_name)
        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("Map khong co spawn point nao.")
        transform = (spawn_points[spawn_index]
                     if 0 <= spawn_index < len(spawn_points)
                     else random.choice(spawn_points))
        self.ego = self.world.try_spawn_actor(vehicle_bp, transform)
        if self.ego is None:
            raise RuntimeError("Khong spawn duoc xe — spawn point co the dang bi chiem.")
        self.world.tick()
        self.spawn_cameras()
        return self.ego

    def destroy_ego(self):
        if self.ego is not None:
            try:
                if self.ego.is_alive:
                    self.ego.destroy()
            except RuntimeError:
                pass
            self.ego = None

    # ---------------------------------------------------------------- cameras
    def spawn_cameras(self):
        if self.ego is None:
            return
        transform = cameras.camera_transform(self.cfg)

        rgb_bp = cameras.camera_blueprint(self.world, "sensor.camera.rgb", self.cfg)
        self.rgb_sensor = self.world.spawn_actor(
            rgb_bp, transform, attach_to=self.ego, attachment_type=carla.AttachmentType.Rigid)
        self.rgb_sensor.listen(self._on_rgb_frame)

        seg_bp = cameras.camera_blueprint(
            self.world, "sensor.camera.semantic_segmentation", self.cfg,
            width=self.cfg.seg_width, height=self.cfg.seg_height, sensor_tick=0.0)
        self.seg_sensor = self.world.spawn_actor(
            seg_bp, transform, attach_to=self.ego, attachment_type=carla.AttachmentType.Rigid)
        self.seg_sensor.listen(self._on_seg_frame)

    def destroy_cameras(self):
        for sensor in (self.rgb_sensor, self.seg_sensor):
            if sensor is None:
                continue
            try:
                sensor.stop()
                if sensor.is_alive:
                    sensor.destroy()
            except RuntimeError:
                pass
        self.rgb_sensor = None
        self.seg_sensor = None

    def _on_rgb_frame(self, image):
        from . import protocol
        jpeg = cameras.rgb_image_to_jpeg(image, self.cfg.jpeg_quality)
        if jpeg:
            self.last_rgb_jpeg = jpeg
            self.hub.publish_stream_binary(
                protocol.encode_binary_frame(protocol.CHANNEL_RGB, jpeg))

    def _on_seg_frame(self, image):
        from . import protocol
        # Raw class-id (red channel of the BGRA buffer) must be read out BEFORE
        # segmentation_image_to_jpeg() below, which calls image.convert() and mutates
        # raw_data in place (class id -> CityScapes RGB) — same convention as
        # data_collection/carla_collector/writer.py and drl_training's own camera callback.
        bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        self.last_seg_class_map = bgra[:, :, 2].copy()

        # Hai nhip KHAC NHAU cho cung mot camera:
        #   - `last_seg_class_map` cap nhat MOI tick, vi day la dau vao cua policy (camera
        #     nay dat sensor_tick=0.0 chinh vi the).
        #   - JPEG chi ma hoa + gui o `publish_fps`. Truoc day moi tick deu encode va gui,
        #     tuc ~175 khung/giay do vao WebSocket cho mot khung hinh chi de NGUOI xem.
        now = time.time()
        if now - self._last_seg_publish < 1.0 / self.cfg.publish_fps:
            return
        self._last_seg_publish = now
        jpeg = cameras.segmentation_image_to_jpeg(image, self.cfg.jpeg_quality)
        if jpeg:
            self.last_seg_jpeg = jpeg
            self.hub.publish_stream_binary(
                protocol.encode_binary_frame(protocol.CHANNEL_SEG, jpeg))

    def apply_camera_params(self, width=None, height=None, fov=None, fps=None):
        # Only the RGB (preview) camera is resized from the UI. The segmentation camera is
        # pinned to cfg.seg_width/seg_height because it is the policy's input — see the
        # comment on those fields in config.py. Changing fov from the UI DOES affect it, and
        # will silently degrade IL/DRL Autopilot; the IL contract assumes fov 90.
        if width:
            self.cfg.camera_width = width
        if height:
            self.cfg.camera_height = height
        if fov:
            self.cfg.camera_fov = fov
        if fps:
            self.cfg.publish_fps = fps
        if self.ego is not None:
            self.destroy_cameras()
            self.spawn_cameras()
