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

from . import cameras, seg_palette

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
        self.last_seg_image = None
        self.last_seg_class_map = None   # (H, W) uint8 raw class-id map — IL/DRL Autopilot input
        self._last_seg_publish = 0.0
        # Bang mau 4 lop cua du an (Background/Road/RoadLine/Sidewalk) — dung cho kenh
        # segmentation tren Live Drive de nguoi xem thay DUNG thu model nhin. Nap mot lan.
        self.seg_color_lut, self.seg_class_names = seg_palette.load_color_lut(
            cfg.deep_rl_carla_root)

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
        # Ha timeout xuong muc "vong lap thuong" sau khi da noi/load xong — xem chu thich
        # cua carla_tick_timeout_s trong config.py.
        self.client.set_timeout(self.cfg.carla_tick_timeout_s)
        logger.info("Connected to CARLA %s:%d | map=%s",
                    self.cfg.carla_host, self.cfg.carla_port, self.map.name)

    def simulator_reachable(self, timeout_s=2.0):
        """CarlaUE4 con song khong? Hoi bang mot lenh RPC re nhat co, voi timeout NGAN.

        Can rieng ham nay vi `world.tick()` dung `carla_tick_timeout_s`: mot lan tick hong
        da ngon may giay, nen lay "tick hong N lan lien tiep" lam dau hieu mat ket noi thi
        phai cho rat lau moi ket luan duoc. Mot lan hoi 2 giay o day tra loi ngay.

        Luon hoi bang mot `carla.Client` MOI TINH, khong dung lai `self.client`. Khi
        CarlaUE4 chet, socket RPC ben trong client cu hong han: no khong tu noi lai khi
        CarlaUE4 bat len tro lai, nen hoi bang client cu thi cau tra loi mai mai la "chua
        song" va Bridge Server se khong bao gio nhan ra CARLA da quay lai.
        """
        try:
            probe = carla.Client(self.cfg.carla_host, self.cfg.carla_port)
            probe.set_timeout(timeout_s)
            probe.get_server_version()
            return True
        except Exception:
            return False

    def reconnect(self):
        """Noi lai tu dau sau khi CarlaUE4 khoi dong lai.

        Moi handle cu (world, map, ego, sensor) deu tro toi mot tien trinh khong con ton
        tai, nen KHONG duoc goi destroy_*() len chung — chi vut di roi connect() lai. Cung
        vi world cu da mat, `_original_settings` phai duoc quen di: settings can khoi phuc
        luc shutdown() la settings cua world MOI.
        """
        self.ego = None
        self.rgb_sensor = None
        self.seg_sensor = None
        self.last_rgb_jpeg = None
        self.last_seg_image = None
        self.last_seg_class_map = None
        self._original_settings = None
        self.connect()

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
        # load_world() mat vai giay — nang timeout len muc "cham that" roi ha lai.
        self.client.set_timeout(self.cfg.carla_timeout_s)
        try:
            self.world = self.client.load_world(town_name)
            self.map = self.world.get_map()
            self._enable_synchronous_mode()
        finally:
            self.client.set_timeout(self.cfg.carla_tick_timeout_s)
        logger.info("Loaded town %s", town_name)

    def set_weather(self, preset_name):
        presets = {
            name: getattr(carla.WeatherParameters, name)
            for name in dir(carla.WeatherParameters) if name and name[0].isupper()
        }
        if preset_name not in presets:
            raise ValueError("Weather preset không hợp lệ: %s" % preset_name)
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
            raise RuntimeError("Bản đồ không có spawn point nào.")
        transform = (spawn_points[spawn_index]
                     if 0 <= spawn_index < len(spawn_points)
                     else random.choice(spawn_points))
        self.ego = self.world.try_spawn_actor(vehicle_bp, transform)
        if self.ego is None:
            raise RuntimeError("Không spawn được xe — spawn point có thể đang bị chiếm.")
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
        # Ve tu `last_seg_class_map` (raw tag) qua bang mau 4 lop cua du an. Chi khi khong
        # nap duoc bang do moi quay ve CityScapesPalette cua CARLA — luc do anh hien thi se
        # khac bo nhan model dung, va seg_palette.py da canh bao ro trong log.
        if self.seg_color_lut is not None:
            jpeg = cameras.class_map_to_png(self.last_seg_class_map, self.seg_color_lut)
        else:
            jpeg = cameras.segmentation_image_to_jpeg(image, self.cfg.jpeg_quality)
        if jpeg:
            self.last_seg_image = jpeg
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
