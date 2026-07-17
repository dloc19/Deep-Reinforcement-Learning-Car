"""Spawn and clean up cameras/event sensors owned by the collector."""

import carla


class SensorSuite:
    def __init__(self, world, ego, args, synchronizer, events):
        self.world = world
        self.ego = ego
        self.args = args
        self.synchronizer = synchronizer
        self.events = events
        self.actors = []

    def camera_blueprint(self, type_id):
        bp = self.world.get_blueprint_library().find(type_id)
        bp.set_attribute("image_size_x", str(self.args.width))
        bp.set_attribute("image_size_y", str(self.args.height))
        bp.set_attribute("fov", str(self.args.fov))
        bp.set_attribute("sensor_tick", str(1.0 / self.args.fps))
        return bp

    def spawn(self):
        camera_tf = carla.Transform(
            carla.Location(
                x=self.args.camera_x, y=self.args.camera_y, z=self.args.camera_z),
            carla.Rotation(pitch=self.args.camera_pitch),
        )
        seg = self.world.spawn_actor(
            self.camera_blueprint("sensor.camera.semantic_segmentation"),
            camera_tf, attach_to=self.ego, attachment_type=carla.AttachmentType.Rigid)
        self.actors.append(seg)
        seg.listen(lambda image: self.synchronizer.put(
            image.frame, "seg", (bytes(image.raw_data), image.width, image.height)))

        if self.args.image_mode == "seg-rgb":
            rgb = self.world.spawn_actor(
                self.camera_blueprint("sensor.camera.rgb"),
                camera_tf, attach_to=self.ego, attachment_type=carla.AttachmentType.Rigid)
            self.actors.append(rgb)
            rgb.listen(lambda image: self.synchronizer.put(
                image.frame, "rgb", (bytes(image.raw_data), image.width, image.height)))

        if not self.args.no_event_sensors:
            collision = self.world.spawn_actor(
                self.world.get_blueprint_library().find("sensor.other.collision"),
                carla.Transform(), attach_to=self.ego)
            invasion = self.world.spawn_actor(
                self.world.get_blueprint_library().find("sensor.other.lane_invasion"),
                carla.Transform(), attach_to=self.ego)
            collision.listen(self.events.collision)
            invasion.listen(self.events.lane_invasion)
            self.actors.extend([collision, invasion])

    def destroy(self):
        for sensor in self.actors:
            try:
                sensor.stop()
            except (AttributeError, RuntimeError):
                pass
        for sensor in self.actors:
            try:
                if sensor.is_alive:
                    sensor.destroy()
            except RuntimeError:
                pass
