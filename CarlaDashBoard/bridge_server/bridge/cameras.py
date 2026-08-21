"""Camera blueprint setup + CARLA image -> JPEG encoding.

Buffer layout follows the same convention already used in data_collection/carla_collector
(writer.py): `image.raw_data` is BGRA, so a straight `[:, :, :3]` slice is already the BGR
order cv2 expects — no channel swap needed for the RGB camera.

The segmentation camera is converted with CARLA's own CityScapesPalette (the same 13-class
palette documented in docs/csv_fields_by_task.md) so the stream is human-viewable straight
away, rather than shipping raw class-id values that would render as a near-black image.
"""

import carla
import cv2
import numpy as np


def camera_blueprint(world, type_id, cfg):
    bp = world.get_blueprint_library().find(type_id)
    bp.set_attribute("image_size_x", str(cfg.camera_width))
    bp.set_attribute("image_size_y", str(cfg.camera_height))
    bp.set_attribute("fov", str(cfg.camera_fov))
    bp.set_attribute("sensor_tick", str(1.0 / cfg.publish_fps))
    return bp


def camera_transform(cfg):
    return carla.Transform(
        carla.Location(x=cfg.camera_x, y=cfg.camera_y, z=cfg.camera_z),
        carla.Rotation(pitch=cfg.camera_pitch),
    )


def rgb_image_to_jpeg(image, quality):
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    bgr = bgra[:, :, :3]
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def segmentation_image_to_jpeg(image, quality):
    # convert() mutates raw_data in place (class id -> CityScapes RGB palette).
    image.convert(carla.ColorConverter.CityScapesPalette)
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    bgr = bgra[:, :, :3]
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None
