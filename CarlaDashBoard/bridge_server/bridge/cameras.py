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


def camera_blueprint(world, type_id, cfg, width=None, height=None, sensor_tick=None):
    """`sensor_tick=0.0` = fire on EVERY world tick.

    That is what the segmentation camera needs: it is the policy's image input, and in
    synchronous mode a `sensor_tick` coarser than `fixed_delta_seconds` makes the sensor skip
    ticks, so `session.last_seg_class_map` would be a frame the car has already driven past.
    The RGB camera is preview-only and keeps the publish-rate tick.
    """
    bp = world.get_blueprint_library().find(type_id)
    bp.set_attribute("image_size_x", str(width if width else cfg.camera_width))
    bp.set_attribute("image_size_y", str(height if height else cfg.camera_height))
    bp.set_attribute("fov", str(cfg.camera_fov))
    bp.set_attribute("sensor_tick",
                     str(1.0 / cfg.publish_fps if sensor_tick is None else sensor_tick))
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


# Anh segmentation di bang PNG, KHONG phai JPEG. Day la anh chi so lop (4 mau phang), khong
# phai anh chup: JPEG luong tu hoa theo khoi 8x8 + lay mau chroma thua, nen no bien mot vung
# mau phang thanh mot dam mau xap xi. Do tren mot khung Town03 that: sau JPEG q75 chi 37%
# pixel con TRUNG DUNG bang mau, va ca ba lop Road/RoadLine/Sidewalk deu con 0% pixel dung
# mau — rieng RoadLine chi rong 1-3 px nen bi vien JPEG an gan het. PNG vua khong mat mau
# vua NHO HON tren loai anh nay: 16.8 KB so voi 26.1 KB cua JPEG q75 tren cung khung do.
# Phia WPF khong can sua gi: BitmapImage tu nhan dang dinh dang tu chinh luong byte.
PNG_COMPRESSION = 6      # 6 vs 9: nho hon 0.7 KB nua nhung ton gap doi CPU moi khung


def class_map_to_png(class_map, color_lut):
    """Ma hoa mot ban do class-id THO (raw tag CARLA) thanh PNG de hien thi.

    `color_lut` la bang (256, 3) RGB tu `seg_palette.load_color_lut()` — bang 4 lop
    (Background/Road/RoadLine/Sidewalk) ma model that su nhin thay.

    Ve tu `class_map` chu khong tu `image` con co mot cai loi phu: khong phai goi
    `image.convert()`, ma convert() ghi de raw_data TAI CHO — dung chung mot doi tuong anh
    voi duong lay `last_seg_class_map` cua policy.
    """
    if color_lut is None:
        return None
    rgb = color_lut[class_map]                       # (H, W, 3) RGB
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])       # cv2 ghi theo thu tu BGR
    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    return buf.tobytes() if ok else None


def segmentation_image_to_jpeg(image, quality):
    """Duong du phong: bang mau CityScapes cua chinh CARLA. Chi dung khi khong nap duoc bang
    mau 4 lop cua du an (xem seg_palette.py)."""
    # convert() mutates raw_data in place (class id -> CityScapes RGB palette).
    image.convert(carla.ColorConverter.CityScapesPalette)
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    bgr = bgra[:, :, :3]
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None
