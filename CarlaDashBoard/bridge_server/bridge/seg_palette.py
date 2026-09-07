"""Bang mau segmentation cho kenh /stream — Phase 5.

Man hinh Live Drive phai ve DUNG bo nhan ma model dang nhin, chu khong phai bo nhan mac
dinh cua CARLA. Du an nay gap 23 raw tag cua CARLA xuong 4 lop
(`Background / Road / RoadLine / Sidewalk`, xem `SEG_CLASS_NAMES` trong
`data_collection/carla_collector/schema.py`), va CA BA noi dung anh segmentation deu theo
bang do: `train-segment-lane.ipynb`, `train_il.ipynb`, va `policy/observation.py::resize_class_map`
— tuc chinh cai ma IL/DRL Autopilot an vao model moi tick.

Truoc day bridge lai ve kenh nay bang `carla.ColorConverter.CityScapesPalette`, tuc bang mau
13+ lop cua CARLA: nguoi xem thay cay xanh la rieng, nha rieng, xe rieng, troi rieng — trong
khi model chi thay ba thu ("duong", "vach ke", "via he") va gop TAT CA phan con lai vao
Background. Anh tren man hinh vi vay khong phai anh model nhin, va mot demo dung no de giai
thich "model dang thay gi" la giai thich sai.

`SEG_COLOR_LUT` trong schema.py da la mot bang tra 256x3 dung san (raw tag -> mau cua lop
huan luyen), chinh la bang `data_collection` dung de ghi `seg_color/*.png`. Module nay chi
lo dua no vao duoc sys.path, khong dinh nghia lai bang mau lan thu hai — dung ly do da ghi
trong schema.py: bang nay la nguon su that, moi ban sao them la mot cho co the lech.
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger("bridge.seg_palette")

_BRIDGE_SERVER_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DEEP_RL_CARLA_ROOT = _BRIDGE_SERVER_ROOT.parent.parent / "Deep_RL_Carla"

_cache = {}


def load_color_lut(deep_rl_carla_root: str = ""):
    """Tra ve `(lut, class_names)`; `lut` la ndarray (256, 3) RGB tra theo RAW tag cua CARLA.

    Tra ve `(None, None)` neu khong nap duoc schema — goi y de `cameras.py` quay ve bang mau
    CityScapes cua CARLA thay vi lam sap ca luong camera chi vi thieu repo anh em.
    """
    key = deep_rl_carla_root or "<default>"
    if key in _cache:
        return _cache[key]

    root = Path(deep_rl_carla_root) if deep_rl_carla_root else _DEFAULT_DEEP_RL_CARLA_ROOT
    data_collection_dir = root.resolve() / "data_collection"
    result = (None, None)
    try:
        if str(data_collection_dir) not in sys.path:
            sys.path.insert(0, str(data_collection_dir))
        from carla_collector.schema import SEG_CLASS_NAMES, SEG_COLOR_LUT
        result = (SEG_COLOR_LUT, list(SEG_CLASS_NAMES))
        logger.info("Kenh segmentation dung bang mau %d lop cua du an: %s",
                    len(SEG_CLASS_NAMES), ", ".join(SEG_CLASS_NAMES))
    except Exception as exc:
        logger.warning(
            "Khong nap duoc bang mau 4 lop tu '%s' (%s) — kenh segmentation se ve bang "
            "CityScapesPalette cua CARLA. Anh hien ra se KHAC bo nhan model dang dung.",
            data_collection_dir, exc)

    _cache[key] = result
    return result
