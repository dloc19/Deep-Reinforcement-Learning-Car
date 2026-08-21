#!/usr/bin/env python3
"""Doi chieu bang nhan segmentation giua BA noi dinh nghia no.

    1. data_collection/carla_collector/schema.py   -> RAW_TO_TRAIN_LANE_LUT  (nguon su that)
    2. behavior_cloning/carla_seg.ipynb            -> LABEL_LUT
    3. behavior_cloning/train_il.ipynb             -> SEG_LABEL_LUT

Hai notebook chay tren Kaggle nen KHONG import duoc `schema.py`; chung buoc phai giu mot
ban chep tay cua cung mot bang. Do la loi da xay ra that trong du an nay: seg xuat
`Background=0, Road=1, ...` con IL doc `Road=0, RoadLine=1, ...`, shape van khop nen
`load_state_dict` khong bao gi ca — model chi don gian nhan sai kenh va lai sai.

Script nay doc THANG source cell cua hai notebook (khong can chay Kaggle, khong can GPU,
khong can dataset) roi so tung phan tu LUT. Chay truoc moi lan train lai:

    python behavior_cloning/check_label_contract.py

Exit code 0 = ba bang khop nhau, 1 = lech (in ra dung raw tag nao lech).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "data_collection"))

from carla_collector.schema import (  # noqa: E402  (sys.path phai duoc set truoc)
    NUM_SEG_CLASSES, RAW_TO_TRAIN_LANE_LUT, SEG_CLASS_NAMES)

MAX_RAW_TAG = 22          # CARLA 0.9.10 dinh nghia raw tag 0..22
SEG_NB = REPO_ROOT / "behavior_cloning" / "carla_seg.ipynb"
IL_NB = REPO_ROOT / "behavior_cloning" / "train_il.ipynb"


def cell_sources(path):
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    return ["".join(cell["source"]) for cell in notebook["cells"]]


def find_cell(sources, needle):
    for src in sources:
        if needle in src:
            return src
    raise SystemExit("Khong tim thay cell chua '%s' trong notebook." % needle)


def seg_notebook_label_space():
    """Chay lai dung logic dinh nghia label space cua carla_seg.ipynb.

    Khong exec ca notebook (no can dataset + GPU): chi exec ba cell thuan tinh toan —
    cau hinh (§1), bang tag CARLA (§1) va §3 (build_label_space).
    """
    sources = cell_sources(SEG_NB)
    config = find_cell(sources, "TASK_SCHEME = ")
    namespace = {"np": np}
    for name in ("TASK_SCHEME", "RAILTRACK_AS_ROAD", "SKY_AS_CLASS",
                 "IGNORE_UNLABELED", "IGNORE_INDEX"):
        match = re.search(r"^%s\s*=\s*(.+?)\s*(?:#.*)?$" % name, config, re.M)
        if match is None:
            raise SystemExit("carla_seg.ipynb: khong tim thay bien %s" % name)
        namespace[name] = eval(match.group(1))  # gia tri hang: str/bool/int
    exec(find_cell(sources, "CARLA_0910_TAGS = {"), namespace)
    exec(find_cell(sources, "def build_label_space("), namespace)
    return namespace["CLASS_NAMES"], namespace["LABEL_LUT"], namespace["TASK_SCHEME"]


def il_notebook_label_space():
    """Doc CLASS_NAMES + RAW_TO_TRAIN cua train_il.ipynb va dung lai LUT tu do."""
    config = find_cell(cell_sources(IL_NB), "RAW_TO_TRAIN = {")
    names_match = re.search(r"^CLASS_NAMES\s*=\s*(\[[^\]]*\])", config, re.M)
    table_match = re.search(r"^RAW_TO_TRAIN\s*=\s*(\{[^}]*\})", config, re.M)
    if not names_match or not table_match:
        raise SystemExit("train_il.ipynb: khong doc duoc CLASS_NAMES / RAW_TO_TRAIN")
    class_names = eval(names_match.group(1))
    lut = np.zeros(256, dtype=np.uint8)
    for raw, train in eval(table_match.group(1)).items():
        lut[raw] = train
    return class_names, lut


def compare(label_a, names_a, lut_a, label_b, names_b, lut_b):
    problems = []
    if list(names_a) != list(names_b):
        problems.append("  ten/thu tu lop lech:\n    %s: %s\n    %s: %s"
                        % (label_a, names_a, label_b, names_b))
    diff = [(raw, int(lut_a[raw]), int(lut_b[raw])) for raw in range(MAX_RAW_TAG + 1)
            if int(lut_a[raw]) != int(lut_b[raw])]
    if diff:
        problems.append("  LUT lech tai (raw tag, %s, %s): %s" % (label_a, label_b, diff))
    return problems


def main():
    seg_names, seg_lut, task_scheme = seg_notebook_label_space()
    il_names, il_lut = il_notebook_label_space()

    print("schema.py        : %d lop %s" % (NUM_SEG_CLASSES, SEG_CLASS_NAMES))
    print("carla_seg.ipynb  : %d lop %s  (TASK_SCHEME=%s)"
          % (len(seg_names), seg_names, task_scheme))
    print("train_il.ipynb   : %d lop %s" % (len(il_names), il_names))
    print()
    print("raw tag -> train id:")
    for raw in range(MAX_RAW_TAG + 1):
        train_id = int(RAW_TO_TRAIN_LANE_LUT[raw])
        print("  raw %2d -> %d %s" % (raw, train_id, SEG_CLASS_NAMES[train_id]))
    print()

    problems = []
    problems += compare("schema", SEG_CLASS_NAMES, RAW_TO_TRAIN_LANE_LUT,
                        "seg_nb", seg_names, seg_lut)
    problems += compare("schema", SEG_CLASS_NAMES, RAW_TO_TRAIN_LANE_LUT,
                        "il_nb", il_names, il_lut)

    if problems:
        print("LECH BANG NHAN:")
        print("\n".join(problems))
        print("\nSua theo huong: schema.py la nguon su that; chep lai vao hai notebook.")
        return 1

    print("OK: ca ba noi dung chung mot bang nhan (%d lop)." % NUM_SEG_CLASSES)
    print("Luu y: §3b cua carla_seg.ipynb con co the prune them lop hiem luc chay that;")
    print("khi do so lop trong checkpoint se nho hon o day va §2b cua train_il.ipynb se bao.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
