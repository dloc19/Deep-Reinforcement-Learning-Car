"""Tiny CSV logger, shared by `train_ppo.py`, `train_sac.py` and `evaluate.py` so metrics
land in a consistent shape regardless of algorithm — convenient for plotting later.
"""

import csv
from pathlib import Path


class CsvLogger(object):
    def __init__(self, path, fieldnames, mode="a"):
        """`mode="a"` (default, dung cho train_ppo.py/train_sac.py): noi tiep vao file cu
        neu co — bat buoc de log khong bi mat khi resume training tu checkpoint giua chung.
        `mode="w"`: LUON ghi de file cu tu dau — dung cho evaluate.py, vi moi lan chay la
        MOT lo danh gia doc lap (N episode tren MOT checkpoint); noi tiep vao file cu se
        am tham tron episode cua nhieu lan danh gia khac nhau thanh 1 "run" khi ve bieu do
        (xem plot_metrics.py::plot_eval_comparison)."""
        if mode not in ("a", "w"):
            raise ValueError("mode phai la 'a' hoac 'w', nhan duoc '%r'" % mode)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = mode == "w" or not self.path.exists()
        self.handle = self.path.open(mode, newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=fieldnames)
        if is_new:
            self.writer.writeheader()
            self.handle.flush()

    def log(self, row):
        self.writer.writerow(row)
        self.handle.flush()

    def close(self):
        self.handle.close()
