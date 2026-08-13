"""Tiny append-mode CSV logger, shared by `train_ppo.py` and `train_sac.py` so training
metrics land in the same `episode_log.csv` / `update_log.csv` shape regardless of algorithm
— convenient for plotting both runs with the same notebook/script later.
"""

import csv
from pathlib import Path


class CsvLogger(object):
    def __init__(self, path, fieldnames):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists()
        self.handle = self.path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=fieldnames)
        if is_new:
            self.writer.writeheader()
            self.handle.flush()

    def log(self, row):
        self.writer.writerow(row)
        self.handle.flush()

    def close(self):
        self.handle.close()
