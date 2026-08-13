"""Thread-safe counters for collision and lane-invasion sensors."""

import threading


class EventCounters:
    def __init__(self):
        self._lock = threading.Lock()
        self.collisions = 0
        self.lane_invasions = 0

    def collision(self, _event):
        with self._lock:
            self.collisions += 1

    def lane_invasion(self, _event):
        with self._lock:
            self.lane_invasions += 1

    def snapshot(self):
        with self._lock:
            return self.collisions, self.lane_invasions
