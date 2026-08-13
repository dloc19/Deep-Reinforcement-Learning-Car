"""Join asynchronous camera and vehicle-state packets by CARLA frame."""

import queue
import threading
from collections import OrderedDict


class FrameSynchronizer:
    def __init__(self, output_queue, required_keys, max_pending=512):
        self.output_queue = output_queue
        self.required_keys = tuple(required_keys)
        self.max_pending = max_pending
        self.lock = threading.Lock()
        self.pending = OrderedDict()
        self.dropped = 0

    def put(self, frame, key, value):
        with self.lock:
            packet = self.pending.setdefault(frame, {})
            packet[key] = value
            if all(name in packet for name in self.required_keys):
                complete = self.pending.pop(frame)
                try:
                    self.output_queue.put_nowait(complete)
                except queue.Full:
                    self.dropped += 1
            while len(self.pending) > self.max_pending:
                self.pending.popitem(last=False)
                self.dropped += 1
