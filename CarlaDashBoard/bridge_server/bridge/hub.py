"""Connection registry + thread-safe broadcast.

CARLA's client API is synchronous and the sim loop / camera callbacks run on plain Python
threads, while websockets serving runs on the asyncio event loop. `Hub` is the one place that
crosses that boundary: sim-thread code calls the `publish_*` methods (safe from any thread),
which hop onto the event loop via `asyncio.run_coroutine_threadsafe` and then fan the message
out to every currently-connected client.
"""

import asyncio
import logging

logger = logging.getLogger("bridge.hub")


class Hub:
    def __init__(self):
        self.loop = None                # set by server.py once the event loop is running
        self.stream_clients = set()
        self.control_clients = set()

    def bind_loop(self, loop):
        self.loop = loop

    # --- called from asyncio handlers (server.py) ---
    def add_stream_client(self, ws):
        self.stream_clients.add(ws)

    def remove_stream_client(self, ws):
        self.stream_clients.discard(ws)

    def add_control_client(self, ws):
        self.control_clients.add(ws)

    def remove_control_client(self, ws):
        self.control_clients.discard(ws)

    # --- called from ANY thread (sim loop, camera callbacks) ---
    def publish_stream_binary(self, payload: bytes):
        self._schedule(self.stream_clients, payload)

    def publish_stream_text(self, payload: str):
        self._schedule(self.stream_clients, payload)

    def publish_control_text(self, payload: str):
        self._schedule(self.control_clients, payload)

    def _schedule(self, clients, payload):
        # Kiem tra `loop` TRUOC khi tao coroutine. Ban truoc tao `self._broadcast(...)` ngay
        # o cho goi roi moi kiem tra: khi loop chua duoc bind (sim thread bat dau publish
        # telemetry ngay tu tick dau, truoc luc server.py goi bind_loop) thi coroutine do bi
        # vut di khong ai await, va Python in ra "RuntimeWarning: coroutine
        # 'Hub._broadcast' was never awaited" moi tick.
        loop = self.loop
        if loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(clients, payload), loop)

    @staticmethod
    async def _broadcast(clients, payload):
        if not clients:
            return
        stale = []
        for ws in list(clients):
            try:
                await ws.send(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            clients.discard(ws)
