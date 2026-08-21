"""Common lifecycle every mode runtime follows (design doc §07's "vì sao dùng chung một khái
niệm Mode Runtime" callout): start() once when selected, tick() once per sim frame, stop()
once when switched away from or the session ends. SimLoop (sim_loop.py) is the only caller.
"""


class ModeRuntime:
    name = "BASE"

    def start(self, session):
        """`session` is the CarlaSession — already has an ego vehicle + cameras spawned."""

    def tick(self, snapshot):
        """Called once per world.tick(). Return a carla.VehicleControl to apply, or None to
        leave control alone (e.g. because CARLA's own Traffic Manager is already driving)."""
        return None

    def stop(self):
        """Release anything the mode owns (sensors, files, controllers). Must not raise."""

    def status_extra(self):
        """Extra fields merged into the telemetry envelope (see protocol/sim_loop)."""
        return {}
