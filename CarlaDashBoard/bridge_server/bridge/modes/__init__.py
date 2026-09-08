from .base import ModeRuntime
from .idle import IdleMode
from .data_collection import DataCollectionMode
from .astar_autopilot import AstarAutopilotMode, RouteContext
from .learned_autopilot import LearnedAutopilotMode
from .route_learned_autopilot import RouteLearnedAutopilotMode

from .. import protocol


def build(mode_name, cfg):
    """Factory for the modes that need nothing beyond `cfg` (protocol.MODE_* -> ModeRuntime).
    ASTAR_AUTOPILOT, IL_AUTOPILOT, DRL_AUTOPILOT and ROUTE_DRL_AUTOPILOT are NOT here — each
    needs something built ahead of time (a RouteContext from a prior SetDestination, or a
    loaded+cached model, or both), so sim_loop constructs those directly instead of going
    through this factory."""
    if mode_name == protocol.MODE_DATA_COLLECTION:
        return DataCollectionMode(cfg)
    if mode_name == protocol.MODE_IDLE:
        return IdleMode()
    raise ValueError("Mode không hợp lệ: %s" % mode_name)
