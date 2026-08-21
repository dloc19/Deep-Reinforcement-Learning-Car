"""Bridges to Deep_RL_Carla/router_plan — Phase 3 (design doc §04/§07: A* Autopilot).

router_plan is a sibling repo folder (`Do_An/Deep_RL_Carla/router_plan`), not part of
bridge_server, and its own modules only add THEIR OWN parent dir to sys.path once they are
already importable — so Deep_RL_Carla itself has to be on sys.path before the very first
`import router_plan...` runs. That's all this module does; everything else re-exports
router_plan's own classes unchanged.
"""

import sys
from pathlib import Path

_BRIDGE_SERVER_ROOT = Path(__file__).resolve().parents[1]          # .../CarlaDashBoard/bridge_server
_DEFAULT_DEEP_RL_CARLA_ROOT = _BRIDGE_SERVER_ROOT.parent.parent / "Deep_RL_Carla"  # .../Do_An/Deep_RL_Carla


def ensure_on_path(deep_rl_carla_root: str = ""):
    root = Path(deep_rl_carla_root) if deep_rl_carla_root else _DEFAULT_DEEP_RL_CARLA_ROOT
    root = root.resolve()
    if not (root / "router_plan").is_dir():
        raise RuntimeError(
            "Khong tim thay 'router_plan' trong %s — truyen --deep-rl-carla-root neu "
            "Deep_RL_Carla khong nam canh CarlaDashBoard." % root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load(deep_rl_carla_root: str = ""):
    """Import + return router_plan's public classes as a small namespace. Deferred to call
    time (not module import time) so BridgeConfig.deep_rl_carla_root can override the path
    first."""
    ensure_on_path(deep_rl_carla_root)
    from router_plan.Global_Route_Planner import GlobalRoutePlanner, RouteNotFoundError
    from router_plan.controller import RoutePurePursuitController

    class _RouterPlan:
        pass

    ns = _RouterPlan()
    ns.GlobalRoutePlanner = GlobalRoutePlanner
    ns.RouteNotFoundError = RouteNotFoundError
    ns.RoutePurePursuitController = RoutePurePursuitController
    return ns
