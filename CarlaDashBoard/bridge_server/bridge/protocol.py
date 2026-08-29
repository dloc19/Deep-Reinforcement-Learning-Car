"""Wire format shared by /stream and /control — matches design doc §02 ("Carla Dashboard" artifact).

Binary frames on /stream are [1 channel-tag byte][JPEG bytes]; everything else (telemetry,
commands, events) is a single JSON object with a "type" field.
"""

import json

# --- /stream binary channel tags ---
CHANNEL_RGB = 0x01
CHANNEL_SEG = 0x02

# --- mode identifiers (SetMode.mode) ---
MODE_IDLE = "IDLE"
MODE_DATA_COLLECTION = "DATA_COLLECTION"
MODE_ASTAR_AUTOPILOT = "ASTAR_AUTOPILOT"     # Phase 3 — wraps router_plan's RoutePurePursuitController
MODE_IL_AUTOPILOT = "IL_AUTOPILOT"           # Phase 4 — wraps a trained IL checkpoint (see il_drl_bridge.py)
MODE_DRL_AUTOPILOT = "DRL_AUTOPILOT"         # Phase 4 — wraps a trained PPO/SAC checkpoint
# Phase 5 — lai theo tuyen A* NHUNG bam lan bang policy da hoc. Can CA HAI: mot RouteContext
# tu SetDestination truoc do, va checkpoint DRL. Xem modes/route_learned_autopilot.py de biet
# vi sao phai lai hai bo dieu khien thay vi de policy tu di het tuyen.
MODE_ROUTE_DRL_AUTOPILOT = "ROUTE_DRL_AUTOPILOT"

KNOWN_MODES = (
    MODE_IDLE, MODE_DATA_COLLECTION, MODE_ASTAR_AUTOPILOT,
    MODE_IL_AUTOPILOT, MODE_DRL_AUTOPILOT, MODE_ROUTE_DRL_AUTOPILOT,
)


def encode_binary_frame(channel_tag, jpeg_bytes):
    return bytes([channel_tag]) + jpeg_bytes


def dumps(message_type, **fields):
    payload = {"type": message_type}
    payload.update(fields)
    return json.dumps(payload)


def error(code, message):
    return dumps("Error", code=code, message=message)
