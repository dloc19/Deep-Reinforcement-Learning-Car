"""Destination selection — reuses the exact CLI convention already established in
`data_collection/carla_collector/config.py` / `map_export.py::MapArtifacts.resolve_goal`
(`--goal-spawn-index` / `--goal-x`/`--goal-y`/`--goal-z`), plus an interactive fallback (list
every spawn point, prompt for an index) so a destination can be picked without knowing world
coordinates ahead of time.
"""

try:
    import carla
except ImportError as exc:
    raise ImportError(
        "Khong import duoc module 'carla'. Cai CARLA 0.9.10 Python API truoc (xem "
        "docs/manual_thu_thap_du_lieu.md muc 2)."
    ) from exc


def list_spawn_points(world_map, limit=None):
    """Print every spawn point with its index and (x, y, z, yaw) — used by both
    `--goal-interactive` and standalone inspection
    (`python -c "..."` or a small REPL snippet, see docs/manual_dieu_huong_astar.md)."""
    spawn_points = world_map.get_spawn_points()
    for index, transform in enumerate(spawn_points):
        if limit is not None and index >= limit:
            print("... con %d diem nua (bo --limit de xem het)" % (len(spawn_points) - limit))
            break
        loc = transform.location
        print("[%3d] x=%8.2f y=%8.2f z=%6.2f yaw=%7.2f" % (
            index, loc.x, loc.y, loc.z, transform.rotation.yaw))
    return spawn_points


def resolve_goal(world_map, args):
    """Return a `carla.Waypoint` for the destination the caller asked for, or `None` if
    nothing was selected (no goal flag AND `--goal-interactive` not set) — caller decides
    whether that is an error.

    `args` only needs the attributes `goal_spawn_index`, `goal_x`, `goal_y`, `goal_z`,
    `goal_interactive` — an `argparse.Namespace` from `drive_to_goal.py` satisfies this, but
    any object with those attributes works (e.g. a small config object in a notebook); every
    attribute is read through `getattr()` with a safe default, so a caller may omit any of
    them entirely. Raises `ValueError` if more than one goal-selection method is set at once
    (mirrors the mutual-exclusivity check `drive_to_goal.py`'s CLI already enforces via
    `parser.error()`, but this function has no parser to rely on for duck-typed callers).
    """
    spawn_points = world_map.get_spawn_points()

    goal_spawn_index = getattr(args, "goal_spawn_index", -1)
    if goal_spawn_index is None:
        goal_spawn_index = -1
    goal_x = getattr(args, "goal_x", None)
    goal_y = getattr(args, "goal_y", None)
    goal_interactive = getattr(args, "goal_interactive", False)

    has_spawn_index = goal_spawn_index >= 0
    has_xy = goal_x is not None or goal_y is not None
    if has_spawn_index and has_xy:
        raise ValueError(
            "Chi duoc chon MOT trong hai: goal_spawn_index hoac goal_x/goal_y, "
            "khong duoc dat ca hai cung luc.")
    if has_xy and (goal_x is None or goal_y is None):
        raise ValueError("Phai truyen dong thoi goal_x va goal_y.")

    if has_spawn_index:
        if goal_spawn_index >= len(spawn_points):
            raise ValueError(
                "--goal-spawn-index=%d khong hop le; map co %d spawn points" %
                (goal_spawn_index, len(spawn_points)))
        location = spawn_points[goal_spawn_index].location
    elif has_xy:
        location = carla.Location(x=goal_x, y=goal_y, z=getattr(args, "goal_z", 0.0) or 0.0)
    elif goal_interactive:
        print("Chon diem den — cac spawn point co san:")
        list_spawn_points(world_map)
        raw = input("Nhap index spawn point lam diem den: ").strip()
        try:
            index = int(raw)
        except ValueError:
            raise ValueError("'%s' khong phai so nguyen hop le." % raw)
        if index < 0 or index >= len(spawn_points):
            raise ValueError("Index %d khong hop le (0..%d)" % (index, len(spawn_points) - 1))
        location = spawn_points[index].location
    else:
        return None

    goal_waypoint = world_map.get_waypoint(
        location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if goal_waypoint is None:
        raise RuntimeError(
            "Khong chieu duoc diem dich (%.1f, %.1f) len Driving waypoint — chon diem khac gan duong hon." %
            (location.x, location.y))
    return goal_waypoint
