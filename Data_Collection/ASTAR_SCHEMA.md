# A* data contract

Pipeline separates global planning from local vehicle control.

## Static map inputs

`map_nodes.csv` contains the searchable vertices. `node_id` is the stable key
used by edges and route outputs. OpenDRIVE coordinates are retained as
`road_id`, `section_id`, `lane_id`, and `s` so a node can be mapped back to a
CARLA waypoint.

`map_edges.csv` is a directed adjacency list:

- `LANE_FOLLOW`: normal forward motion.
- `JUNCTION_BRANCH`: one of the legal outgoing junction branches.
- `LANE_CHANGE_LEFT` / `LANE_CHANGE_RIGHT`: legal adjacent-lane transition.
- `distance_m`: geometric edge length.
- `cost_m`: default A* cost; lane-change edges include the configured penalty.
- `yaw_delta_deg`: useful for generating a high-level route command.

`map.xodr` is the authoritative fallback. The CSV graph is a convenient dense
representation, while OpenDRIVE permits rebuilding the graph at a different
resolution without collecting images again.

## Dynamic start and optional goal

Every row in `states.csv` contains the current `waypoint_id` and OpenDRIVE
coordinates. This is the start-node hint. Snap it to the closest node in
`map_nodes.csv` if an exact ID is not present because graph resolution differs
from the vehicle projection.

When `--goal-spawn-index` or `--goal-x/--goal-y` is supplied, the goal columns
and `goal.json` contain the destination waypoint. If no goal was selected,
these fields are blank by design; graph and state data remain usable for a goal
chosen later by a WPF UI.

## Planner output contract

The future A* module should return an ordered list of `node_id` values and fill:

| Field | Meaning |
| --- | --- |
| `route_id` | Unique route/goal selection ID |
| `route_target_index` | Active index in the A* node list |
| `route_target_waypoint_id` | Active local target node |
| `route_target_local_x/y` | Target in ego frame: forward/right metres |
| `route_command` | `LANEFOLLOW`, `LEFT`, `RIGHT`, `STRAIGHT`, `CHANGELANELEFT`, or `CHANGELANERIGHT` |
| `route_progress_m` | Travel distance accumulated along the planned path |
| `route_remaining_m` | Remaining planned path length, not Euclidean distance |
| `route_total_m` | Total planned path length |
| `route_completed` | 1 when the destination tolerance is reached |

These route fields are reserved in the current CSV schema. They stay blank
until the planner and route tracker are running; inventing them during passive
autopilot collection would create incorrect supervision at junctions.

## A* definition

Recommended cost and admissible heuristic:

```text
g(next) = g(current) + edge.cost_m
h(node) = EuclideanDistance(node, goal)
f(node) = g(node) + h(node)
```

If traffic-aware planning is added later, keep `distance_m` unchanged and
derive a separate dynamic cost from speed limit, congestion, closures, or
collision risk.

## Policy observation after A*

The local imitation/DRL policy should consume route-relative information, not
absolute map coordinates:

```text
semantic image
forward speed
lane_offset_m
heading_error_deg
route_target_local_x
route_target_local_y
route_command one-hot
```

This keeps the policy transferable between Town01, Town02, and Town03.
