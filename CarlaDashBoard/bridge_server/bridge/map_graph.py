"""Serializes a router_plan RouteGraph (planner.graph — node_id -> carla.Waypoint, node_id ->
list[Edge]) into the JSON shape design doc §04/§08 expects for GET /maps/{town}: a `nodes`
list and an `edges` list, same field names as map_nodes.csv/map_edges.csv so this reads the
same whether it came from a live graph or an exported CSV (see router_plan/README.md,
"Quyết định đã chốt", for why the graph itself is built live rather than round-tripped
through CSV).
"""


def build_payload(town, graph):
    nodes = []
    # `graph.nodes` anh xa node_id -> carla.Transform, KHONG phai carla.Waypoint:
    # router_plan/graph_builder.py co y chi giu Transform ("chi giu Transform, khong giu ca
    # Waypoint") de khong om hang nghin doi tuong Waypoint song. Vi vay road_id/section_id/
    # lane_id/s khong con doc duoc o day va bi bo khoi payload — MapCanvas ben WPF chi ve
    # tu x/y nen khong dung toi chung. Ban truoc doc `waypoint.transform.location` va nem
    # AttributeError, lam /maps/{town} tra 503 vinh vien (minimap trong).
    for node_id, transform in graph.nodes.items():
        loc = transform.location
        nodes.append({
            "node_id": node_id,
            "x": round(loc.x, 2),
            "y": round(loc.y, 2),
            "z": round(loc.z, 2),
            "yaw_deg": round(transform.rotation.yaw, 2),
        })

    edges = []
    for from_id, edge_list in graph.edges.items():
        for edge in edge_list:
            edges.append({
                "from_node_id": from_id,
                "to_node_id": edge.to_id,
                "edge_type": edge.edge_type,
                "cost_m": round(edge.cost_m, 2),
                "yaw_delta_deg": round(edge.yaw_delta_deg, 2),
            })

    return {
        "type": "MapGraph",
        "town": town,
        "resolution_m": graph.resolution_m,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }
