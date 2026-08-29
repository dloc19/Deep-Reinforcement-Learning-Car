using System.Text.Json.Serialization;

namespace CarlaDashBoard.Wpf.Models;

/// <summary>One node of the A* graph — same fields as map_graph.py's build_payload().</summary>
public sealed class MapNode
{
    [JsonPropertyName("node_id")] public string NodeId { get; set; } = "";
    [JsonPropertyName("road_id")] public int RoadId { get; set; }
    [JsonPropertyName("section_id")] public int SectionId { get; set; }
    [JsonPropertyName("lane_id")] public int LaneId { get; set; }
    [JsonPropertyName("s")] public double S { get; set; }
    [JsonPropertyName("x")] public double X { get; set; }
    [JsonPropertyName("y")] public double Y { get; set; }
    [JsonPropertyName("z")] public double Z { get; set; }
    [JsonPropertyName("yaw_deg")] public double YawDeg { get; set; }
}

public sealed class MapEdge
{
    [JsonPropertyName("from_node_id")] public string FromNodeId { get; set; } = "";
    [JsonPropertyName("to_node_id")] public string ToNodeId { get; set; } = "";
    [JsonPropertyName("edge_type")] public string EdgeType { get; set; } = "";
    [JsonPropertyName("cost_m")] public double CostM { get; set; }
    [JsonPropertyName("yaw_delta_deg")] public double YawDeltaDeg { get; set; }
}

/// <summary>Mirrors the JSON body of GET /maps/{town} (design doc §01/§04, bridge's map_graph.py).</summary>
public sealed class MapGraph
{
    [JsonPropertyName("town")] public string Town { get; set; } = "";
    [JsonPropertyName("resolution_m")] public double ResolutionM { get; set; }
    [JsonPropertyName("node_count")] public int NodeCount { get; set; }
    [JsonPropertyName("edge_count")] public int EdgeCount { get; set; }
    [JsonPropertyName("nodes")] public List<MapNode> Nodes { get; set; } = new();
    [JsonPropertyName("edges")] public List<MapEdge> Edges { get; set; } = new();
}
