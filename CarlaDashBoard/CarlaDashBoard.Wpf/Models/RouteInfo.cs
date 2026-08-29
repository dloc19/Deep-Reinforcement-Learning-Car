using System.Text.Json.Serialization;

namespace CarlaDashBoard.Wpf.Models;

public readonly record struct WorldPoint(double X, double Y);

/// <summary>Mirrors the "RouteComputed" /control event (design doc §02/§04 Hình 03).</summary>
public sealed class RouteInfo
{
    [JsonPropertyName("polyline")] public List<WorldPointDto> Polyline { get; set; } = new();
    [JsonPropertyName("distance_m")] public double DistanceM { get; set; }
    [JsonPropertyName("eta_s")] public double EtaS { get; set; }
    [JsonPropertyName("node_count")] public int NodeCount { get; set; }
}

public sealed class WorldPointDto
{
    [JsonPropertyName("x")] public double X { get; set; }
    [JsonPropertyName("y")] public double Y { get; set; }
}
