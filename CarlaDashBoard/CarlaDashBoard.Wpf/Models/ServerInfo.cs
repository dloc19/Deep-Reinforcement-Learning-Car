using System.Text.Json.Serialization;

namespace CarlaDashBoard.Wpf.Models;

public sealed class WeatherPreset
{
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("group")] public string Group { get; set; } = "";
    public override string ToString() => Name;
}

public sealed class SpawnPointInfo
{
    [JsonPropertyName("index")] public int Index { get; set; }
    [JsonPropertyName("x")] public double X { get; set; }
    [JsonPropertyName("y")] public double Y { get; set; }
    [JsonPropertyName("yaw")] public double Yaw { get; set; }
    public override string ToString() => $"#{Index}  ({X:0}, {Y:0})";
}

/// <summary>Mirrors the "ServerInfo" event sent once when a /control client connects.</summary>
public sealed class ServerInfo
{
    [JsonPropertyName("towns")] public List<string> Towns { get; set; } = new();
    [JsonPropertyName("current_town")] public string CurrentTown { get; set; } = "";
    [JsonPropertyName("weather_presets")] public List<WeatherPreset> WeatherPresets { get; set; } = new();
    [JsonPropertyName("vehicle_filter")] public string VehicleFilter { get; set; } = "";
    [JsonPropertyName("spawn_point_count")] public int SpawnPointCount { get; set; }
    [JsonPropertyName("spawn_points")] public List<SpawnPointInfo> SpawnPoints { get; set; } = new();
    [JsonPropertyName("mode")] public string Mode { get; set; } = "IDLE";
    [JsonPropertyName("has_session")] public bool HasSession { get; set; }
}
