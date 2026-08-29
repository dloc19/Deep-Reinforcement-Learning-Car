using System.Text.Json.Serialization;

namespace CarlaDashBoard.Wpf.Models;

/// <summary>
/// Mirrors the "telemetry" text frame published on /stream — design doc §02. The bridge
/// speaks snake_case JSON (Python convention); explicit <see cref="JsonPropertyNameAttribute"/>
/// on every field is the honest mapping instead of a guessed naming policy.
/// </summary>
public sealed class Telemetry
{
    [JsonPropertyName("t")] public double T { get; set; }
    [JsonPropertyName("mode")] public string Mode { get; set; } = "IDLE";
    [JsonPropertyName("connected")] public bool Connected { get; set; }
    [JsonPropertyName("ego_alive")] public bool EgoAlive { get; set; }
    [JsonPropertyName("speed_kmh")] public double SpeedKmh { get; set; }
    [JsonPropertyName("steer")] public double Steer { get; set; }
    [JsonPropertyName("throttle")] public double Throttle { get; set; }
    [JsonPropertyName("brake")] public double Brake { get; set; }
    [JsonPropertyName("x")] public double X { get; set; }
    [JsonPropertyName("y")] public double Y { get; set; }
    [JsonPropertyName("yaw")] public double Yaw { get; set; }
    [JsonPropertyName("town")] public string Town { get; set; } = "";

    // Data Collection mode extras (status_extra() in bridge/modes/data_collection.py)
    [JsonPropertyName("recording")] public bool? Recording { get; set; }
    [JsonPropertyName("collision_count")] public int? CollisionCount { get; set; }
    [JsonPropertyName("lane_invasion_count")] public int? LaneInvasionCount { get; set; }
    [JsonPropertyName("frames_recorded")] public int? FramesRecorded { get; set; }
    [JsonPropertyName("record_elapsed_s")] public double? RecordElapsedS { get; set; }

    public static readonly Telemetry Empty = new() { Mode = "IDLE", Connected = false };
}
