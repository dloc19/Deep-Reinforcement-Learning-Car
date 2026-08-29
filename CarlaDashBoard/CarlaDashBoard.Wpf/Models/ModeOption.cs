namespace CarlaDashBoard.Wpf.Models;

/// <summary>
/// One tile in the Settings "Chế độ vận hành" group (Hình 04). WireName must match a
/// protocol.MODE_* constant on the bridge side exactly.
/// </summary>
public sealed class ModeOption
{
    public required string WireName { get; init; }
    public required string DisplayName { get; init; }
    public required string Description { get; init; }
    public bool Implemented { get; init; } = true;

    public static readonly IReadOnlyList<ModeOption> All = new List<ModeOption>
    {
        new()
        {
            WireName = "DATA_COLLECTION",
            DisplayName = "Data Collection",
            Description = "Traffic Manager tự lái, có thể bật ghi states.csv + ảnh camera.",
            Implemented = true,
        },
        new()
        {
            WireName = "ASTAR_AUTOPILOT",
            DisplayName = "A* Autopilot",
            Description = "Chọn đích ở màn hình Route & Map rồi lái theo tuyến A*. Cần chọn đích trước.",
            Implemented = true, // Phase 3
        },
        new()
        {
            WireName = "IL_AUTOPILOT",
            DisplayName = "IL Autopilot",
            Description = "Lái bằng model Imitation Learning đã huấn luyện (checkpoint cấu hình ở Bridge Server, xem README).",
            Implemented = true, // Phase 4
        },
        new()
        {
            WireName = "DRL_AUTOPILOT",
            DisplayName = "DRL Autopilot",
            Description = "Lái bằng policy PPO/SAC đã huấn luyện (checkpoint cấu hình ở Bridge Server, xem README).",
            Implemented = true, // Phase 4
        },
        new()
        {
            WireName = "ROUTE_DRL_AUTOPILOT",
            DisplayName = "Route + DRL Autopilot",
            Description = "Đi theo tuyến A* tới điểm đích, bám làn bằng policy đã huấn luyện; " +
                          "pure-pursuit chỉ cầm lái qua ngã tư và lúc đổi làn. Cần chọn đích trước.",
            Implemented = true, // Phase 5
        },
    };
}
