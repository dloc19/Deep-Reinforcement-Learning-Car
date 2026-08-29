using System.Collections.ObjectModel;
using System.Globalization;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CarlaDashBoard.Wpf.Models;
using CarlaDashBoard.Wpf.Services;

namespace CarlaDashBoard.Wpf.ViewModels;

/// <summary>Màn hình 2 — Route & Map (design doc §04, Hình 03).</summary>
public sealed partial class RouteMapViewModel : ObservableObject
{
    private readonly ConnectionService _connection;
    private readonly MapDataService _mapData = new();
    private string _graphTown = "";

    [ObservableProperty] private MapGraph? _graph;
    [ObservableProperty] private IReadOnlyList<WorldPoint>? _routePolyline;
    [ObservableProperty] private WorldPoint? _carPosition;
    [ObservableProperty] private WorldPoint? _destinationPosition;

    [ObservableProperty] private bool _hasRoute;
    [ObservableProperty] private double _routeDistanceM;
    [ObservableProperty] private double _routeEtaS;

    [ObservableProperty] private string _manualX = "";
    [ObservableProperty] private string _manualY = "";
    [ObservableProperty] private SpawnPointInfo? _selectedSpawnPoint;

    [ObservableProperty] private bool _isLoadingGraph;
    [ObservableProperty] private string _statusMessage = "Kết nối server rồi bắt đầu phiên để xem bản đồ.";

    public ObservableCollection<SpawnPointInfo> SpawnPoints { get; } = new();

    public RouteMapViewModel(ConnectionService connection)
    {
        _connection = connection;
        _connection.Control.ServerInfoReceived += OnServerInfo;
        _connection.Control.EventReceived += OnControlEvent;
        _connection.Streaming.TelemetryReady += t =>
            CarPosition = t.EgoAlive ? new WorldPoint(t.X, t.Y) : null;
    }

    private void OnServerInfo(ServerInfo info)
    {
        SpawnPoints.Clear();
        foreach (var sp in info.SpawnPoints) SpawnPoints.Add(sp);

        if (info.CurrentTown != _graphTown)
            _ = LoadGraphAsync(info.CurrentTown);
    }

    private void OnControlEvent(ControlEvent e)
    {
        switch (e.Type)
        {
            case "RouteComputed":
                var polyline = e.Payload.GetProperty("polyline").EnumerateArray()
                    .Select(p => new WorldPoint(p.GetProperty("x").GetDouble(), p.GetProperty("y").GetDouble()))
                    .ToList();
                RoutePolyline = polyline;
                RouteDistanceM = e.Payload.GetProperty("distance_m").GetDouble();
                RouteEtaS = e.Payload.GetProperty("eta_s").GetDouble();
                HasRoute = true;
                StatusMessage = $"Đã tính tuyến: {RouteDistanceM:0} m, ~{RouteEtaS:0} s.";
                break;
            case "RouteCompleted":
                StatusMessage = "Xe đã tới đích.";
                break;
            case "TownChanged":
                var town = e.Payload.GetProperty("town").GetString() ?? "";
                RoutePolyline = null;
                HasRoute = false;
                DestinationPosition = null;
                _ = LoadGraphAsync(town);
                break;
            case "Error":
                StatusMessage = $"Lỗi: {e.Payload.GetProperty("message").GetString()}";
                break;
        }
    }

    private async Task LoadGraphAsync(string town)
    {
        if (string.IsNullOrEmpty(town))
            return;
        IsLoadingGraph = true;
        StatusMessage = $"Đang tải bản đồ {town}…";
        try
        {
            Graph = await _mapData.GetGraphAsync(_connection.Host, _connection.Port, town);
            _graphTown = town;
            StatusMessage = $"Đã tải bản đồ {town} — {Graph.NodeCount} node, {Graph.EdgeCount} cạnh.";
        }
        catch (Exception ex)
        {
            StatusMessage = $"Không tải được bản đồ {town}: {ex.Message}";
        }
        finally
        {
            IsLoadingGraph = false;
        }
    }

    /// <summary>Called by RouteMapView's code-behind when the user clicks the map canvas.</summary>
    public async void OnMapClicked(double worldX, double worldY)
    {
        DestinationPosition = new WorldPoint(worldX, worldY);
        await SendDestinationAsync(new { type = "SetDestination", x = worldX, y = worldY });
    }

    [RelayCommand]
    private async Task SetDestinationFromSpawnPointAsync()
    {
        if (SelectedSpawnPoint is null)
        {
            StatusMessage = "Chọn một spawn point trước.";
            return;
        }
        DestinationPosition = new WorldPoint(SelectedSpawnPoint.X, SelectedSpawnPoint.Y);
        await SendDestinationAsync(new { type = "SetDestination", spawn_index = SelectedSpawnPoint.Index });
    }

    [RelayCommand]
    private async Task SetDestinationFromCoordinatesAsync()
    {
        if (!double.TryParse(ManualX, NumberStyles.Float, CultureInfo.InvariantCulture, out var x) ||
            !double.TryParse(ManualY, NumberStyles.Float, CultureInfo.InvariantCulture, out var y))
        {
            StatusMessage = "Toạ độ x, y không hợp lệ.";
            return;
        }
        DestinationPosition = new WorldPoint(x, y);
        await SendDestinationAsync(new { type = "SetDestination", x, y });
    }

    [RelayCommand]
    private Task StartDrivingAsync() => StartModeAsync("ASTAR_AUTOPILOT");

    /// <summary>
    /// Đi cùng tuyến A* nhưng để policy đã huấn luyện bám làn; pure-pursuit chỉ cầm lái qua
    /// ngã tư và lúc đổi làn — vì observation của policy không chứa hướng rẽ nên nó không thể
    /// tự quyết định đi nhánh nào (xem bridge/modes/route_learned_autopilot.py).
    /// </summary>
    [RelayCommand]
    private Task StartDrivingWithPolicyAsync() => StartModeAsync("ROUTE_DRL_AUTOPILOT");

    private async Task StartModeAsync(string mode)
    {
        if (!HasRoute)
        {
            StatusMessage = "Chưa có tuyến đường — chọn điểm đến trước.";
            return;
        }
        try
        {
            await _connection.Control.SendAsync(new { type = "SetMode", mode });
        }
        catch (Exception ex)
        {
            StatusMessage = $"Không gửi được lệnh: {ex.Message}";
        }
    }

    private async Task SendDestinationAsync(object command)
    {
        HasRoute = false;
        RoutePolyline = null;
        StatusMessage = "Đang tính tuyến đường…";
        try
        {
            await _connection.Control.SendAsync(command);
        }
        catch (Exception ex)
        {
            StatusMessage = $"Không gửi được lệnh: {ex.Message}";
        }
    }
}
