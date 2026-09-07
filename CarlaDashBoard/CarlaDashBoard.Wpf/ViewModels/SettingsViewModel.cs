using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CarlaDashBoard.Wpf.Models;
using CarlaDashBoard.Wpf.Services;

namespace CarlaDashBoard.Wpf.ViewModels;

/// <summary>Màn hình 3 — Settings (design doc §05, Hình 04).</summary>
public sealed partial class SettingsViewModel : ObservableObject
{
    private readonly ConnectionService _connection;

    // --- Kết nối Bridge Server ---
    [ObservableProperty] private string _host = "127.0.0.1";
    [ObservableProperty] private int _port = 8765;

    [NotifyCanExecuteChangedFor(nameof(StartSessionCommand))]
    [NotifyCanExecuteChangedFor(nameof(StopSessionCommand))]
    [ObservableProperty] private ConnectionState _connectionState = ConnectionState.Disconnected;

    // --- Bản đồ & thời tiết ---
    public ObservableCollection<string> Towns { get; } = new();
    public ObservableCollection<WeatherPreset> WeatherPresets { get; } = new();
    [ObservableProperty] private string? _selectedTown;
    [ObservableProperty] private WeatherPreset? _selectedWeatherPreset;

    [ObservableProperty] private string _vehicleFilter = "";
    [ObservableProperty] private int _spawnPointCount;

    [NotifyCanExecuteChangedFor(nameof(StartSessionCommand))]
    [NotifyCanExecuteChangedFor(nameof(StopSessionCommand))]
    [ObservableProperty] private bool _hasSession;

    // --- Chế độ vận hành ---
    public IReadOnlyList<ModeOption> Modes { get; } = ModeOption.All;
    [ObservableProperty] private ModeOption? _selectedMode;
    [ObservableProperty] private string _currentModeWire = "IDLE";

    // --- Camera ---
    [ObservableProperty] private int _cameraWidth = 800;
    [ObservableProperty] private int _cameraHeight = 450;
    [ObservableProperty] private double _cameraFov = 90.0;
    [ObservableProperty] private double _cameraFps = 15.0;

    [ObservableProperty] private string _statusMessage = "";

    public SettingsViewModel(ConnectionService connection)
    {
        _connection = connection;
        SelectedMode = Modes[0];

        _connection.StateChanged += state => ConnectionState = state;
        _connection.Control.ServerInfoReceived += OnServerInfo;
        _connection.Control.EventReceived += OnControlEvent;
    }

    private void OnServerInfo(ServerInfo info)
    {
        Towns.Clear();
        foreach (var town in info.Towns) Towns.Add(town);
        SelectedTown = info.Towns.Contains(info.CurrentTown) ? info.CurrentTown : info.Towns.FirstOrDefault();

        WeatherPresets.Clear();
        foreach (var preset in info.WeatherPresets) WeatherPresets.Add(preset);
        SelectedWeatherPreset ??= WeatherPresets.FirstOrDefault();

        VehicleFilter = info.VehicleFilter;
        SpawnPointCount = info.SpawnPointCount;
        HasSession = info.HasSession;
        CurrentModeWire = info.Mode;
        StatusMessage = "Đã nhận thông tin server.";
    }

    private void OnControlEvent(ControlEvent e)
    {
        StatusMessage = e.Type switch
        {
            "Error" => $"Lỗi: {e.Payload.GetProperty("message").GetString()}",
            "SessionStarted" => "Đã bắt đầu phiên (xe đã spawn).",
            "SessionStopped" => "Đã dừng phiên.",
            "ModeChanging" => "Đang chuyển mode…",
            "ModeChanged" => $"Đã chuyển sang mode {e.Payload.GetProperty("mode").GetString()}.",
            "TownChanged" => $"Đã đổi bản đồ sang {e.Payload.GetProperty("town").GetString()}.",
            "WeatherChanged" => $"Đã áp dụng thời tiết {e.Payload.GetProperty("preset").GetString()}.",
            "CameraParamsChanged" => "Đã áp dụng cấu hình camera mới.",
            "RecordingStarted" => "Đã bắt đầu ghi dữ liệu.",
            "RecordingStopped" => "Đã dừng ghi dữ liệu.",
            "CarlaReconnected" => $"Đã nối lại CARLA ({e.Payload.GetProperty("town").GetString()}) — " +
                                  "phiên cũ đã mất, bấm \"Bắt đầu phiên\" để spawn xe mới.",
            _ => StatusMessage,
        };
        if (e.Type is "SessionStarted") HasSession = true;
        if (e.Type is "SessionStopped") { HasSession = false; CurrentModeWire = "IDLE"; }
        // CarlaUE4 chet/khoi dong lai: xe cu bien mat cung the gioi cu. Bridge Server da bo
        // ego + mode cua no, nhung phia nay chi cap nhat HasSession qua ServerInfo (gui mot
        // lan luc nối) va SessionStarted/Stopped — khong co dong nay thi HasSession ket o
        // true, nen nut "Bat dau phien" bi khoa vinh vien va nguoi dung khong con duong nao
        // spawn lai xe ngoai viec khoi dong lai ca app.
        if (e.Type is "CarlaReconnected" || IsCarlaLost(e))
        {
            HasSession = false;
            CurrentModeWire = "IDLE";
        }
        if (e.Type is "ModeChanged") CurrentModeWire = e.Payload.GetProperty("mode").GetString() ?? CurrentModeWire;
    }

    /// <summary>Error kèm code "CARLA_LOST" (sim_loop._handle_tick_failure).</summary>
    internal static bool IsCarlaLost(ControlEvent e) =>
        e.Type == "Error" &&
        e.Payload.TryGetProperty("code", out var code) &&
        code.GetString() == "CARLA_LOST";

    [RelayCommand]
    private void Connect() => _connection.Connect(Host, Port);

    [RelayCommand]
    private void Disconnect() => _connection.Disconnect();

    [RelayCommand(CanExecute = nameof(CanStartSession))]
    private async Task StartSessionAsync() => await SendAsync(new { type = "StartSession" });
    private bool CanStartSession() => ConnectionState == ConnectionState.Connected && !HasSession;

    [RelayCommand(CanExecute = nameof(CanStopSession))]
    private async Task StopSessionAsync() => await SendAsync(new { type = "StopSession" });
    private bool CanStopSession() => ConnectionState == ConnectionState.Connected && HasSession;

    [RelayCommand]
    private async Task ApplyTownAsync()
    {
        if (SelectedTown is null) return;
        await SendAsync(new { type = "SetTown", town = SelectedTown });
    }

    [RelayCommand]
    private async Task ApplyWeatherAsync()
    {
        if (SelectedWeatherPreset is null) return;
        await SendAsync(new { type = "SetWeather", preset = SelectedWeatherPreset.Name });
    }

    [RelayCommand]
    private async Task ApplyModeAsync()
    {
        if (SelectedMode is null) return;
        if (!SelectedMode.Implemented)
        {
            StatusMessage = $"{SelectedMode.DisplayName} chưa được triển khai ở Bridge Server (xem lộ trình §12).";
            return;
        }
        if (!HasSession)
        {
            StatusMessage = "Chưa có xe — bấm \"Bắt đầu phiên\" trước.";
            return;
        }
        await SendAsync(new { type = "SetMode", mode = SelectedMode.WireName });
    }

    [RelayCommand]
    private async Task ApplyCameraAsync() => await SendAsync(new
    {
        type = "SetCameraParams",
        width = CameraWidth,
        height = CameraHeight,
        fov = CameraFov,
        fps = CameraFps,
    });

    private async Task SendAsync(object command)
    {
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
