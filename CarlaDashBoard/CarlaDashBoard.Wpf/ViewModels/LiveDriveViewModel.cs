using System.Windows.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CarlaDashBoard.Wpf.Models;
using CarlaDashBoard.Wpf.Services;

namespace CarlaDashBoard.Wpf.ViewModels;

public enum CameraDisplayMode { Rgb, Segmentation, Split }

/// <summary>Màn hình 1 — Live Drive (design doc §03, Hình 02).</summary>
public sealed partial class LiveDriveViewModel : ObservableObject
{
    private readonly ConnectionService _connection;

    [ObservableProperty] private CameraDisplayMode _displayMode = CameraDisplayMode.Rgb;
    [ObservableProperty] private BitmapImage? _rgbFrame;
    [ObservableProperty] private BitmapImage? _segFrame;

    [ObservableProperty] private Telemetry _telemetry = Telemetry.Empty;
    [ObservableProperty] private string _statusMessage = "";

    public bool IsDataCollectionMode => Telemetry.Mode == "DATA_COLLECTION";

    public LiveDriveViewModel(ConnectionService connection)
    {
        _connection = connection;
        _connection.Streaming.RgbFrameReady += f => RgbFrame = f;
        _connection.Streaming.SegFrameReady += f => SegFrame = f;
        _connection.Streaming.TelemetryReady += t => Telemetry = t;
        _connection.Control.EventReceived += e => StatusMessage = DescribeEvent(e);
    }

    partial void OnTelemetryChanged(Telemetry value) => OnPropertyChanged(nameof(IsDataCollectionMode));

    [RelayCommand] private void ShowRgb() => DisplayMode = CameraDisplayMode.Rgb;
    [RelayCommand] private void ShowSegmentation() => DisplayMode = CameraDisplayMode.Segmentation;
    [RelayCommand] private void ShowSplit() => DisplayMode = CameraDisplayMode.Split;

    [RelayCommand]
    private async Task ToggleRecordAsync()
    {
        var command = Telemetry.Recording == true
            ? new { type = "RecordStop" }
            : (object)new { type = "RecordStart" };
        try
        {
            await _connection.Control.SendAsync(command);
        }
        catch (Exception ex)
        {
            StatusMessage = $"Không gửi được lệnh ghi dữ liệu: {ex.Message}";
        }
    }

    private static string DescribeEvent(ControlEvent e) => e.Type switch
    {
        "Error" => $"Lỗi: {e.Payload.GetProperty("message").GetString()}",
        "ModeChanged" => $"Đã chuyển mode: {e.Payload.GetProperty("mode").GetString()}",
        "ModeChanging" => "Đang chuyển mode…",
        _ => "",
    };
}
