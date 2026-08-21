using System.Text.Json;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using CarlaDashBoard.Wpf.Infrastructure;
using CarlaDashBoard.Wpf.Models;

namespace CarlaDashBoard.Wpf.Services;

/// <summary>
/// Owns the /stream WebSocket (design doc §02/§08 — camera JPEG + telemetry, server -> client
/// only). Decodes frames off the UI thread and hands the *latest* one to the dispatcher via
/// <see cref="LatestValuePump{T}"/> — never a growing backlog. Pumps and event wiring are set
/// up once, in the constructor, so calling <see cref="ConnectAsync"/> again after a drop
/// (reconnect — see ConnectionService) never double-subscribes.
/// </summary>
public sealed class StreamingService : IAsyncDisposable
{
    private static readonly JsonSerializerOptions JsonOpts = new() { PropertyNameCaseInsensitive = true };

    private readonly WebSocketClient _client = new();
    private readonly LatestValuePump<BitmapImage> _rgbPump;
    private readonly LatestValuePump<BitmapImage> _segPump;
    private readonly LatestValuePump<Telemetry> _telemetryPump;

    public event Action<BitmapImage>? RgbFrameReady;
    public event Action<BitmapImage>? SegFrameReady;
    public event Action<Telemetry>? TelemetryReady;
    public event Action<Exception?>? Disconnected;

    public StreamingService(Dispatcher dispatcher)
    {
        _rgbPump = new LatestValuePump<BitmapImage>(dispatcher, img => RgbFrameReady?.Invoke(img));
        _segPump = new LatestValuePump<BitmapImage>(dispatcher, img => SegFrameReady?.Invoke(img));
        _telemetryPump = new LatestValuePump<Telemetry>(dispatcher, t => TelemetryReady?.Invoke(t));

        _client.BinaryReceived += OnBinary;
        _client.TextReceived += OnText;
        _client.Closed += ex => Disconnected?.Invoke(ex);
    }

    public Task ConnectAsync(string host, int port, CancellationToken ct) =>
        _client.ConnectAsync(new Uri($"ws://{host}:{port}/stream"), ct);

    private void OnBinary(byte[] frame)
    {
        // Wire layout: byte[0] = channel tag (0x01 rgb / 0x02 seg), byte[1..] = JPEG — protocol.py.
        if (frame.Length < 2)
            return;
        byte channel = frame[0];
        var jpeg = new byte[frame.Length - 1];
        Buffer.BlockCopy(frame, 1, jpeg, 0, jpeg.Length);

        var bitmap = JpegFrameDecoder.Decode(jpeg);
        if (bitmap is null)
            return;

        switch (channel)
        {
            case 0x01: _rgbPump.Post(bitmap); break;
            case 0x02: _segPump.Post(bitmap); break;
        }
    }

    private void OnText(string json)
    {
        try
        {
            var telemetry = JsonSerializer.Deserialize<Telemetry>(json, JsonOpts);
            if (telemetry is not null)
                _telemetryPump.Post(telemetry);
        }
        catch (JsonException)
        {
            // a malformed telemetry frame is not worth crashing the UI over — skip it
        }
    }

    public ValueTask DisposeAsync() => _client.DisposeAsync();
}
