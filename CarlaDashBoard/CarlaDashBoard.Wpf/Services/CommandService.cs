using System.Text.Json;
using System.Windows.Threading;
using CarlaDashBoard.Wpf.Infrastructure;
using CarlaDashBoard.Wpf.Models;

namespace CarlaDashBoard.Wpf.Services;

/// <summary>Everything the "loại sự kiện" column of design doc §02 lists for /control except
/// ServerInfo (which gets its own strongly-typed event since Settings needs every field).</summary>
public sealed record ControlEvent(string Type, JsonElement Payload);

/// <summary>
/// Owns the /control WebSocket: sends commands (SetMode, SetWeather, SetTown, StartSession,
/// StopSession, SetCameraParams, RecordStart, RecordStop — design doc §02 table) and surfaces
/// server -> client events (ServerInfo, ModeChanged, Error, ...). Pumps/event wiring are set up
/// once in the constructor so a reconnect (ConnectionService) never double-subscribes.
/// </summary>
public sealed class CommandService : IAsyncDisposable
{
    private static readonly JsonSerializerOptions JsonOpts = new() { PropertyNameCaseInsensitive = true };

    private readonly WebSocketClient _client = new();
    private readonly LatestValuePump<ControlEvent> _eventPump;
    private readonly LatestValuePump<ServerInfo> _serverInfoPump;

    public event Action<ServerInfo>? ServerInfoReceived;
    public event Action<ControlEvent>? EventReceived;
    public event Action<Exception?>? Disconnected;

    public CommandService(Dispatcher dispatcher)
    {
        _eventPump = new LatestValuePump<ControlEvent>(dispatcher, e => EventReceived?.Invoke(e));
        _serverInfoPump = new LatestValuePump<ServerInfo>(dispatcher, i => ServerInfoReceived?.Invoke(i));
        _client.TextReceived += OnText;
        _client.Closed += ex => Disconnected?.Invoke(ex);
    }

    public Task ConnectAsync(string host, int port, CancellationToken ct) =>
        _client.ConnectAsync(new Uri($"ws://{host}:{port}/control"), ct);

    // Runs on WebSocketClient's background receive-loop thread — must not touch UI state
    // directly, only ever hand values off through a LatestValuePump.
    private void OnText(string json)
    {
        JsonElement root;
        try
        {
            root = JsonDocument.Parse(json).RootElement;
        }
        catch (JsonException)
        {
            return;
        }
        if (!root.TryGetProperty("type", out var typeProp))
            return;
        var type = typeProp.GetString() ?? "";

        if (type == "ServerInfo")
        {
            var info = JsonSerializer.Deserialize<ServerInfo>(json, JsonOpts);
            if (info is not null)
                _serverInfoPump.Post(info);
            return;
        }
        _eventPump.Post(new ControlEvent(type, root));
    }

    public Task SendAsync(object command, CancellationToken ct = default)
    {
        var json = JsonSerializer.Serialize(command);
        return _client.SendTextAsync(json, ct);
    }

    public ValueTask DisposeAsync() => _client.DisposeAsync();
}
