using System.Windows.Threading;

namespace CarlaDashBoard.Wpf.Services;

public enum ConnectionState { Disconnected, Connecting, Connected, Reconnecting }

/// <summary>
/// Orchestrates <see cref="StreamingService"/> + <see cref="CommandService"/> as one logical
/// connection (design doc §06: "trạng thái kết nối luôn hiện diện" across every screen) and
/// owns the reconnect-with-backoff policy from §11 (1s, 2s, 4s… capped at 15s).
/// </summary>
public sealed class ConnectionService : IAsyncDisposable
{
    private const double BackoffCapSeconds = 15.0;

    private readonly Dispatcher _dispatcher;
    private CancellationTokenSource? _lifecycleCts;
    private string _host = "127.0.0.1";
    private int _port = 8765;
    private bool _wantsConnection;

    public StreamingService Streaming { get; }
    public CommandService Control { get; }

    public string Host => _host;
    public int Port => _port;

    public ConnectionState State { get; private set; } = ConnectionState.Disconnected;
    public event Action<ConnectionState>? StateChanged;

    public ConnectionService(Dispatcher dispatcher)
    {
        _dispatcher = dispatcher;
        Streaming = new StreamingService(dispatcher);
        Control = new CommandService(dispatcher);
        Streaming.Disconnected += _ => OnChannelDropped();
        Control.Disconnected += _ => OnChannelDropped();
    }

    public void Connect(string host, int port)
    {
        _host = host;
        _port = port;
        _wantsConnection = true;
        _lifecycleCts?.Cancel();
        _lifecycleCts = new CancellationTokenSource();
        _ = RunConnectionLoopAsync(_lifecycleCts.Token);
    }

    public void Disconnect()
    {
        _wantsConnection = false;
        _lifecycleCts?.Cancel();
        SetState(ConnectionState.Disconnected);
    }

    private async Task RunConnectionLoopAsync(CancellationToken ct)
    {
        var attempt = 0;
        while (_wantsConnection && !ct.IsCancellationRequested)
        {
            SetState(attempt == 0 ? ConnectionState.Connecting : ConnectionState.Reconnecting);
            try
            {
                await Task.WhenAll(
                    Streaming.ConnectAsync(_host, _port, ct),
                    Control.ConnectAsync(_host, _port, ct)
                ).ConfigureAwait(false);
                SetState(ConnectionState.Connected);
                return; // Streaming/Control.Disconnected will kick off the next loop on drop
            }
            catch (OperationCanceledException)
            {
                return;
            }
            catch (Exception)
            {
                attempt++;
                var delay = Math.Min(Math.Pow(2, attempt - 1), BackoffCapSeconds);
                try
                {
                    await Task.Delay(TimeSpan.FromSeconds(delay), ct).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    return;
                }
            }
        }
    }

    private void OnChannelDropped()
    {
        if (!_wantsConnection || State == ConnectionState.Disconnected)
            return;
        SetState(ConnectionState.Reconnecting);
        _lifecycleCts?.Cancel();
        _lifecycleCts = new CancellationTokenSource();
        _ = RunConnectionLoopAsync(_lifecycleCts.Token);
    }

    private void SetState(ConnectionState state)
    {
        State = state;
        _dispatcher.Invoke(() => StateChanged?.Invoke(state));
    }

    public async ValueTask DisposeAsync()
    {
        _wantsConnection = false;
        _lifecycleCts?.Cancel();
        await Streaming.DisposeAsync().ConfigureAwait(false);
        await Control.DisposeAsync().ConfigureAwait(false);
    }
}
