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
    private readonly object _gate = new();
    private CancellationTokenSource? _lifecycleCts;
    private string _host = "127.0.0.1";
    private int _port = 8765;
    private bool _wantsConnection;
    // Mot vong ket noi DUY NHAT tai mot thoi diem. /stream va /control rot gan nhu cung
    // luc, nen ban truoc nhan hai su kien Disconnected -> khoi dong hai vong song song,
    // moi vong lai mo ca hai kenh: 4 socket cho 2 kenh, va server ghi day traceback vi
    // mot nua so ket noi bi bo ngay sau khi bat tay xong.
    private Task _connectionLoop = Task.CompletedTask;
    // 0 = chua co lan rot nao dang cho noi lai; 1 = da co. Interlocked vi hai su kien
    // Disconnected den tu hai luong receive khac nhau.
    private int _reconnectArmed;

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
        Volatile.Write(ref _reconnectArmed, 0);
        RestartConnectionLoop(isReconnect: false);
    }

    public void Disconnect()
    {
        _wantsConnection = false;
        lock (_gate)
        {
            _lifecycleCts?.Cancel();
        }
        SetState(ConnectionState.Disconnected);
    }

    /// <summary>Huy vong dang chay (neu co) roi khoi dong DUNG MOT vong moi, noi tiep vong
    /// cu de hai vong khong bao gio chay chong len nhau.</summary>
    private void RestartConnectionLoop(bool isReconnect)
    {
        lock (_gate)
        {
            _lifecycleCts?.Cancel();
            var cts = new CancellationTokenSource();
            _lifecycleCts = cts;
            var previous = _connectionLoop;
            _connectionLoop = RunAfterAsync(previous, isReconnect, cts.Token);
        }
    }

    private async Task RunAfterAsync(Task previous, bool isReconnect, CancellationToken ct)
    {
        try { await previous.ConfigureAwait(false); } catch { /* vong truoc da tu xu ly */ }
        await RunConnectionLoopAsync(isReconnect, ct).ConfigureAwait(false);
    }

    private async Task RunConnectionLoopAsync(bool isReconnect, CancellationToken ct)
    {
        var attempt = 0;
        while (_wantsConnection && !ct.IsCancellationRequested)
        {
            SetState(isReconnect || attempt > 0 ? ConnectionState.Reconnecting : ConnectionState.Connecting);
            try
            {
                await Task.WhenAll(
                    Streaming.ConnectAsync(_host, _port, ct),
                    Control.ConnectAsync(_host, _port, ct)
                ).ConfigureAwait(false);
                // Mo lai chot TRUOC khi bao Connected: tu day tro di, mot su kien
                // Disconnected moi la mot lan rot that, dang duoc noi lai.
                Volatile.Write(ref _reconnectArmed, 0);
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
        // Mot lan rot keo theo CA HAI kenh: /stream va /control cung mat server, nen hai su
        // kien Disconnected ve gan nhu cung luc. Chi su kien dau tien duoc khoi dong lai
        // vong ket noi; chot chi mo lai khi da noi duoc (xem RunConnectionLoopAsync).
        if (Interlocked.Exchange(ref _reconnectArmed, 1) == 1)
            return;
        SetState(ConnectionState.Reconnecting);
        RestartConnectionLoop(isReconnect: true);
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
