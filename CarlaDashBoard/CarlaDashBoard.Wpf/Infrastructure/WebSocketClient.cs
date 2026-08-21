using System.IO;
using System.Net.WebSockets;
using System.Text;

namespace CarlaDashBoard.Wpf.Infrastructure;

/// <summary>
/// Thin wrapper around <see cref="ClientWebSocket"/> for one endpoint (one of /stream or
/// /control). Owns the receive loop; reconnect/backoff policy is the caller's job
/// (<see cref="Services.ConnectionService"/>) — this class only knows how to be connected or
/// not, once.
/// </summary>
public sealed class WebSocketClient : IAsyncDisposable
{
    private const int ReceiveBufferSize = 64 * 1024;

    private ClientWebSocket? _socket;
    private CancellationTokenSource? _receiveCts;
    private Task? _receiveTask;

    public event Action<byte[]>? BinaryReceived;
    public event Action<string>? TextReceived;
    public event Action<Exception?>? Closed;

    public bool IsConnected => _socket?.State == WebSocketState.Open;

    public async Task ConnectAsync(Uri uri, CancellationToken ct)
    {
        _socket = new ClientWebSocket();
        await _socket.ConnectAsync(uri, ct).ConfigureAwait(false);
        _receiveCts = new CancellationTokenSource();
        _receiveTask = Task.Run(() => ReceiveLoopAsync(_receiveCts.Token));
    }

    public Task SendTextAsync(string text, CancellationToken ct = default)
    {
        if (_socket is not { State: WebSocketState.Open })
            throw new InvalidOperationException("WebSocket chưa kết nối.");
        var bytes = Encoding.UTF8.GetBytes(text);
        return _socket.SendAsync(bytes, WebSocketMessageType.Text, true, ct);
    }

    private async Task ReceiveLoopAsync(CancellationToken ct)
    {
        var buffer = new byte[ReceiveBufferSize];
        Exception? failure = null;
        try
        {
            while (!ct.IsCancellationRequested && _socket is { State: WebSocketState.Open })
            {
                using var message = new MemoryStream();
                WebSocketReceiveResult result;
                do
                {
                    result = await _socket.ReceiveAsync(buffer, ct).ConfigureAwait(false);
                    if (result.MessageType == WebSocketMessageType.Close)
                    {
                        await _socket.CloseAsync(WebSocketCloseStatus.NormalClosure, null, ct)
                            .ConfigureAwait(false);
                        return;
                    }
                    message.Write(buffer, 0, result.Count);
                } while (!result.EndOfMessage);

                if (result.MessageType == WebSocketMessageType.Binary)
                    BinaryReceived?.Invoke(message.ToArray());
                else
                    TextReceived?.Invoke(Encoding.UTF8.GetString(message.ToArray()));
            }
        }
        catch (OperationCanceledException)
        {
            // expected on Disconnect()
        }
        catch (Exception ex)
        {
            failure = ex;
        }
        finally
        {
            Closed?.Invoke(failure);
        }
    }

    public async ValueTask DisposeAsync()
    {
        _receiveCts?.Cancel();
        if (_socket is { State: WebSocketState.Open })
        {
            try
            {
                await _socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "bye", CancellationToken.None)
                    .ConfigureAwait(false);
            }
            catch
            {
                // best-effort close
            }
        }
        _socket?.Dispose();
        if (_receiveTask is not null)
        {
            try { await _receiveTask.ConfigureAwait(false); } catch { /* already surfaced via Closed */ }
        }
    }
}
