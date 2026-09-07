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
        // Dep sach ket noi CU truoc khi mo ket noi moi. Ban truoc gan de `_socket`/
        // `_receiveTask` bi ghi de: vong receive cu VAN CHAY tren socket cu, van ban tiep
        // TextReceived/BinaryReceived, va khi no chet lai ban them mot `Closed` nua —
        // ma `Closed` chinh la thu kich hoat reconnect. Moi lan reconnect vi vay de ra
        // them mot vong receive mo coi, va so ket noi tang gap doi sau moi lan rot.
        await TeardownAsync().ConfigureAwait(false);

        var socket = new ClientWebSocket();
        _socket = socket;
        await socket.ConnectAsync(uri, ct).ConfigureAwait(false);
        _receiveCts = new CancellationTokenSource();
        var token = _receiveCts.Token;
        _receiveTask = Task.Run(() => ReceiveLoopAsync(socket, token), CancellationToken.None);
    }

    /// <summary>Huy vong receive dang chay + dong/giai phong socket hien tai. An toan khi
    /// goi luc chua co ket noi nao.</summary>
    private async Task TeardownAsync()
    {
        var cts = _receiveCts;
        var task = _receiveTask;
        var socket = _socket;
        _receiveCts = null;
        _receiveTask = null;
        _socket = null;

        cts?.Cancel();
        if (socket is not null)
        {
            try { socket.Abort(); } catch { /* best-effort */ }
            socket.Dispose();
        }
        if (task is not null)
        {
            try { await task.ConfigureAwait(false); } catch { /* da bao qua Closed */ }
        }
        cts?.Dispose();
    }

    public Task SendTextAsync(string text, CancellationToken ct = default)
    {
        if (_socket is not { State: WebSocketState.Open })
            throw new InvalidOperationException("WebSocket chưa kết nối.");
        var bytes = Encoding.UTF8.GetBytes(text);
        return _socket.SendAsync(bytes, WebSocketMessageType.Text, true, ct);
    }

    private async Task ReceiveLoopAsync(ClientWebSocket socket, CancellationToken ct)
    {
        var buffer = new byte[ReceiveBufferSize];
        Exception? failure = null;
        try
        {
            while (!ct.IsCancellationRequested && socket.State == WebSocketState.Open)
            {
                using var message = new MemoryStream();
                WebSocketReceiveResult result;
                do
                {
                    result = await socket.ReceiveAsync(buffer, ct).ConfigureAwait(false);
                    if (result.MessageType == WebSocketMessageType.Close)
                    {
                        await socket.CloseAsync(WebSocketCloseStatus.NormalClosure, null, ct)
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
            // Chi bao "rot ket noi" neu socket nay VAN la socket hien hanh. Neu no da bi
            // TeardownAsync/ConnectAsync thay the thi day la mot vong cu dang tan, va bao
            // `Closed` o day se kich hoat them mot lan reconnect thua.
            if (ReferenceEquals(_socket, socket))
                Closed?.Invoke(failure);
        }
    }

    public async ValueTask DisposeAsync()
    {
        var socket = _socket;
        if (socket is { State: WebSocketState.Open })
        {
            try
            {
                await socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "bye", CancellationToken.None)
                    .ConfigureAwait(false);
            }
            catch
            {
                // best-effort close
            }
        }
        await TeardownAsync().ConfigureAwait(false);
    }
}
