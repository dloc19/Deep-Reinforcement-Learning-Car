using System.Net.Http;
using System.Text.Json;
using CarlaDashBoard.Wpf.Models;

namespace CarlaDashBoard.Wpf.Services;

/// <summary>
/// GET /maps/{town} (design doc §01/§04) — plain HTTP, not WebSocket: the road graph is
/// static per town, so it's fetched once and cached client-side here instead of riding the
/// real-time channels (§01's "kênh real-time tách khỏi kênh tĩnh" decision).
/// </summary>
public sealed class MapDataService
{
    private static readonly JsonSerializerOptions JsonOpts = new() { PropertyNameCaseInsensitive = true };

    private readonly HttpClient _http = new() { Timeout = TimeSpan.FromSeconds(30) };
    private readonly Dictionary<string, MapGraph> _cache = new();

    public async Task<MapGraph> GetGraphAsync(string host, int port, string town, CancellationToken ct = default)
    {
        if (_cache.TryGetValue(town, out var cached))
            return cached;

        var url = $"http://{host}:{port}/maps/{Uri.EscapeDataString(town)}";
        using var response = await _http.GetAsync(url, ct).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
        {
            var body = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
            throw new InvalidOperationException(
                $"Bridge Server trả lỗi {(int)response.StatusCode} cho /maps/{town}: {body}");
        }

        var json = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        var graph = await JsonSerializer.DeserializeAsync<MapGraph>(json, JsonOpts, ct).ConfigureAwait(false)
                    ?? throw new InvalidOperationException("Phản hồi /maps rỗng hoặc không hợp lệ.");
        _cache[town] = graph;
        return graph;
    }
}
