using System.Windows.Threading;

namespace CarlaDashBoard.Wpf.Infrastructure;

/// <summary>
/// The WPF-side half of the design doc's §11 backpressure rule ("Channel dung lượng 1,
/// DropOldest"): if frames/telemetry arrive faster than the UI thread can apply them, only the
/// most recent value survives — never an unbounded backlog of stale <see cref="Dispatcher.BeginInvoke"/>
/// calls. At most one dispatch is ever in flight per pump.
/// </summary>
public sealed class LatestValuePump<T> where T : class
{
    private readonly Dispatcher _dispatcher;
    private readonly Action<T> _apply;
    private T? _pending;
    private int _dispatchQueued;

    public LatestValuePump(Dispatcher dispatcher, Action<T> apply)
    {
        _dispatcher = dispatcher;
        _apply = apply;
    }

    public void Post(T value)
    {
        Volatile.Write(ref _pending, value);
        if (Interlocked.CompareExchange(ref _dispatchQueued, 1, 0) != 0)
            return; // a dispatch is already queued — it will pick up the latest _pending

        _dispatcher.BeginInvoke(() =>
        {
            var latest = Volatile.Read(ref _pending);
            Volatile.Write(ref _dispatchQueued, 0);
            if (latest is not null)
                _apply(latest);
        }, DispatcherPriority.Render);
    }
}
