using System.Windows;
using CarlaDashBoard.Wpf.Services;
using CarlaDashBoard.Wpf.ViewModels;

namespace CarlaDashBoard.Wpf;

/// <summary>
/// Composition root. No DI container — three view models and one connection service is small
/// enough to wire up by hand (design doc §08 keeps the service layer intentionally thin).
/// </summary>
public partial class App : Application
{
    private ShellViewModel? _shell;

    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        var connection = new ConnectionService(Dispatcher);
        _shell = new ShellViewModel(connection);

        var window = new MainWindow { DataContext = _shell };
        window.Show();
    }

    protected override void OnExit(ExitEventArgs e)
    {
        _shell?.DisposeAsync().AsTask().GetAwaiter().GetResult();
        base.OnExit(e);
    }
}
