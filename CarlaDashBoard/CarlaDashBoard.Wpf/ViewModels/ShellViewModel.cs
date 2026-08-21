using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CarlaDashBoard.Wpf.Services;

namespace CarlaDashBoard.Wpf.ViewModels;

/// <summary>
/// The nav rail + connection dot that every screen shares (design doc §06). Live Drive is
/// created once and kept alive for the whole app lifetime — switching tabs only changes which
/// view is on screen, it never tears down the subscription to camera frames, so flipping back
/// to Live Drive is instant instead of re-syncing.
/// </summary>
public sealed partial class ShellViewModel : ObservableObject, IAsyncDisposable
{
    public ConnectionService Connection { get; }

    public LiveDriveViewModel LiveDrive { get; }
    public SettingsViewModel Settings { get; }
    public RouteMapViewModel RouteMap { get; }

    [ObservableProperty]
    private object _currentView;

    [ObservableProperty]
    private ConnectionState _connectionState = ConnectionState.Disconnected;

    public ShellViewModel(ConnectionService connection)
    {
        Connection = connection;
        Connection.StateChanged += state => ConnectionState = state;

        LiveDrive = new LiveDriveViewModel(connection);
        Settings = new SettingsViewModel(connection);
        RouteMap = new RouteMapViewModel(connection);

        _currentView = LiveDrive;
    }

    // --- Nav rail active-item highlight — MainWindow.xaml binds each nav Button's "selected"
    // look to one of these instead of comparing CurrentView in XAML (WPF has no such operator).
    public bool IsLiveDriveActive => ReferenceEquals(CurrentView, LiveDrive);
    public bool IsRouteMapActive => ReferenceEquals(CurrentView, RouteMap);
    public bool IsSettingsActive => ReferenceEquals(CurrentView, Settings);

    partial void OnCurrentViewChanged(object value)
    {
        OnPropertyChanged(nameof(IsLiveDriveActive));
        OnPropertyChanged(nameof(IsRouteMapActive));
        OnPropertyChanged(nameof(IsSettingsActive));
    }

    [RelayCommand] private void ShowLiveDrive() => CurrentView = LiveDrive;
    [RelayCommand] private void ShowRouteMap() => CurrentView = RouteMap;
    [RelayCommand] private void ShowSettings() => CurrentView = Settings;

    public async ValueTask DisposeAsync() => await Connection.DisposeAsync();
}
