using System.Windows.Controls;
using CarlaDashBoard.Wpf.Controls;
using CarlaDashBoard.Wpf.Models;
using CarlaDashBoard.Wpf.ViewModels;

namespace CarlaDashBoard.Wpf.Views;

public partial class RouteMapView : UserControl
{
    public RouteMapView()
    {
        InitializeComponent();
    }

    // MapCanvas raises a plain CLR event (no Behaviors.Wpf dependency for one click handler) —
    // just forward the world point straight into the view model.
    private void Canvas_MapClicked(object? sender, WorldPoint point)
    {
        if (DataContext is RouteMapViewModel vm)
            vm.OnMapClicked(point.X, point.Y);
    }
}
