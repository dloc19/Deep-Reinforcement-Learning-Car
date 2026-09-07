using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using CarlaDashBoard.Wpf.Models;

namespace CarlaDashBoard.Wpf.Controls;

/// <summary>
/// Draws the A* road graph + route + car/destination markers with <see cref="DrawingContext"/>
/// instead of a XAML shape per edge — design doc §08 picked this deliberately over binding a
/// Shape per node/edge (a full Town can have 5–15k edges) or a third-party map control (the
/// graph already IS the map, in CARLA's own world coordinates — no tiles/projection needed).
/// </summary>
public sealed class MapCanvas : FrameworkElement
{
    public static readonly DependencyProperty GraphProperty = DependencyProperty.Register(
        nameof(Graph), typeof(MapGraph), typeof(MapCanvas),
        new FrameworkPropertyMetadata(null, OnGraphChanged));

    public static readonly DependencyProperty RoutePolylineProperty = DependencyProperty.Register(
        nameof(RoutePolyline), typeof(IReadOnlyList<WorldPoint>), typeof(MapCanvas),
        new FrameworkPropertyMetadata(null, OnRouteChanged));

    public static readonly DependencyProperty CarPositionProperty = DependencyProperty.Register(
        nameof(CarPosition), typeof(WorldPoint?), typeof(MapCanvas),
        new FrameworkPropertyMetadata(null, OnVisualPropertyChanged));

    public static readonly DependencyProperty DestinationPositionProperty = DependencyProperty.Register(
        nameof(DestinationPosition), typeof(WorldPoint?), typeof(MapCanvas),
        new FrameworkPropertyMetadata(null, OnVisualPropertyChanged));

    public event EventHandler<WorldPoint>? MapClicked;

    public MapGraph? Graph
    {
        get => (MapGraph?)GetValue(GraphProperty);
        set => SetValue(GraphProperty, value);
    }

    public IReadOnlyList<WorldPoint>? RoutePolyline
    {
        get => (IReadOnlyList<WorldPoint>?)GetValue(RoutePolylineProperty);
        set => SetValue(RoutePolylineProperty, value);
    }

    public WorldPoint? CarPosition
    {
        get => (WorldPoint?)GetValue(CarPositionProperty);
        set => SetValue(CarPositionProperty, value);
    }

    public WorldPoint? DestinationPosition
    {
        get => (WorldPoint?)GetValue(DestinationPositionProperty);
        set => SetValue(DestinationPositionProperty, value);
    }

    private static readonly Brush EdgeBrush = new SolidColorBrush(Color.FromArgb(90, 0xA6, 0xB1, 0xB9));
    private static readonly Brush RouteBrush = new SolidColorBrush(Color.FromRgb(0x57, 0xC6, 0xD2));
    private static readonly Brush CarBrush = new SolidColorBrush(Color.FromRgb(0xE8, 0xA3, 0x3D));
    private static readonly Brush DestinationBrush = new SolidColorBrush(Color.FromRgb(0xE2, 0x63, 0x5F));
    private static readonly Pen CarOutline = new(Brushes.Black, 1.5);

    private StreamGeometry? _edgeGeometry;
    private Rect _worldBounds = Rect.Empty;

    public MapCanvas()
    {
        Focusable = false;
        SizeChanged += (_, _) => InvalidateVisual();
    }

    private static void OnGraphChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        var self = (MapCanvas)d;
        self.RebuildEdgeGeometry(e.NewValue as MapGraph);
        self.InvalidateVisual();
    }

    private static void OnRouteChanged(DependencyObject d, DependencyPropertyChangedEventArgs e) =>
        ((MapCanvas)d).InvalidateVisual();

    private static void OnVisualPropertyChanged(DependencyObject d, DependencyPropertyChangedEventArgs e) =>
        ((MapCanvas)d).InvalidateVisual();

    private void RebuildEdgeGeometry(MapGraph? graph)
    {
        if (graph is null || graph.Nodes.Count == 0)
        {
            _edgeGeometry = null;
            _worldBounds = Rect.Empty;
            return;
        }

        var byId = new Dictionary<string, MapNode>(graph.Nodes.Count);
        foreach (var node in graph.Nodes)
            byId[node.NodeId] = node;

        double minX = double.MaxValue, minY = double.MaxValue, maxX = double.MinValue, maxY = double.MinValue;
        foreach (var node in graph.Nodes)
        {
            if (node.X < minX) minX = node.X;
            if (node.X > maxX) maxX = node.X;
            if (node.Y < minY) minY = node.Y;
            if (node.Y > maxY) maxY = node.Y;
        }
        _worldBounds = new Rect(minX, minY, Math.Max(maxX - minX, 1), Math.Max(maxY - minY, 1));

        var geometry = new StreamGeometry();
        using (var ctx = geometry.Open())
        {
            foreach (var edge in graph.Edges)
            {
                if (!byId.TryGetValue(edge.FromNodeId, out var from) ||
                    !byId.TryGetValue(edge.ToNodeId, out var to))
                    continue;
                ctx.BeginFigure(new Point(from.X, from.Y), false, false);
                ctx.LineTo(new Point(to.X, to.Y), true, false);
            }
        }
        geometry.Freeze();
        _edgeGeometry = geometry;
    }

    /// <summary>World (CARLA x/y, metres) -> screen pixels: fit `_worldBounds` into the
    /// current render size with a margin, keeping CARLA's own top-down orientation.
    ///
    /// KHONG lat truc Y. CARLA dung he toa do THUAN TAY TRAI (+X truoc, +Y phai, +Z len),
    /// nen nhin tu tren xuong voi +X sang phai thi +Y di XUONG man hinh — dung bang y het
    /// `world_to_pixel()` trong `no_rendering_mode.py` cua chinh CARLA (no map thang
    /// `location.y` vao pixel y, khong doi dau), va do cung la huong cua moi anh ban do
    /// Town0x chinh thuc. Ban truoc nhan `-scale` vao Y "cho doc north-up": ket qua la ca
    /// thi tran bi LAT GUONG. Doi chieu Town03: buc xuyen tam o giua thi van dung cho, nhung
    /// nhanh cut (cul-de-sac) le ra o goc duoi-phai lai nhay len goc tren-phai, va moi khuc
    /// re trong tuyen A* hien ra nguoc ben so voi luc lai that.</summary>
    private Matrix ComputeWorldToScreen()
    {
        if (_worldBounds.IsEmpty || ActualWidth <= 1 || ActualHeight <= 1)
            return Matrix.Identity;

        const double margin = 28;
        var availW = Math.Max(ActualWidth - 2 * margin, 1);
        var availH = Math.Max(ActualHeight - 2 * margin, 1);
        var scale = Math.Min(availW / _worldBounds.Width, availH / _worldBounds.Height);

        var m = Matrix.Identity;
        m.Translate(-(_worldBounds.X + _worldBounds.Width / 2), -(_worldBounds.Y + _worldBounds.Height / 2));
        m.Scale(scale, scale);
        m.Translate(ActualWidth / 2, ActualHeight / 2);
        return m;
    }

    protected override void OnRender(DrawingContext dc)
    {
        // Transparent fill so the whole area is hit-testable (needed for click-to-place).
        dc.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, ActualWidth, ActualHeight));

        var matrix = ComputeWorldToScreen();
        if (matrix.IsIdentity)
            return;
        var scale = Math.Sqrt(Math.Abs(matrix.M11 * matrix.M22 - matrix.M12 * matrix.M21));

        if (_edgeGeometry is not null)
        {
            var pen = new Pen(EdgeBrush, Math.Max(0.5, 1.0 / scale));
            dc.PushTransform(new MatrixTransform(matrix));
            dc.DrawGeometry(null, pen, _edgeGeometry);
            dc.Pop();
        }

        var route = RoutePolyline;
        if (route is { Count: > 1 })
        {
            var routeGeometry = new StreamGeometry();
            using (var ctx = routeGeometry.Open())
            {
                ctx.BeginFigure(new Point(route[0].X, route[0].Y), false, false);
                for (int i = 1; i < route.Count; i++)
                    ctx.LineTo(new Point(route[i].X, route[i].Y), true, false);
            }
            var routePen = new Pen(RouteBrush, Math.Max(1.5, 3.0 / scale)) { StartLineCap = PenLineCap.Round, EndLineCap = PenLineCap.Round };
            dc.PushTransform(new MatrixTransform(matrix));
            dc.DrawGeometry(null, routePen, routeGeometry);
            dc.Pop();
        }

        if (DestinationPosition is { } dest)
        {
            var p = matrix.Transform(new Point(dest.X, dest.Y));
            dc.DrawEllipse(DestinationBrush, CarOutline, p, 7, 7);
        }

        if (CarPosition is { } car)
        {
            var p = matrix.Transform(new Point(car.X, car.Y));
            dc.DrawEllipse(CarBrush, CarOutline, p, 6, 6);
        }
    }

    protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
    {
        base.OnMouseLeftButtonDown(e);
        var matrix = ComputeWorldToScreen();
        if (matrix.IsIdentity || !matrix.HasInverse)
            return;
        var inverse = matrix;
        inverse.Invert();
        var screenPoint = e.GetPosition(this);
        var worldPoint = inverse.Transform(screenPoint);
        MapClicked?.Invoke(this, new WorldPoint(worldPoint.X, worldPoint.Y));
        e.Handled = true;
    }
}
