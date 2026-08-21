using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;
using CarlaDashBoard.Wpf.Services;

namespace CarlaDashBoard.Wpf.Converters;

public sealed class ConnectionStateToBrushConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture) =>
        value is ConnectionState state
            ? state switch
            {
                ConnectionState.Connected => new SolidColorBrush(Color.FromRgb(0x5E, 0xC9, 0x8A)),
                ConnectionState.Connecting or ConnectionState.Reconnecting =>
                    new SolidColorBrush(Color.FromRgb(0xE8, 0xA3, 0x3D)),
                _ => new SolidColorBrush(Color.FromRgb(0x6F, 0x7A, 0x83)),
            }
            : Brushes.Gray;

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        throw new NotSupportedException();
}

public sealed class ConnectionStateToTextConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture) =>
        value is ConnectionState state
            ? state switch
            {
                ConnectionState.Connected => "Đã kết nối",
                ConnectionState.Connecting => "Đang kết nối…",
                ConnectionState.Reconnecting => "Đang kết nối lại…",
                _ => "Chưa kết nối",
            }
            : "—";

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        throw new NotSupportedException();
}
