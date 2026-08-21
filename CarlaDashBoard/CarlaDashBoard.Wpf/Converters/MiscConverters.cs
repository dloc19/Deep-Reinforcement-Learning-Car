using System.Globalization;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;
using CarlaDashBoard.Wpf.ViewModels;

namespace CarlaDashBoard.Wpf.Converters;

/// <summary>Shows an element only when the bound <see cref="CameraDisplayMode"/> matches the
/// converter parameter ("Rgb" | "Segmentation" | "Split") — drives Hình 02's RGB/Seg/Split panes.</summary>
public sealed class CameraModeToVisibilityConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is not CameraDisplayMode mode || parameter is not string wanted)
            return Visibility.Collapsed;
        return Enum.TryParse<CameraDisplayMode>(wanted, out var target) && target == mode
            ? Visibility.Visible
            : Visibility.Collapsed;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        throw new NotSupportedException();
}

public sealed class BoolToStringConverter : IValueConverter
{
    // parameter format: "TrueText|FalseText"
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        var parts = (parameter as string)?.Split('|') ?? new[] { "Có", "Không" };
        var isTrue = value is bool b && b;
        return isTrue ? parts[0] : (parts.Length > 1 ? parts[1] : parts[0]);
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        throw new NotSupportedException();
}

/// <summary>null -> Collapsed, non-null -> Visible. Pass ConverterParameter="inverse" to flip.</summary>
public sealed class NullToVisibilityConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        var isNull = value is null;
        if (string.Equals(parameter as string, "inverse", StringComparison.OrdinalIgnoreCase))
            isNull = !isNull;
        return isNull ? Visibility.Collapsed : Visibility.Visible;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        throw new NotSupportedException();
}

/// <summary>True when the bound value equals the converter parameter (by string comparison) —
/// used to highlight the selected mode tile without a full ListBox item-container style.</summary>
public sealed class EqualityToBoolConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture) =>
        Equals(value?.ToString(), parameter?.ToString());

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        throw new NotSupportedException();
}

/// <summary>bool -> Visibility. Pass ConverterParameter="inverse" to flip (unlike the stock
/// BooleanToVisibilityConverter, which ignores its parameter entirely).</summary>
public sealed class BoolToVisibilityConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        var isTrue = value is bool b && b;
        if (string.Equals(parameter as string, "inverse", StringComparison.OrdinalIgnoreCase))
            isTrue = !isTrue;
        return isTrue ? Visibility.Visible : Visibility.Collapsed;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        throw new NotSupportedException();
}

/// <summary>bool -> SolidColorBrush. parameter format: "#TrueHex|#FalseHex".</summary>
public sealed class BoolToBrushConverter : IValueConverter
{
    public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        var parts = (parameter as string)?.Split('|') ?? new[] { "#5EC98A", "#6F7A83" };
        var isTrue = value is bool b && b;
        var hex = isTrue ? parts[0] : (parts.Length > 1 ? parts[1] : parts[0]);
        return new SolidColorBrush((Color)ColorConverter.ConvertFromString(hex));
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture) =>
        throw new NotSupportedException();
}
