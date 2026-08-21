using System.IO;
using System.Windows.Media.Imaging;

namespace CarlaDashBoard.Wpf.Infrastructure;

/// <summary>
/// Decodes a camera JPEG off the UI thread. <see cref="BitmapImage"/> is a Freezable — once
/// frozen it can be created on a background thread and handed to the UI thread for display
/// without any further marshalling, which is the simplest way in WPF to keep JPEG decode off
/// the dispatcher at 15+ fps (the design doc's §03/§11 note about not decoding on the UI
/// thread — this is the pragmatic one-object-per-frame version of that; a manual
/// WriteableBitmap.WritePixels pipeline would allocate less per frame but isn't needed at this
/// resolution/frame rate).
/// </summary>
public static class JpegFrameDecoder
{
    public static BitmapImage? Decode(byte[] jpegBytes)
    {
        try
        {
            using var stream = new MemoryStream(jpegBytes);
            var image = new BitmapImage();
            image.BeginInit();
            image.CacheOption = BitmapCacheOption.OnLoad;   // fully load, so stream can close
            image.StreamSource = stream;
            image.EndInit();
            image.Freeze();                                  // now safe to touch from the UI thread
            return image;
        }
        catch (Exception)
        {
            return null; // a torn/partial frame — just skip it, the next one is seconds away
        }
    }
}
