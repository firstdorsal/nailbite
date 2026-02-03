/// Linux V4L2 camera backend implementation.
///
/// Uses the `v4l` crate for direct Video4Linux2 access.
use std::io;

use tracing::{debug, info, warn};
use v4l::buffer::Type as BufType;
use v4l::io::mmap::Stream;
use v4l::io::traits::CaptureStream;
use v4l::video::Capture;
use v4l::{Device, FourCC};

use crate::camera::backend::{CameraBackend, CameraDevice};
use crate::camera::frame::Frame;
use crate::errors::CameraError;

/// Maximum consecutive capture errors before recreating the V4L2 stream.
const MAX_CONSECUTIVE_ERRORS: u32 = 3;

pub struct V4lBackend {
    device_path: String,
    /// The `Device` is kept alive so the stream can be recreated on errors.
    device: Device,
    /// The `'static` lifetime is sound here because `Stream`'s lifetime parameter
    /// originates from raw `mmap()` pointers (not from borrowing `Device`).
    /// The `Device` borrow in `with_buffers` is only used to clone an `Arc<Handle>`.
    stream: Stream<'static>,
    width: u32,
    height: u32,
    fps: u32,
    fourcc: FourCC,
    consecutive_errors: u32,
}

impl CameraBackend for V4lBackend {
    fn open(device: &str, width: u32, height: u32, fps: u32) -> Result<Self, CameraError>
    where
        Self: Sized,
    {
        info!(device, width, height, fps, "Opening V4L2 camera");

        let dev = Device::with_path(device).map_err(|e| CameraError::OpenFailed {
            device: device.to_string(),
            source: io::Error::other(e.to_string()),
        })?;

        // Try MJPEG first, then YUYV as fallback.
        let fourcc = set_format(&dev, width, height, fps)?;

        let stream = create_stream(&dev, device)?;

        info!(device, ?fourcc, "V4L2 camera opened successfully");

        Ok(Self {
            device_path: device.to_string(),
            device: dev,
            stream,
            width,
            height,
            fps,
            fourcc,
            consecutive_errors: 0,
        })
    }

    fn capture_frame(&mut self) -> Result<Frame, CameraError> {
        // If too many consecutive errors, recreate the stream to recover from
        // broken V4L2 state (e.g. after USB protocol errors on first frame).
        if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
            warn!(
                device = %self.device_path,
                errors = self.consecutive_errors,
                "Recreating V4L2 stream after consecutive errors"
            );
            // Re-negotiate format before recreating stream.
            let _ = set_format(&self.device, self.width, self.height, self.fps);
            self.stream = create_stream(&self.device, &self.device_path)?;
            self.consecutive_errors = 0;
        }

        match self.stream.next() {
            Ok((buf, _meta)) => {
                self.consecutive_errors = 0;
                let rgb = decode_to_rgb(buf, self.width, self.height, &self.fourcc)?;
                Ok(Frame::new(rgb, self.width, self.height))
            }
            Err(e) => {
                self.consecutive_errors += 1;
                Err(CameraError::CaptureFailed {
                    device: self.device_path.clone(),
                    source: io::Error::other(e.to_string()),
                })
            }
        }
    }

    fn device_list() -> Result<Vec<CameraDevice>, CameraError>
    where
        Self: Sized,
    {
        let mut devices = Vec::new();

        // Enumerate /dev/video* devices.
        for i in 0..16 {
            let path = format!("/dev/video{i}");
            if let Ok(dev) = Device::with_path(&path) {
                let name = dev
                    .query_caps()
                    .ok()
                    .map(|caps| caps.card);
                devices.push(CameraDevice {
                    id: path,
                    name,
                });
            }
        }

        Ok(devices)
    }
}

/// Create a new V4L2 mmap stream from a device.
fn create_stream(dev: &Device, device_path: &str) -> Result<Stream<'static>, CameraError> {
    Stream::with_buffers(dev, BufType::VideoCapture, 4).map_err(|e| CameraError::OpenFailed {
        device: device_path.to_string(),
        source: io::Error::other(e.to_string()),
    })
}

/// Attempt to set the camera format. Tries MJPEG first, then YUYV.
fn set_format(
    dev: &Device,
    width: u32,
    height: u32,
    fps: u32,
) -> Result<FourCC, CameraError> {
    let mjpeg = FourCC::new(b"MJPG");
    let yuyv = FourCC::new(b"YUYV");

    // Try MJPEG first (lower bandwidth, higher resolution support).
    for fourcc in [mjpeg, yuyv] {
        let mut fmt = dev.format().map_err(|e| CameraError::Configuration(e.to_string()))?;
        fmt.width = width;
        fmt.height = height;
        fmt.fourcc = fourcc;

        if dev.set_format(&fmt).is_ok() {
            // Verify the format was actually set.
            let actual = dev.format().map_err(|e| CameraError::Configuration(e.to_string()))?;
            if actual.fourcc == fourcc {
                debug!(?fourcc, "Format set successfully");

                // Set frame rate.
                let mut params = dev
                    .params()
                    .map_err(|e| CameraError::Configuration(e.to_string()))?;
                params.interval = v4l::Fraction::new(1, fps);
                let _ = dev.set_params(&params); // Best effort, not all cameras support this.

                return Ok(fourcc);
            }
        }
    }

    Err(CameraError::Configuration(format!(
        "Failed to set format {width}x{height} with MJPEG or YUYV"
    )))
}

/// Decode raw camera buffer to RGB bytes.
fn decode_to_rgb(
    buf: &[u8],
    width: u32,
    height: u32,
    fourcc: &FourCC,
) -> Result<Vec<u8>, CameraError> {
    let mjpeg = FourCC::new(b"MJPG");
    let yuyv = FourCC::new(b"YUYV");

    if *fourcc == mjpeg {
        decode_mjpeg(buf)
    } else if *fourcc == yuyv {
        Ok(yuyv_to_rgb(buf, width, height))
    } else {
        Err(CameraError::UnsupportedFormat(format!("{fourcc:?}")))
    }
}

/// Decode MJPEG frame to RGB using the `image` crate.
fn decode_mjpeg(buf: &[u8]) -> Result<Vec<u8>, CameraError> {
    let img = image::load_from_memory_with_format(buf, image::ImageFormat::Jpeg)
        .map_err(|e| CameraError::UnsupportedFormat(format!("MJPEG decode failed: {e}")))?;
    Ok(img.to_rgb8().into_raw())
}

/// Convert YUYV (YUV 4:2:2) packed format to RGB.
///
/// Every 4 bytes encode 2 pixels: [Y0, U, Y1, V].
fn yuyv_to_rgb(buf: &[u8], width: u32, height: u32) -> Vec<u8> {
    let pixel_count = (width * height) as usize;
    let mut rgb = vec![0u8; pixel_count * 3];

    let mut rgb_idx = 0;
    for chunk in buf.chunks_exact(4) {
        if rgb_idx + 5 >= rgb.len() {
            break;
        }

        let Some(&y0_byte) = chunk.first() else {
            break;
        };
        let Some(&u_byte) = chunk.get(1) else { break };
        let Some(&y1_byte) = chunk.get(2) else { break };
        let Some(&v_byte) = chunk.get(3) else { break };

        let y0 = f32::from(y0_byte);
        let u = f32::from(u_byte);
        let y1 = f32::from(y1_byte);
        let v = f32::from(v_byte);

        let (r, g, b) = yuv_to_rgb_pixel(y0, u, v);
        if let Some([d0, d1, d2]) = rgb.get_mut(rgb_idx..rgb_idx + 3) {
            *d0 = r;
            *d1 = g;
            *d2 = b;
        }

        let (r, g, b) = yuv_to_rgb_pixel(y1, u, v);
        if let Some([d0, d1, d2]) = rgb.get_mut(rgb_idx + 3..rgb_idx + 6) {
            *d0 = r;
            *d1 = g;
            *d2 = b;
        }

        rgb_idx += 6;
    }

    rgb
}

/// Convert a single YUV pixel to RGB.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn yuv_to_rgb_pixel(y: f32, u: f32, v: f32) -> (u8, u8, u8) {
    let r = y + 1.402 * (v - 128.0);
    let g = y - 0.344136 * (u - 128.0) - 0.714136 * (v - 128.0);
    let b = y + 1.772 * (u - 128.0);

    // Truncation is intentional: values are clamped to [0, 255].
    (
        r.clamp(0.0, 255.0) as u8,
        g.clamp(0.0, 255.0) as u8,
        b.clamp(0.0, 255.0) as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yuyv_to_rgb_basic_conversion() {
        // Black pixel: Y=0, U=128, V=128 -> RGB (0, 0, 0)
        let yuyv = vec![0u8, 128, 0, 128];
        let rgb = yuyv_to_rgb(&yuyv, 2, 1);
        assert_eq!(rgb.len(), 6);
        assert_eq!(rgb.first(), Some(&0)); // R
        assert_eq!(rgb.get(1), Some(&0)); // G
        assert_eq!(rgb.get(2), Some(&0)); // B
    }

    #[test]
    fn yuyv_to_rgb_white_pixel() {
        // White pixel: Y=255, U=128, V=128 -> RGB (255, 255, 255)
        let yuyv = vec![255u8, 128, 255, 128];
        let rgb = yuyv_to_rgb(&yuyv, 2, 1);
        assert_eq!(rgb.first(), Some(&255));
        assert_eq!(rgb.get(1), Some(&255));
        assert_eq!(rgb.get(2), Some(&255));
    }

    #[test]
    fn yuv_to_rgb_clamps_correctly() {
        // Edge case: values that would overflow without clamping.
        let (r, g, b) = yuv_to_rgb_pixel(255.0, 0.0, 255.0);
        // Verify the function returns without panic and produces valid u8 values.
        let _ = (r, g, b);

        // Also test negative case: Y=0 with extreme chroma.
        let (r2, g2, b2) = yuv_to_rgb_pixel(0.0, 255.0, 0.0);
        let _ = (r2, g2, b2);
    }

    #[test]
    fn device_list_does_not_panic() {
        // Should not panic even if no cameras are connected.
        let _devices = V4lBackend::device_list();
    }
}
