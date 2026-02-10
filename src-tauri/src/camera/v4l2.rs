//! V4L2 camera capture implementation for Linux.

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tracing::{debug, error, info, warn};
use v4l::buffer::Type;
use v4l::io::mmap::Stream;
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;
use v4l::FourCC;

use crate::camera::{CameraError, CapturedFrame};
use crate::frame::Frame;

/// V4L2 camera capture handle.
pub struct CameraCapture {
    device_path: String,
    width: u32,
    height: u32,
    running: Arc<AtomicBool>,
    latest_frame: Arc<Mutex<Option<CapturedFrame>>>,
    capture_thread: Option<std::thread::JoinHandle<()>>,
}

impl CameraCapture {
    /// Create a new camera capture for the specified device.
    pub fn new(device_path: &str, width: u32, height: u32) -> Self {
        Self {
            device_path: device_path.to_string(),
            width,
            height,
            running: Arc::new(AtomicBool::new(false)),
            latest_frame: Arc::new(Mutex::new(None)),
            capture_thread: None,
        }
    }

    /// Start capturing frames in a background thread.
    pub fn start(&mut self) -> Result<(), CameraError> {
        if self.running.load(Ordering::Relaxed) {
            return Ok(()); // Already running
        }

        let device_path = self.device_path.clone();
        let width = self.width;
        let height = self.height;
        let running = Arc::clone(&self.running);
        let latest_frame = Arc::clone(&self.latest_frame);

        // Test device access before spawning thread
        let dev = Device::with_path(&device_path).map_err(|e| {
            CameraError::DeviceOpen(format!("{}: {}", device_path, e))
        })?;

        // Configure format
        let mut format = dev.format().map_err(|e| {
            CameraError::Configuration(format!("Failed to get format: {}", e))
        })?;

        format.width = width;
        format.height = height;
        // Prefer MJPEG for efficiency, fall back to YUYV
        format.fourcc = FourCC::new(b"MJPG");

        let format = dev.set_format(&format).map_err(|e| {
            CameraError::Configuration(format!("Failed to set format: {}", e))
        })?;

        info!(
            device = %device_path,
            width = format.width,
            height = format.height,
            fourcc = ?format.fourcc,
            "Camera configured"
        );

        let is_mjpeg = format.fourcc == FourCC::new(b"MJPG");
        let actual_width = format.width;
        let actual_height = format.height;

        drop(dev); // Close device, will reopen in thread

        running.store(true, Ordering::Relaxed);

        let handle = std::thread::Builder::new()
            .name("camera-capture".into())
            .spawn(move || {
                if let Err(e) = capture_loop(
                    &device_path,
                    actual_width,
                    actual_height,
                    is_mjpeg,
                    &running,
                    &latest_frame,
                ) {
                    error!(error = %e, "Camera capture loop failed");
                }
                running.store(false, Ordering::Relaxed);
            })
            .map_err(|e| CameraError::DeviceOpen(format!("Failed to spawn thread: {}", e)))?;

        self.capture_thread = Some(handle);
        Ok(())
    }

    /// Stop capturing frames.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.capture_thread.take() {
            let _ = handle.join();
        }
    }

    /// Check if the camera is currently capturing.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Get the latest captured frame, if available.
    pub fn get_frame(&self) -> Option<CapturedFrame> {
        self.latest_frame.lock().clone()
    }

    /// Take the latest captured frame, clearing it from the buffer.
    pub fn take_frame(&self) -> Option<CapturedFrame> {
        self.latest_frame.lock().take()
    }
}

impl Drop for CameraCapture {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Main capture loop running in background thread.
fn capture_loop(
    device_path: &str,
    width: u32,
    height: u32,
    is_mjpeg: bool,
    running: &AtomicBool,
    latest_frame: &Mutex<Option<CapturedFrame>>,
) -> Result<(), CameraError> {
    let dev = Device::with_path(device_path).map_err(|e| {
        CameraError::DeviceOpen(format!("{}: {}", device_path, e))
    })?;

    let mut stream = Stream::with_buffers(&dev, Type::VideoCapture, 4).map_err(|e| {
        CameraError::Configuration(format!("Failed to create stream: {}", e))
    })?;

    info!("Camera capture loop started");

    while running.load(Ordering::Relaxed) {
        match stream.next() {
            Ok((buf, _meta)) => {
                let timestamp = Instant::now();

                let frame = if is_mjpeg {
                    // Decode MJPEG frame
                    match Frame::from_jpeg(buf) {
                        Ok(f) => f,
                        Err(e) => {
                            warn!(error = %e, "Failed to decode MJPEG frame");
                            continue;
                        }
                    }
                } else {
                    // Convert YUYV to RGB
                    match yuyv_to_rgb(buf, width, height) {
                        Ok(rgb_data) => Frame::new(rgb_data, width, height),
                        Err(e) => {
                            warn!(error = %e, "Failed to convert YUYV frame");
                            continue;
                        }
                    }
                };

                debug!(
                    width = frame.width,
                    height = frame.height,
                    "Frame captured"
                );

                *latest_frame.lock() = Some(CapturedFrame { frame, timestamp });
            }
            Err(e) => {
                // EAGAIN is normal for non-blocking I/O
                if e.kind() != io::ErrorKind::WouldBlock {
                    warn!(error = %e, "Frame capture error");
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }

    info!("Camera capture loop stopped");
    Ok(())
}

/// Convert YUYV (YUV 4:2:2) to RGB.
fn yuyv_to_rgb(yuyv: &[u8], width: u32, height: u32) -> Result<Vec<u8>, CameraError> {
    let expected_len = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().map(|h| w * h * 2))
        .ok_or_else(|| CameraError::Conversion("Dimensions too large".to_string()))?;

    if yuyv.len() < expected_len {
        return Err(CameraError::Conversion(format!(
            "YUYV buffer too small: {} < {}",
            yuyv.len(),
            expected_len
        )));
    }

    let pixel_count = expected_len / 2;
    let mut rgb = vec![0u8; pixel_count * 3];

    // Process YUYV in chunks of 4 bytes (2 pixels)
    for (chunk_idx, chunk) in yuyv.chunks_exact(4).enumerate().take(pixel_count / 2) {
        // chunks_exact guarantees exactly 4 elements, destructure to satisfy clippy
        let (&y0_byte, &u_byte, &y1_byte, &v_byte) = match chunk {
            [y0, u, y1, v] => (y0, u, y1, v),
            _ => continue, // unreachable due to chunks_exact(4)
        };

        let y0 = f32::from(y0_byte);
        let u = f32::from(u_byte) - 128.0;
        let y1 = f32::from(y1_byte);
        let v = f32::from(v_byte) - 128.0;

        // First pixel - clamp guarantees range [0, 255], safe to truncate
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let r0 = (y0 + 1.402 * v).clamp(0.0, 255.0) as u8;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let g0 = (y0 - 0.344 * u - 0.714 * v).clamp(0.0, 255.0) as u8;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let b0 = (y0 + 1.772 * u).clamp(0.0, 255.0) as u8;

        // Second pixel
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let r1 = (y1 + 1.402 * v).clamp(0.0, 255.0) as u8;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let g1 = (y1 - 0.344 * u - 0.714 * v).clamp(0.0, 255.0) as u8;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let b1 = (y1 + 1.772 * u).clamp(0.0, 255.0) as u8;

        let rgb_base = chunk_idx * 6;
        // get_mut returns exactly 6-element slice, copy using array pattern
        if let Some(out) = rgb.get_mut(rgb_base..rgb_base + 6) {
            out.copy_from_slice(&[r0, g0, b0, r1, g1, b1]);
        }
    }

    Ok(rgb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yuyv_conversion_size() {
        // 2x2 YUYV image (8 bytes)
        let yuyv = vec![128u8; 8]; // Neutral grey
        let result = yuyv_to_rgb(&yuyv, 2, 2);
        assert!(result.is_ok());
        let rgb = result.unwrap();
        assert_eq!(rgb.len(), 12); // 2x2x3
    }

    #[test]
    fn yuyv_buffer_too_small() {
        let yuyv = vec![0u8; 4];
        let result = yuyv_to_rgb(&yuyv, 2, 2);
        assert!(result.is_err());
    }
}
