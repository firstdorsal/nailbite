//! Camera capture module for V4L2-based frame acquisition.

#[cfg(target_os = "linux")]
mod v4l2;

#[cfg(target_os = "linux")]
pub use v4l2::CameraCapture;

use crate::frame::Frame;
use thiserror::Error;

/// Camera capture errors.
#[derive(Debug, Error)]
pub enum CameraError {
    #[error("Failed to open camera device: {0}")]
    DeviceOpen(String),
    #[error("Failed to configure camera: {0}")]
    Configuration(String),
    #[error("Failed to capture frame: {0}")]
    Capture(String),
    #[error("Failed to convert frame format: {0}")]
    Conversion(String),
    #[error("Camera not available on this platform")]
    NotAvailable,
}

/// A captured frame with metadata.
#[derive(Debug, Clone)]
pub struct CapturedFrame {
    /// The RGB frame data.
    pub frame: Frame,
    /// Timestamp when the frame was captured (monotonic).
    pub timestamp: std::time::Instant,
}
