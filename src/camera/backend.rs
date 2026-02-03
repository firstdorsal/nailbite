use crate::camera::frame::Frame;
use crate::errors::CameraError;

/// Describes a discovered camera device.
#[derive(Debug, Clone)]
pub struct CameraDevice {
    /// Platform-specific device identifier (e.g., "/dev/video0" on Linux)
    pub id: String,
    /// Human-readable name if available
    pub name: Option<String>,
}

/// Platform-agnostic camera capture interface.
///
/// Implementations exist per platform:
/// - Linux: `V4lBackend` (V4L2 via the `v4l` crate)
/// - Windows/macOS: future implementations
pub trait CameraBackend: Send {
    /// Open a camera device with the specified parameters.
    fn open(device: &str, width: u32, height: u32, fps: u32) -> Result<Self, CameraError>
    where
        Self: Sized;

    /// Capture a single frame from the camera.
    fn capture_frame(&mut self) -> Result<Frame, CameraError>;

    /// List available camera devices on this system.
    fn device_list() -> Result<Vec<CameraDevice>, CameraError>
    where
        Self: Sized;
}
