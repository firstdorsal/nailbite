/// A captured camera frame in RGB format.
#[derive(Debug, Clone)]
pub struct Frame {
    /// RGB pixel data (row-major, 3 bytes per pixel)
    pub data: Vec<u8>,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
}

impl Frame {
    /// Create a new frame from RGB data.
    pub fn new(data: Vec<u8>, width: u32, height: u32) -> Self {
        Self {
            data,
            width,
            height,
        }
    }
}
