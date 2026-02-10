//! Frame type for image data.

use std::io::Cursor;

use image::{ImageBuffer, Rgb};

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

    /// Decode a JPEG image into an RGB frame.
    pub fn from_jpeg(jpeg_data: &[u8]) -> Result<Self, image::ImageError> {
        let img = image::load_from_memory_with_format(jpeg_data, image::ImageFormat::Jpeg)?;
        let rgb = img.to_rgb8();
        let width = rgb.width();
        let height = rgb.height();
        let data = rgb.into_raw();
        Ok(Self::new(data, width, height))
    }

    /// Encode frame to JPEG format.
    pub fn to_jpeg(&self) -> Result<Vec<u8>, image::ImageError> {
        let img: ImageBuffer<Rgb<u8>, _> =
            ImageBuffer::from_raw(self.width, self.height, self.data.clone()).ok_or_else(|| {
                image::ImageError::Parameter(image::error::ParameterError::from_kind(
                    image::error::ParameterErrorKind::DimensionMismatch,
                ))
            })?;

        let mut buffer = Cursor::new(Vec::new());
        img.write_to(&mut buffer, image::ImageFormat::Jpeg)?;
        Ok(buffer.into_inner())
    }

    /// Encode frame to JPEG and return as base64 string.
    pub fn to_base64_jpeg(&self) -> Result<String, image::ImageError> {
        use base64::Engine;
        let jpeg = self.to_jpeg()?;
        Ok(base64::engine::general_purpose::STANDARD.encode(&jpeg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_new_stores_dimensions() {
        let frame = Frame::new(vec![0u8; 640 * 480 * 3], 640, 480);
        assert_eq!(frame.width, 640);
        assert_eq!(frame.height, 480);
        assert_eq!(frame.data.len(), 640 * 480 * 3);
    }

    #[test]
    fn frame_jpeg_roundtrip() {
        // Create a small test frame (10x10 red image)
        let mut data = Vec::with_capacity(10 * 10 * 3);
        for _ in 0..(10 * 10) {
            data.extend_from_slice(&[255, 0, 0]); // Red pixels
        }
        let frame = Frame::new(data, 10, 10);

        // Encode to JPEG
        let jpeg = frame.to_jpeg().expect("JPEG encode failed");
        assert!(!jpeg.is_empty());

        // Decode back
        let decoded = Frame::from_jpeg(&jpeg).expect("JPEG decode failed");
        assert_eq!(decoded.width, 10);
        assert_eq!(decoded.height, 10);
    }
}
