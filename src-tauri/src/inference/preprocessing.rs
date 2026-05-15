//! Image preprocessing for ONNX model input.
//!
//! Handles resizing, normalization, and tensor creation.
//! Full-frame preprocessing uses letterbox padding to preserve aspect ratio.

use ndarray::Array4;

use crate::frame::Frame;

/// Normalization range for model input.
#[derive(Debug, Clone, Copy)]
pub enum NormalizeRange {
    /// Normalize to [0.0, 1.0] by dividing by 255
    ZeroOne,
    /// Normalize to [-1.0, 1.0] by (pixel / 127.5) - 1.0
    NegOneOne,
}

/// Letterbox padding parameters returned by `preprocess_frame_letterbox`.
///
/// Used to map model-space detections back to original image coordinates.
#[derive(Debug, Clone, Copy)]
pub struct LetterboxParams {
    /// Scale factor applied to the image.
    pub scale: f32,
    /// Horizontal padding (pixels in model input space, on each side).
    pub pad_x: f32,
    /// Vertical padding (pixels in model input space, on each side).
    pub pad_y: f32,
    /// Model input size.
    pub input_size: f32,
}

impl LetterboxParams {
    /// Convert a coordinate from model space [0..input_size] to normalized
    /// image space [0..1].
    pub fn to_image_x(&self, model_x: f32) -> f32 {
        (model_x - self.pad_x) / (self.input_size - 2.0 * self.pad_x)
    }

    /// Convert a coordinate from model space [0..input_size] to normalized
    /// image space [0..1].
    pub fn to_image_y(&self, model_y: f32) -> f32 {
        (model_y - self.pad_y) / (self.input_size - 2.0 * self.pad_y)
    }

    /// Convert a normalized [0..1] coordinate (relative to model input) to
    /// normalized image space.
    pub fn to_image_x_normalized(&self, nx: f32) -> f32 {
        self.to_image_x(nx * self.input_size)
    }

    /// Convert a normalized [0..1] coordinate (relative to model input) to
    /// normalized image space.
    pub fn to_image_y_normalized(&self, ny: f32) -> f32 {
        self.to_image_y(ny * self.input_size)
    }

    /// Convert a normalized width in model space to image space.
    pub fn to_image_w(&self, nw: f32) -> f32 {
        nw * self.input_size / (self.input_size - 2.0 * self.pad_x)
    }

    /// Convert a normalized height in model space to image space.
    pub fn to_image_h(&self, nh: f32) -> f32 {
        nh * self.input_size / (self.input_size - 2.0 * self.pad_y)
    }
}

/// Resize an RGB frame with letterbox padding and create a normalized tensor.
///
/// Preserves aspect ratio by scaling the image to fit within `target_size`
/// and padding the shorter dimension with black (zeros). Returns the tensor
/// and letterbox parameters for coordinate mapping.
///
/// ndarray `Array4` indexing with `[0, y, x, c]` is bounds-checked
/// by the loop range (`y < ts`, `x < ts`, `c < 3`).
#[allow(clippy::indexing_slicing, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn preprocess_frame_letterbox(
    frame: &Frame,
    target_size: u32,
    range: NormalizeRange,
) -> (Array4<f32>, LetterboxParams) {
    let ts = target_size as usize;
    let fw = frame.width as f32;
    let fh = frame.height as f32;

    // Compute scale to fit within target_size while preserving aspect ratio.
    let scale = (target_size as f32 / fw).min(target_size as f32 / fh);
    let new_w = (fw * scale) as u32;
    let new_h = (fh * scale) as u32;

    // Padding on each side.
    let pad_x = (target_size.saturating_sub(new_w)) / 2;
    let pad_y = (target_size.saturating_sub(new_h)) / 2;

    // Resize the frame to the scaled dimensions.
    let resized = resize_rgb(&frame.data, frame.width, frame.height, new_w, new_h);

    // Create tensor with letterbox padding (zeros = black).
    let mut tensor = Array4::<f32>::zeros((1, ts, ts, 3));

    let pad_value = match range {
        NormalizeRange::ZeroOne => 0.0,
        NormalizeRange::NegOneOne => -1.0,
    };

    // Fill with pad value (NegOneOne needs -1.0 for black, not 0.0).
    if matches!(range, NormalizeRange::NegOneOne) {
        tensor.fill(pad_value);
    }

    let nw = new_w as usize;
    for (pixel_idx, pixel) in resized.chunks_exact(3).enumerate() {
        let src_y = pixel_idx / nw;
        let src_x = pixel_idx % nw;

        let dst_y = src_y + pad_y as usize;
        let dst_x = src_x + pad_x as usize;

        if dst_y >= ts || dst_x >= ts {
            continue;
        }

        let Some(&r) = pixel.first() else { continue };
        let Some(&g) = pixel.get(1) else { continue };
        let Some(&b) = pixel.get(2) else { continue };

        let (r, g, b) = match range {
            NormalizeRange::ZeroOne => {
                (f32::from(r) / 255.0, f32::from(g) / 255.0, f32::from(b) / 255.0)
            }
            NormalizeRange::NegOneOne => (
                f32::from(r) / 127.5 - 1.0,
                f32::from(g) / 127.5 - 1.0,
                f32::from(b) / 127.5 - 1.0,
            ),
        };

        tensor[[0, dst_y, dst_x, 0]] = r;
        tensor[[0, dst_y, dst_x, 1]] = g;
        tensor[[0, dst_y, dst_x, 2]] = b;
    }

    let params = LetterboxParams {
        scale,
        pad_x: pad_x as f32,
        pad_y: pad_y as f32,
        input_size: target_size as f32,
    };

    (tensor, params)
}

/// Square ROI coordinates returned by `preprocess_roi`.
///
/// When the original ROI is non-square, `preprocess_roi` expands the shorter
/// dimension in pixel space to make a square crop. These are the actual
/// (square) coordinates that were fed to the model. Landmark outputs must
/// be remapped using these coordinates, **not** the original ROI.
#[derive(Debug, Clone, Copy)]
pub struct SquareRoi {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

/// Preprocess a cropped region of a frame (specified by a bounding box).
///
/// Makes the crop square in pixel space (expanding the shorter side) before
/// resizing to the model's target size. This avoids aspect ratio distortion
/// in ROI-based models.
///
/// Returns the preprocessed tensor and the actual square ROI coordinates
/// that were used for cropping. Use the returned `SquareRoi` (not the
/// original ROI) when mapping model-output landmarks back to image space.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn preprocess_roi(
    frame: &Frame,
    x_min: f32,
    y_min: f32,
    x_max: f32,
    y_max: f32,
    target_size: u32,
    range: NormalizeRange,
) -> (Array4<f32>, SquareRoi) {
    // Convert normalized ROI to pixel coordinates.
    let fw = frame.width as f32;
    let fh = frame.height as f32;

    let px_x_min = x_min * fw;
    let px_y_min = y_min * fh;
    let px_x_max = x_max * fw;
    let px_y_max = y_max * fh;

    let px_w = px_x_max - px_x_min;
    let px_h = px_y_max - px_y_min;

    // Make the crop square in pixel space by expanding the shorter dimension.
    let side = px_w.max(px_h);
    let cx = (px_x_min + px_x_max) / 2.0;
    let cy = (px_y_min + px_y_max) / 2.0;

    // Convert back to normalized coordinates, clamped to [0, 1].
    let sq_x_min = ((cx - side / 2.0) / fw).clamp(0.0, 1.0);
    let sq_y_min = ((cy - side / 2.0) / fh).clamp(0.0, 1.0);
    let sq_x_max = ((cx + side / 2.0) / fw).clamp(0.0, 1.0);
    let sq_y_max = ((cy + side / 2.0) / fh).clamp(0.0, 1.0);

    let crop = extract_roi(
        &frame.data,
        frame.width,
        frame.height,
        sq_x_min,
        sq_y_min,
        sq_x_max,
        sq_y_max,
    );

    let crop_w = ((sq_x_max - sq_x_min) * fw).max(1.0) as u32;
    let crop_h = ((sq_y_max - sq_y_min) * fh).max(1.0) as u32;

    let crop_frame = Frame::new(crop, crop_w, crop_h);
    let tensor = preprocess_frame_simple(&crop_frame, target_size, range);
    let square_roi = SquareRoi {
        x_min: sq_x_min,
        y_min: sq_y_min,
        x_max: sq_x_max,
        y_max: sq_y_max,
    };

    (tensor, square_roi)
}

/// Simple resize + normalize without letterbox padding.
///
/// Used for ROI crops that are already approximately square.
///
/// ndarray `Array4` indexing with `[0, y, x, c]` is bounds-checked
/// by the loop range (`y < ts`, `x < ts`, `c < 3`).
#[allow(clippy::indexing_slicing)]
fn preprocess_frame_simple(
    frame: &Frame,
    target_size: u32,
    range: NormalizeRange,
) -> Array4<f32> {
    let resized = resize_rgb(
        &frame.data,
        frame.width,
        frame.height,
        target_size,
        target_size,
    );

    let ts = target_size as usize;
    let mut tensor = Array4::<f32>::zeros((1, ts, ts, 3));

    for (pixel_idx, pixel) in resized.chunks_exact(3).enumerate() {
        let y = pixel_idx / ts;
        let x = pixel_idx % ts;
        if y >= ts || x >= ts {
            break;
        }

        let Some(&r) = pixel.first() else { continue };
        let Some(&g) = pixel.get(1) else { continue };
        let Some(&b) = pixel.get(2) else { continue };

        let (r, g, b) = match range {
            NormalizeRange::ZeroOne => {
                (f32::from(r) / 255.0, f32::from(g) / 255.0, f32::from(b) / 255.0)
            }
            NormalizeRange::NegOneOne => (
                f32::from(r) / 127.5 - 1.0,
                f32::from(g) / 127.5 - 1.0,
                f32::from(b) / 127.5 - 1.0,
            ),
        };

        tensor[[0, y, x, 0]] = r;
        tensor[[0, y, x, 1]] = g;
        tensor[[0, y, x, 2]] = b;
    }

    tensor
}

/// Extract a region of interest from an RGB buffer.
///
/// Coordinates are normalized [0.0, 1.0].
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn extract_roi(
    data: &[u8],
    width: u32,
    height: u32,
    x_min: f32,
    y_min: f32,
    x_max: f32,
    y_max: f32,
) -> Vec<u8> {
    // Casts are safe: values are clamped to [0, dimension].
    let x0 = (x_min.clamp(0.0, 1.0) * width as f32) as u32;
    let y0 = (y_min.clamp(0.0, 1.0) * height as f32) as u32;
    let x1 = (x_max.clamp(0.0, 1.0) * width as f32).min(width as f32) as u32;
    let y1 = (y_max.clamp(0.0, 1.0) * height as f32).min(height as f32) as u32;

    let roi_w = x1.saturating_sub(x0).max(1);
    let roi_h = y1.saturating_sub(y0).max(1);
    let mut crop = vec![0u8; (roi_w * roi_h * 3) as usize];

    for row in 0..roi_h {
        for col in 0..roi_w {
            let src_x = x0 + col;
            let src_y = y0 + row;
            if src_x < width && src_y < height {
                let src_idx = ((src_y * width + src_x) * 3) as usize;
                let dst_idx = ((row * roi_w + col) * 3) as usize;
                let src = data.get(src_idx..src_idx + 3);
                let dst = crop.get_mut(dst_idx..dst_idx + 3);
                if let (Some(s), Some(d)) = (src, dst) {
                    d.copy_from_slice(s);
                }
            }
        }
    }

    crop
}

/// Bilinear resize of an RGB buffer.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn resize_rgb(
    data: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Vec<u8> {
    if src_w == dst_w && src_h == dst_h {
        return data.to_vec();
    }

    let mut out = vec![0u8; (dst_w * dst_h * 3) as usize];
    let x_ratio = src_w as f32 / dst_w as f32;
    let y_ratio = src_h as f32 / dst_h as f32;

    for y in 0..dst_h {
        for x in 0..dst_w {
            let src_x = x as f32 * x_ratio;
            let src_y = y as f32 * y_ratio;

            // Truncation is intentional: we want the integer grid coordinate.
            let x0 = src_x as u32;
            let y0 = src_y as u32;
            let x1 = (x0 + 1).min(src_w.saturating_sub(1));
            let y1 = (y0 + 1).min(src_h.saturating_sub(1));

            let x_frac = src_x - x0 as f32;
            let y_frac = src_y - y0 as f32;

            let dst_idx = ((y * dst_w + x) * 3) as usize;

            for c in 0..3u32 {
                let get = |px: u32, py: u32| -> f32 {
                    let idx = ((py * src_w + px) * 3 + c) as usize;
                    data.get(idx).map_or(0.0, |&v| f32::from(v))
                };

                let top = get(x0, y0) * (1.0 - x_frac) + get(x1, y0) * x_frac;
                let bot = get(x0, y1) * (1.0 - x_frac) + get(x1, y1) * x_frac;
                let val = top * (1.0 - y_frac) + bot * y_frac;

                // Truncation is intentional: pixel value after clamp is in [0, 255].
                if let Some(dst) = out.get_mut(dst_idx + c as usize) {
                    *dst = val.clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    out
}

#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
mod tests {
    use super::*;

    fn make_test_frame(width: u32, height: u32) -> Frame {
        let pixel_count = (width * height) as usize;
        let mut data = vec![0u8; pixel_count * 3];
        for (i, chunk) in data.chunks_exact_mut(3).enumerate() {
            chunk[0] = (i % 256) as u8;
            chunk[1] = ((i * 2) % 256) as u8;
            chunk[2] = ((i * 3) % 256) as u8;
        }
        Frame::new(data, width, height)
    }

    #[test]
    fn preprocess_zero_one_range() {
        let frame = make_test_frame(4, 4);
        let (tensor, _params) = preprocess_frame_letterbox(&frame, 2, NormalizeRange::ZeroOne);
        assert_eq!(tensor.shape(), &[1, 2, 2, 3]);
        for val in tensor.iter() {
            assert!(*val >= 0.0 && *val <= 1.0, "value {val} out of [0, 1]");
        }
    }

    #[test]
    fn preprocess_neg_one_one_range() {
        let frame = make_test_frame(4, 4);
        let (tensor, _params) = preprocess_frame_letterbox(&frame, 2, NormalizeRange::NegOneOne);
        assert_eq!(tensor.shape(), &[1, 2, 2, 3]);
        for val in tensor.iter() {
            assert!(
                *val >= -1.0 && *val <= 1.01,
                "value {val} out of [-1, 1]"
            );
        }
    }

    #[test]
    fn preprocess_preserves_identity_size() {
        let frame = make_test_frame(192, 192);
        let (tensor, params) = preprocess_frame_letterbox(&frame, 192, NormalizeRange::ZeroOne);
        assert_eq!(tensor.shape(), &[1, 192, 192, 3]);
        // Square input should have no padding.
        assert!((params.pad_x).abs() < 1.0);
        assert!((params.pad_y).abs() < 1.0);
    }

    #[test]
    fn letterbox_preserves_aspect_ratio() {
        // 640x480 (4:3) -> 192x192 with letterbox.
        let frame = make_test_frame(640, 480);
        let (_tensor, params) = preprocess_frame_letterbox(&frame, 192, NormalizeRange::ZeroOne);
        // Scale should be 192/640 = 0.3, producing 192x144. Pad top/bottom = (192-144)/2 = 24.
        assert!((params.scale - 0.3).abs() < 0.01);
        assert!((params.pad_x).abs() < 1.0); // No horizontal padding
        assert!((params.pad_y - 24.0).abs() < 1.0); // 24px vertical padding
    }

    #[test]
    fn letterbox_coordinate_roundtrip() {
        let frame = make_test_frame(640, 480);
        let (_tensor, params) = preprocess_frame_letterbox(&frame, 192, NormalizeRange::ZeroOne);
        // Center of image (0.5, 0.5) should map to ~center of letterboxed image.
        let mx = 0.5 * params.input_size;
        let my = params.pad_y + 0.5 * (params.input_size - 2.0 * params.pad_y);
        let ix = params.to_image_x(mx);
        let iy = params.to_image_y(my);
        assert!((ix - 0.5).abs() < 0.05, "ix={ix}");
        assert!((iy - 0.5).abs() < 0.05, "iy={iy}");
    }

    #[test]
    fn extract_roi_basic() {
        // 4x4 white image.
        let data = vec![255u8; 4 * 4 * 3];
        let roi = extract_roi(&data, 4, 4, 0.25, 0.25, 0.75, 0.75);
        // ROI should be 2x2 = 12 bytes.
        assert_eq!(roi.len(), 2 * 2 * 3);
        assert!(roi.iter().all(|&v| v == 255));
    }

    #[test]
    fn resize_same_size_is_identity() {
        let data = vec![128u8; 4 * 4 * 3];
        let out = resize_rgb(&data, 4, 4, 4, 4);
        assert_eq!(out, data);
    }

    #[test]
    fn preprocess_roi_returns_square_roi_for_nonsquare_input() {
        // 640x480 frame, non-square ROI (wider than tall in normalized coords).
        let frame = make_test_frame(640, 480);
        let (_tensor, sq) = preprocess_roi(
            &frame,
            0.3125, // x_min
            0.2125, // y_min
            0.6875, // x_max
            0.5875, // y_max
            192,
            NormalizeRange::ZeroOne,
        );

        // Original ROI: w=0.375 (240px), h=0.375 (180px).
        // In pixel space, 240 > 180, so preprocess_roi expands vertically to 240px.
        // The square ROI should be wider in y (normalized) than the original.
        let sq_w = sq.x_max - sq.x_min;
        let sq_h = sq.y_max - sq.y_min;

        // x dimensions should stay the same (it was the larger dimension).
        assert!((sq.x_min - 0.3125).abs() < 0.01, "sq.x_min={}", sq.x_min);
        assert!((sq.x_max - 0.6875).abs() < 0.01, "sq.x_max={}", sq.x_max);

        // y dimensions should be expanded (sq_h > original 0.375).
        assert!(sq_h > 0.375 + 0.01, "sq_h={sq_h} should be > 0.375");

        // The crop should be square in pixel space: sq_w * 640 == sq_h * 480.
        let px_w = sq_w * 640.0;
        let px_h = sq_h * 480.0;
        assert!(
            (px_w - px_h).abs() < 2.0,
            "pixel crop should be square: {px_w} x {px_h}"
        );
    }

    #[test]
    fn preprocess_roi_square_roi_matches_original_for_square_input() {
        // Square frame: ROI should be unchanged.
        let frame = make_test_frame(480, 480);
        let (_tensor, sq) = preprocess_roi(
            &frame,
            0.2, 0.3, 0.6, 0.7,
            192,
            NormalizeRange::ZeroOne,
        );

        // On a square frame, normalized square ROI is also square in pixel space.
        assert!((sq.x_min - 0.2).abs() < 0.01, "sq.x_min={}", sq.x_min);
        assert!((sq.y_min - 0.3).abs() < 0.01, "sq.y_min={}", sq.y_min);
        assert!((sq.x_max - 0.6).abs() < 0.01, "sq.x_max={}", sq.x_max);
        assert!((sq.y_max - 0.7).abs() < 0.01, "sq.y_max={}", sq.y_max);
    }
}
