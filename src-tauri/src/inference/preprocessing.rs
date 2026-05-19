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

/// Rotated square ROI in source pixel space.
///
/// MediaPipe's hand landmark model expects rotation-normalised crops: the
/// hand pre-rotated so the fingers point up. Feeding it axis-aligned crops
/// at arbitrary angles puts the input out of distribution and the model
/// returns garbage geometry / unstable confidences. This struct captures
/// the centre, side length, and orientation of the crop so we can sample
/// the source frame along rotated axes and inverse-rotate the model output
/// back into image coordinates.
///
/// `rotation_rad` is the angle (counter-clockwise, image-coord convention
/// where +y points down) at which the crop's local "+x" axis sits relative
/// to image "+x". In other words, when sampling, the crop's local up axis
/// (decreasing v) corresponds to image direction
/// `(sin(rotation_rad), -cos(rotation_rad))` — exactly the wrist→middle-MCP
/// direction emitted by the palm detector when `rotation_rad` is computed
/// via [`RotatedRoi::from_keypoints`].
#[derive(Debug, Clone, Copy)]
pub struct RotatedRoi {
    /// Centre of the crop in source pixel coordinates.
    pub cx_px: f32,
    pub cy_px: f32,
    /// Side length of the square crop, in source pixels.
    pub size_px: f32,
    /// Rotation in radians (counter-clockwise in image coords).
    pub rotation_rad: f32,
}

impl RotatedRoi {
    /// Build a rotated ROI from two image-normalised points along the hand
    /// axis (wrist → middle-finger MCP, or elbow → wrist) and a square side
    /// length in pixels.
    ///
    /// The "up" axis of the resulting crop points along `from → to` in the
    /// source frame. The crop is centred at `centre` (not at the midpoint
    /// of the two points) so callers can decide whether to bias the box
    /// toward the fingers or the palm.
    pub fn from_axis(
        centre_nx: f32,
        centre_ny: f32,
        from_nx: f32,
        from_ny: f32,
        to_nx: f32,
        to_ny: f32,
        size_px: f32,
        frame_w: u32,
        frame_h: u32,
    ) -> Self {
        let dx = to_nx - from_nx;
        let dy = to_ny - from_ny;
        // theta is the image-coord angle of the `from → to` vector. We
        // want the crop's "up" axis (image dir `(sin R, -cos R)`) to align
        // with that vector, which gives R = theta + π/2.
        let theta = dy.atan2(dx);
        let rotation_rad = theta + std::f32::consts::FRAC_PI_2;
        Self {
            cx_px: centre_nx * frame_w as f32,
            cy_px: centre_ny * frame_h as f32,
            size_px: size_px.max(1.0),
            rotation_rad,
        }
    }

    /// Build a rotated ROI from a horizontal reference (e.g. eye line)
    /// that should align with the crop's `+x` axis. Used by the face
    /// mesh pipeline: align the right-eye → left-eye vector with crop
    /// "+x" so the model receives an upright, eye-level face.
    pub fn from_horizontal_axis(
        centre_nx: f32,
        centre_ny: f32,
        from_nx: f32,
        from_ny: f32,
        to_nx: f32,
        to_ny: f32,
        size_px: f32,
        frame_w: u32,
        frame_h: u32,
    ) -> Self {
        let dx = to_nx - from_nx;
        let dy = to_ny - from_ny;
        // For the horizontal reference, the from→to angle equals the
        // crop's `+x` axis image angle directly (R = theta).
        let rotation_rad = dy.atan2(dx);
        Self {
            cx_px: centre_nx * frame_w as f32,
            cy_px: centre_ny * frame_h as f32,
            size_px: size_px.max(1.0),
            rotation_rad,
        }
    }

    /// Axis-aligned fallback: build a rotated ROI with zero rotation from
    /// a normalised bbox. Useful when keypoints aren't available.
    pub fn from_axis_aligned_bbox(
        bbox: &[f32; 4],
        frame_w: u32,
        frame_h: u32,
    ) -> Self {
        let cx_n = (bbox[0] + bbox[2]) * 0.5;
        let cy_n = (bbox[1] + bbox[3]) * 0.5;
        let w_px = (bbox[2] - bbox[0]) * frame_w as f32;
        let h_px = (bbox[3] - bbox[1]) * frame_h as f32;
        Self {
            cx_px: cx_n * frame_w as f32,
            cy_px: cy_n * frame_h as f32,
            size_px: w_px.max(h_px).max(1.0),
            rotation_rad: 0.0,
        }
    }

    /// Map a model-output landmark (in normalised crop coordinates, where
    /// (0, 0) is the crop's top-left and (1, 1) is the bottom-right) back
    /// to image-normalised coordinates ([0, 1] of `frame_w` × `frame_h`).
    ///
    /// This is the inverse of the sampling done in
    /// [`preprocess_rotated_nhwc`] / [`preprocess_rotated_nchw_imagenet`]:
    /// it translates the landmark into crop-local pixel offsets, rotates
    /// by `rotation_rad`, adds the crop centre, then normalises by the
    /// frame size. Z is scaled by `size_px` so the depth coordinate is in
    /// the same pixel unit as x/y.
    pub fn landmark_to_image(
        &self,
        nx: f32,
        ny: f32,
        nz: f32,
        frame_w: u32,
        frame_h: u32,
    ) -> (f32, f32, f32) {
        let offset_u = (nx - 0.5) * self.size_px;
        let offset_v = (ny - 0.5) * self.size_px;
        let cos = self.rotation_rad.cos();
        let sin = self.rotation_rad.sin();
        let rot_u = offset_u * cos - offset_v * sin;
        let rot_v = offset_u * sin + offset_v * cos;
        let px = self.cx_px + rot_u;
        let py = self.cy_px + rot_v;
        let img_x = px / frame_w as f32;
        let img_y = py / frame_h as f32;
        // Z is reported by the model in the same normalised units as x/y
        // (relative to the crop), so scale it to source pixels.
        let img_z = nz * self.size_px / frame_w as f32;
        (img_x, img_y, img_z)
    }
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

/// Sample a rotated square crop from the source frame into an NHWC tensor
/// with values in `[0, 1]` (`ZeroOne`) or `[-1, 1]` (`NegOneOne`).
///
/// For each output pixel `(du, dv)` we compute its location in the
/// **rotated** source frame:
///
/// 1. Translate into crop-local pixel space, centred at zero.
/// 2. Rotate by `roi.rotation_rad` to get the source-frame pixel offset.
/// 3. Add `roi.cx_px, roi.cy_px` to get the absolute source pixel.
/// 4. Bilinear sample, normalise, and write into the tensor.
///
/// Pixels that land outside the source frame are zero-filled (black). This
/// matches MediaPipe's hand pipeline: the model is robust to a small amount
/// of padding around the crop edges.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::indexing_slicing)]
pub fn preprocess_rotated_nhwc(
    frame: &Frame,
    roi: &RotatedRoi,
    target_size: u32,
    range: NormalizeRange,
) -> Array4<f32> {
    let ts = target_size as usize;
    let mut tensor = Array4::<f32>::zeros((1, ts, ts, 3));

    let pad_value = match range {
        NormalizeRange::ZeroOne => 0.0,
        NormalizeRange::NegOneOne => -1.0,
    };
    if matches!(range, NormalizeRange::NegOneOne) {
        tensor.fill(pad_value);
    }

    let cos = roi.rotation_rad.cos();
    let sin = roi.rotation_rad.sin();
    let step = roi.size_px / target_size as f32;
    let half = (target_size as f32 - 1.0) * 0.5;

    let fw = frame.width as i32;
    let fh = frame.height as i32;

    for dy in 0..ts {
        for dx in 0..ts {
            // Crop-local offset in source-pixel units, centred at zero.
            let u = (dx as f32 - half) * step;
            let v = (dy as f32 - half) * step;
            // Rotate into source frame.
            let sx = roi.cx_px + u * cos - v * sin;
            let sy = roi.cy_px + u * sin + v * cos;

            let (r, g, b) = bilinear_sample(&frame.data, frame.width, frame.height, sx, sy);

            let (r, g, b) = match range {
                NormalizeRange::ZeroOne => (r / 255.0, g / 255.0, b / 255.0),
                NormalizeRange::NegOneOne => (
                    r / 127.5 - 1.0,
                    g / 127.5 - 1.0,
                    b / 127.5 - 1.0,
                ),
            };

            tensor[[0, dy, dx, 0]] = r;
            tensor[[0, dy, dx, 1]] = g;
            tensor[[0, dy, dx, 2]] = b;
        }
    }

    // Bounds reference so dead-code lint doesn't complain when this fn is
    // used by only one of the two landmark backends.
    let _ = (fw, fh);
    tensor
}

/// Same sampling geometry as [`preprocess_rotated_nhwc`] but emits the
/// `[1, 3, H, W]` ImageNet-normalised layout that RTMPose expects.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::indexing_slicing)]
pub fn preprocess_rotated_nchw_imagenet(
    frame: &Frame,
    roi: &RotatedRoi,
    target_size: u32,
    mean: [f32; 3],
    std: [f32; 3],
) -> Array4<f32> {
    let ts = target_size as usize;
    let mut tensor = Array4::<f32>::zeros((1, 3, ts, ts));

    let cos = roi.rotation_rad.cos();
    let sin = roi.rotation_rad.sin();
    let step = roi.size_px / target_size as f32;
    let half = (target_size as f32 - 1.0) * 0.5;

    for dy in 0..ts {
        for dx in 0..ts {
            let u = (dx as f32 - half) * step;
            let v = (dy as f32 - half) * step;
            let sx = roi.cx_px + u * cos - v * sin;
            let sy = roi.cy_px + u * sin + v * cos;

            let (r, g, b) = bilinear_sample(&frame.data, frame.width, frame.height, sx, sy);
            let channels = [r, g, b];
            for c in 0..3 {
                tensor[[0, c, dy, dx]] = (channels[c] - mean[c]) / std[c];
            }
        }
    }

    tensor
}

/// Bilinear sampler over an RGB byte buffer. Returns 0 for samples that
/// fall outside the frame.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn bilinear_sample(data: &[u8], width: u32, height: u32, sx: f32, sy: f32) -> (f32, f32, f32) {
    if sx < 0.0 || sy < 0.0 || sx > (width - 1) as f32 || sy > (height - 1) as f32 {
        return (0.0, 0.0, 0.0);
    }
    let x0 = sx.floor() as u32;
    let y0 = sy.floor() as u32;
    let x1 = (x0 + 1).min(width.saturating_sub(1));
    let y1 = (y0 + 1).min(height.saturating_sub(1));
    let xf = sx - x0 as f32;
    let yf = sy - y0 as f32;

    let get = |px: u32, py: u32| -> (f32, f32, f32) {
        let idx = ((py * width + px) * 3) as usize;
        let r = data.get(idx).copied().map_or(0.0, f32::from);
        let g = data.get(idx + 1).copied().map_or(0.0, f32::from);
        let b = data.get(idx + 2).copied().map_or(0.0, f32::from);
        (r, g, b)
    };

    let (r00, g00, b00) = get(x0, y0);
    let (r10, g10, b10) = get(x1, y0);
    let (r01, g01, b01) = get(x0, y1);
    let (r11, g11, b11) = get(x1, y1);

    let blend = |a: f32, b: f32, c: f32, d: f32| -> f32 {
        let top = a * (1.0 - xf) + b * xf;
        let bot = c * (1.0 - xf) + d * xf;
        top * (1.0 - yf) + bot * yf
    };

    (
        blend(r00, r10, r01, r11),
        blend(g00, g10, g01, g11),
        blend(b00, b10, b01, b11),
    )
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
    fn rotated_roi_identity_rotation_matches_axis_aligned() {
        // With rotation = 0, the rotated sampler must produce the same
        // landmark mapping as a plain centred crop.
        let roi = RotatedRoi {
            cx_px: 320.0,
            cy_px: 240.0,
            size_px: 100.0,
            rotation_rad: 0.0,
        };
        let (x, y, _z) = roi.landmark_to_image(0.5, 0.5, 0.0, 640, 480);
        assert!((x - 0.5).abs() < 1e-6, "centre maps to centre: x={x}");
        assert!((y - 0.5).abs() < 1e-6, "centre maps to centre: y={y}");

        let (x_left, y_left, _) = roi.landmark_to_image(0.0, 0.5, 0.0, 640, 480);
        // Left edge of model crop -> centre_x - size/2 in pixels.
        let expected_x = (320.0 - 50.0) / 640.0;
        assert!(
            (x_left - expected_x).abs() < 1e-4,
            "left edge x={x_left}, expected {expected_x}"
        );
        assert!((y_left - 0.5).abs() < 1e-4);
    }

    #[test]
    fn rotated_roi_90deg_rotates_landmarks() {
        // rotation = π/2 means crop's local "up" (-v) points toward image
        // "+x" (right). So a landmark at the *top* of the crop should map
        // to a pixel to the *right* of the crop centre in the image.
        let roi = RotatedRoi {
            cx_px: 300.0,
            cy_px: 200.0,
            size_px: 100.0,
            rotation_rad: std::f32::consts::FRAC_PI_2,
        };
        // Landmark at top of crop: (0.5, 0.0) -> offset_u=0, offset_v=-50
        // After R=π/2: rot_u = 0*0 - (-50)*1 = 50, rot_v = 0*1 + (-50)*0 = 0
        // So image px = (350, 200).
        let (x, y, _) = roi.landmark_to_image(0.5, 0.0, 0.0, 640, 480);
        let expected_x = 350.0 / 640.0;
        let expected_y = 200.0 / 480.0;
        assert!(
            (x - expected_x).abs() < 1e-4,
            "top-of-crop x={x}, expected {expected_x}"
        );
        assert!(
            (y - expected_y).abs() < 1e-4,
            "top-of-crop y={y}, expected {expected_y}"
        );
    }

    #[test]
    fn rotated_roi_from_axis_aligns_up_with_direction() {
        // wrist below mid_mcp in image (fingers pointing up).
        let wrist = (0.5, 0.7);
        let mid_mcp = (0.5, 0.4);
        let roi = RotatedRoi::from_axis(
            0.5, 0.55, // centre
            wrist.0, wrist.1, mid_mcp.0, mid_mcp.1,
            120.0, 640, 480,
        );
        // For a vertical "fingers up" hand, rotation should be ~0
        // (no rotation needed — already upright).
        assert!(
            roi.rotation_rad.abs() < 1e-4,
            "vertical hand should not rotate, got {}",
            roi.rotation_rad
        );

        // Hand pointing to the right: wrist at left, mid_mcp at right.
        let wrist2 = (0.2, 0.5);
        let mid_mcp2 = (0.6, 0.5);
        let roi2 = RotatedRoi::from_axis(
            0.4, 0.5, wrist2.0, wrist2.1, mid_mcp2.0, mid_mcp2.1,
            120.0, 640, 480,
        );
        // dx=+0.4, dy=0 → theta=0 → R=π/2.
        let expected = std::f32::consts::FRAC_PI_2;
        assert!(
            (roi2.rotation_rad - expected).abs() < 1e-4,
            "horizontal hand R={}, expected {expected}",
            roi2.rotation_rad
        );

        // Verify: landmark at the top of the model crop (where fingers
        // sit) for the horizontal hand should map to a pixel near the
        // mid_mcp side of the source.
        let (x_top, _, _) = roi2.landmark_to_image(0.5, 0.0, 0.0, 640, 480);
        // Crop is centred at 0.4 (image-normalised x); top of crop maps
        // to image direction +x = right of centre, so x > 0.4.
        assert!(x_top > 0.4, "fingers should map to the right of centre");
    }

    #[test]
    fn rotated_preprocessing_with_zero_rotation_matches_centre_crop() {
        // Build a 64x64 frame where each pixel encodes its column in red.
        // Sampling a centred 32x32 crop with zero rotation should read out
        // a horizontal gradient that ranges roughly over the central 32
        // columns.
        let w = 64_u32;
        let h = 64_u32;
        let mut data = vec![0_u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) * 3) as usize;
                data[idx] = (x * 4) as u8;
            }
        }
        let frame = Frame::new(data, w, h);
        let roi = RotatedRoi {
            cx_px: 32.0,
            cy_px: 32.0,
            size_px: 32.0,
            rotation_rad: 0.0,
        };
        let tensor = preprocess_rotated_nhwc(&frame, &roi, 32, NormalizeRange::ZeroOne);
        // Top row: red channel should increase monotonically across the row.
        let mut prev = -1.0_f32;
        for dx in 0..32 {
            let v = tensor[[0, 0, dx, 0]];
            assert!(
                v >= prev - 1e-3,
                "red channel must be monotonic across x (dx={dx} v={v} prev={prev})"
            );
            prev = v;
        }
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
