//! Hand landmark model wrapper for RTMPose-m (SimCC head).
//!
//! Higher-quality alternative to the MediaPipe lite hand landmark model.
//! Trained on five hand datasets (Hand5: COCO-WholeBody-Hand,
//! OneHand10K, FreiHand, RHD, HALPE) so it generalizes much better to
//! unusual poses, partial occlusion, and varied skin tones — the kinds
//! of images the BFRB pipeline regularly sees when a hand approaches the
//! face or another hand.
//!
//! Model details (verified against the downloaded weights):
//!   * Input: `[N, 3, 256, 256]`, NCHW, ImageNet normalization.
//!   * Output: `simcc_x` and `simcc_y`, both `[N, K, 512]`. With a square
//!     256x256 input and split ratio 2.0, each axis has 512 SimCC bins.
//!   * `K = 21` keypoints (same topology as MediaPipe Hand).
//!
//! Decoding: per keypoint, take argmax along the SimCC bin axis for both
//! `simcc_x` and `simcc_y`, then divide by the split ratio (2.0) to get
//! pixel coordinates in the 256x256 input space, then normalize by 256.0
//! and remap into the original frame using the square-padded ROI.

use std::sync::Arc;

use ndarray::Array4;
use ort::inputs;
use ort::value::TensorRef;
use tracing::debug;

use crate::detection::types::{HandDetection, HandSide, Landmark};
use crate::errors::InferenceError;
use crate::frame::Frame;
use crate::inference::preprocessing::{preprocess_roi, NormalizeRange, SquareRoi};
use crate::inference::session::ModelSession;

const INPUT_SIZE: u32 = 256;
const NUM_KEYPOINTS: usize = 21;
const SIMCC_SPLIT_RATIO: f32 = 2.0;

/// ImageNet normalization in BGR-mean form used by mmpose. Values are
/// applied to RGB channels (mmpose stores them in this order in the
/// `Normalize` config; we match channel order to RGB at preprocess time).
const IMAGENET_MEAN: [f32; 3] = [123.675, 116.28, 103.53];
const IMAGENET_STD: [f32; 3] = [58.395, 57.12, 57.375];

/// Per-keypoint confidence threshold below which the entire detection
/// is rejected. Mirrors the lite model's permissive threshold so the
/// downstream tracker (`detection::tracker`) keeps full control over
/// hysteresis decisions.
const CONFIDENCE_THRESHOLD: f32 = 0.10;

pub struct RtmPoseHandLandmarker {
    session: Arc<ModelSession>,
}

impl RtmPoseHandLandmarker {
    pub fn new(session: Arc<ModelSession>) -> Self {
        Self { session }
    }

    /// Run hand landmark estimation on a cropped hand ROI.
    ///
    /// `roi` is `[x_min, y_min, x_max, y_max]` in normalized coordinates.
    /// Returns landmarks mapped back to full-frame normalized coordinates,
    /// matching the output convention of [`crate::inference::hand_landmark::HandLandmarker`]
    /// so callers can swap the two implementations.
    pub fn estimate(
        &self,
        frame: &Frame,
        roi: &[f32; 4],
    ) -> Result<Option<HandDetection>, InferenceError> {
        // RTMPose expects NCHW with ImageNet normalization. preprocess_roi
        // produces NHWC ZeroOne, so we run our own NCHW pass below — but
        // we still want the square-ROI side effect (expanding the shorter
        // dimension to make the crop square in pixel space). To get that,
        // we call preprocess_roi for the SquareRoi only and then redo
        // preprocessing in NCHW with the right normalization.
        let (_discard, square_roi) = preprocess_roi(
            frame,
            roi[0],
            roi[1],
            roi[2],
            roi[3],
            INPUT_SIZE,
            NormalizeRange::ZeroOne,
        );

        let tensor = preprocess_square_nchw(frame, &square_roi, INPUT_SIZE);

        let input = TensorRef::from_array_view(&tensor)
            .map_err(|e| InferenceError::Ort(e.to_string()))?;

        #[allow(clippy::indexing_slicing)] // SessionOutputs only supports Index
        let (simcc_x, simcc_y, k_dim, x_bins, y_bins) =
            self.session.run_inference(|session| {
                let outputs = session
                    .run(inputs![input])
                    .map_err(|e| InferenceError::Ort(e.to_string()))?;

                let x_view = outputs[0]
                    .try_extract_array::<f32>()
                    .map_err(|e| InferenceError::Ort(e.to_string()))?;
                let y_view = outputs[1]
                    .try_extract_array::<f32>()
                    .map_err(|e| InferenceError::Ort(e.to_string()))?;

                // Expected shapes: [1, K, X_BINS] and [1, K, Y_BINS].
                let x_shape = x_view.shape().to_vec();
                let y_shape = y_view.shape().to_vec();
                if x_shape.len() != 3 || y_shape.len() != 3 {
                    return Err(InferenceError::Ort(format!(
                        "Unexpected SimCC shape: x={x_shape:?} y={y_shape:?}"
                    )));
                }
                let k_dim = x_shape[1];
                let x_bins = x_shape[2];
                let y_bins = y_shape[2];
                if k_dim != y_shape[1] || k_dim == 0 {
                    return Err(InferenceError::Ort(format!(
                        "SimCC keypoint dim mismatch: x={x_shape:?} y={y_shape:?}"
                    )));
                }

                let x_vec: Vec<f32> = x_view.iter().copied().collect();
                let y_vec: Vec<f32> = y_view.iter().copied().collect();
                Ok((x_vec, y_vec, k_dim, x_bins, y_bins))
            })?;

        // Decode each keypoint via per-axis argmax. Confidence is the
        // mean over keypoints of `0.5 * (peak_x + peak_y)`, matching
        // mmpose's `get_simcc_maximum` decoder. The exported model
        // softmax-normalizes its SimCC outputs so peaks live in [0, 1]:
        // ~0.10-0.25 for empty/noise crops, ~0.4-0.8 for confident hands.
        let kp_count = k_dim.min(NUM_KEYPOINTS);
        let mut landmarks = [Landmark { x: 0.0, y: 0.0, z: 0.0 }; NUM_KEYPOINTS];
        let mut peak_sum = 0.0_f32;
        let mut peak_count = 0_u32;

        let roi_w = square_roi.x_max - square_roi.x_min;
        let roi_h = square_roi.y_max - square_roi.y_min;

        for (k, slot) in landmarks.iter_mut().enumerate().take(kp_count) {
            let x_slice_start = k * x_bins;
            let y_slice_start = k * y_bins;
            let x_slice = simcc_x
                .get(x_slice_start..x_slice_start + x_bins)
                .ok_or_else(|| InferenceError::Ort("simcc_x slice OOB".to_string()))?;
            let y_slice = simcc_y
                .get(y_slice_start..y_slice_start + y_bins)
                .ok_or_else(|| InferenceError::Ort("simcc_y slice OOB".to_string()))?;

            let (x_bin, x_peak) = argmax(x_slice);
            let (y_bin, y_peak) = argmax(y_slice);

            // Pixel coordinates in INPUT_SIZE-space.
            let px = x_bin as f32 / SIMCC_SPLIT_RATIO;
            let py = y_bin as f32 / SIMCC_SPLIT_RATIO;
            // Normalize within the square crop, then map to original frame.
            let nx = px / INPUT_SIZE as f32;
            let ny = py / INPUT_SIZE as f32;

            *slot = Landmark {
                x: square_roi.x_min + nx * roi_w,
                y: square_roi.y_min + ny * roi_h,
                // Z is not produced by SimCC; emit 0 to stay
                // schema-compatible with the MediaPipe path.
                z: 0.0,
            };

            peak_sum += 0.5 * (x_peak + y_peak);
            peak_count += 1;
        }

        let confidence = if peak_count == 0 {
            0.0
        } else {
            // Already in [0, 1] thanks to the softmax in the SimCC head.
            // Clamping just guards against numerical overshoot.
            (peak_sum / peak_count as f32).clamp(0.0, 1.0)
        };

        if confidence < CONFIDENCE_THRESHOLD {
            debug!(confidence, "RTMPose hand confidence below threshold");
            return Ok(None);
        }

        // RTMPose-m hand5 doesn't predict handedness (it was trained on
        // the COCO-WholeBody-Hand topology which is side-agnostic). We
        // leave handedness to the tracker, which already infers it from
        // pose geometry / wrist-shoulder spatial heuristics.
        let side: Option<HandSide> = None;

        debug!(confidence, kp = kp_count, "RTMPose hand landmarks decoded");

        Ok(Some(HandDetection {
            side,
            landmarks,
            confidence,
        }))
    }
}

/// Preprocess a square ROI into RTMPose's expected `[1, 3, H, W]` tensor.
///
/// The crop is taken in pixel space using the already-square `SquareRoi`
/// returned by `preprocess_roi`, then bilinearly resized to `INPUT_SIZE`
/// and normalized with ImageNet statistics in NCHW layout.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::indexing_slicing)]
fn preprocess_square_nchw(frame: &Frame, roi: &SquareRoi, input_size: u32) -> Array4<f32> {
    let fw = frame.width as f32;
    let fh = frame.height as f32;

    // Pixel-space crop bounds.
    let px_x0 = (roi.x_min * fw).clamp(0.0, fw) as u32;
    let px_y0 = (roi.y_min * fh).clamp(0.0, fh) as u32;
    let px_x1 = (roi.x_max * fw).clamp(0.0, fw) as u32;
    let px_y1 = (roi.y_max * fh).clamp(0.0, fh) as u32;

    let crop_w = px_x1.saturating_sub(px_x0).max(1);
    let crop_h = px_y1.saturating_sub(px_y0).max(1);

    // Bilinear resize directly from the source frame into INPUT_SIZE x INPUT_SIZE.
    let isz = input_size as usize;
    let mut tensor = Array4::<f32>::zeros((1, 3, isz, isz));

    let x_ratio = crop_w as f32 / input_size as f32;
    let y_ratio = crop_h as f32 / input_size as f32;

    for dy in 0..isz {
        for dx in 0..isz {
            let sx = (dx as f32 + 0.5) * x_ratio - 0.5;
            let sy = (dy as f32 + 0.5) * y_ratio - 0.5;
            let sx = sx.clamp(0.0, (crop_w - 1) as f32);
            let sy = sy.clamp(0.0, (crop_h - 1) as f32);

            let x0 = sx as u32;
            let y0 = sy as u32;
            let x1 = (x0 + 1).min(crop_w - 1);
            let y1 = (y0 + 1).min(crop_h - 1);
            let xf = sx - x0 as f32;
            let yf = sy - y0 as f32;

            let mut samples = [0.0_f32; 3];
            for (c, out) in samples.iter_mut().enumerate() {
                let s00 = sample(frame, px_x0 + x0, px_y0 + y0, c);
                let s10 = sample(frame, px_x0 + x1, px_y0 + y0, c);
                let s01 = sample(frame, px_x0 + x0, px_y0 + y1, c);
                let s11 = sample(frame, px_x0 + x1, px_y0 + y1, c);
                let top = s00 * (1.0 - xf) + s10 * xf;
                let bot = s01 * (1.0 - xf) + s11 * xf;
                *out = top * (1.0 - yf) + bot * yf;
            }

            for c in 0..3 {
                let normalized = (samples[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
                tensor[[0, c, dy, dx]] = normalized;
            }
        }
    }

    tensor
}

#[inline]
fn sample(frame: &Frame, x: u32, y: u32, c: usize) -> f32 {
    let x = x.min(frame.width.saturating_sub(1));
    let y = y.min(frame.height.saturating_sub(1));
    let idx = ((y * frame.width + x) * 3) as usize + c;
    frame.data.get(idx).copied().map_or(0.0, f32::from)
}

#[inline]
fn argmax(slice: &[f32]) -> (usize, f32) {
    let mut best_idx = 0_usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in slice.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    (best_idx, best_val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_highest() {
        let (idx, val) = argmax(&[0.1, -2.0, 5.5, 5.4]);
        assert_eq!(idx, 2);
        assert!((val - 5.5).abs() < 1e-6);
    }

    #[test]
    fn argmax_handles_empty_via_negative_infinity() {
        let (idx, val) = argmax(&[]);
        assert_eq!(idx, 0);
        assert!(val.is_infinite() && val.is_sign_negative());
    }

    #[test]
    fn preprocess_produces_nchw_shape() {
        let frame = Frame::new(vec![128_u8; 64 * 64 * 3], 64, 64);
        let roi = SquareRoi {
            x_min: 0.0,
            y_min: 0.0,
            x_max: 1.0,
            y_max: 1.0,
        };
        let tensor = preprocess_square_nchw(&frame, &roi, 256);
        assert_eq!(tensor.shape(), &[1, 3, 256, 256]);
        // Mid-grey input minus mean / std should be a constant per channel.
        for c in 0..3 {
            let expected = (128.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            let got = tensor[[0, c, 128, 128]];
            assert!(
                (got - expected).abs() < 1e-3,
                "channel {c}: expected {expected}, got {got}"
            );
        }
    }
}
