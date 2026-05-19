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

use ort::inputs;
use ort::value::TensorRef;
use tracing::debug;

use crate::detection::types::{HandDetection, HandSide, Landmark};
use crate::errors::InferenceError;
use crate::frame::Frame;
use crate::inference::preprocessing::{preprocess_rotated_nchw_imagenet, RotatedRoi};
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

    /// Run hand landmark estimation on a rotation-normalised crop.
    ///
    /// The crop is sampled along `roi`'s axes (so the hand fingers point
    /// up in the model's frame of reference) and landmarks are inverse-
    /// rotated back into image-normalised coordinates.
    pub fn estimate(
        &self,
        frame: &Frame,
        roi: &RotatedRoi,
    ) -> Result<Option<HandDetection>, InferenceError> {
        // RTMPose expects NCHW with ImageNet normalisation, sampled from
        // a rotation-normalised square crop.
        let tensor = preprocess_rotated_nchw_imagenet(
            frame,
            roi,
            INPUT_SIZE,
            IMAGENET_MEAN,
            IMAGENET_STD,
        );

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
            // Normalise within the square crop, then inverse-rotate back
            // into image-normalised coordinates.
            let nx = px / INPUT_SIZE as f32;
            let ny = py / INPUT_SIZE as f32;

            let (img_x, img_y, _img_z) =
                roi.landmark_to_image(nx, ny, 0.0, frame.width, frame.height);
            *slot = Landmark {
                x: img_x,
                y: img_y,
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
#[allow(clippy::indexing_slicing, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
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
    fn rotated_preprocess_produces_nchw_shape() {
        let frame = Frame::new(vec![128_u8; 64 * 64 * 3], 64, 64);
        let roi = RotatedRoi {
            cx_px: 32.0,
            cy_px: 32.0,
            size_px: 64.0,
            rotation_rad: 0.0,
        };
        let tensor = preprocess_rotated_nchw_imagenet(
            &frame, &roi, 256, IMAGENET_MEAN, IMAGENET_STD,
        );
        assert_eq!(tensor.shape(), &[1, 3, 256, 256]);
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
