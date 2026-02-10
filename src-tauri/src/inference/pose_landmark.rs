/// Pose landmark (BlazePose full) model wrapper.
///
/// Input: [1, 256, 256, 3] float32, normalized [0, 1]
/// Output: [1, 195] = 33 landmarks x 5 (x, y, z, visibility, presence)
///         [1, 1] = pose presence flag
use std::sync::Arc;

use ort::inputs;
use ort::value::TensorRef;
use tracing::{debug, trace};

use crate::detection::types::{Landmark, PoseDetection, PoseLandmark};
use crate::errors::InferenceError;
use crate::frame::Frame;
use crate::inference::preprocessing::{preprocess_roi, NormalizeRange};
use crate::inference::session::ModelSession;

const INPUT_SIZE: u32 = 256;
/// Lowered to catch more poses - individual landmark visibility filters low-confidence parts.
const CONFIDENCE_THRESHOLD: f32 = 0.3;
/// Number of body landmarks in BlazePose model.
/// Note: The Unity model outputs 195 values = 39 landmarks, but we only use the first 33.
pub const NUM_POSE_LANDMARKS: usize = 33;

/// BlazePose landmark indices.
pub mod landmark_index {
    pub const NOSE: usize = 0;
    pub const LEFT_EYE_INNER: usize = 1;
    pub const LEFT_EYE: usize = 2;
    pub const LEFT_EYE_OUTER: usize = 3;
    pub const RIGHT_EYE_INNER: usize = 4;
    pub const RIGHT_EYE: usize = 5;
    pub const RIGHT_EYE_OUTER: usize = 6;
    pub const LEFT_EAR: usize = 7;
    pub const RIGHT_EAR: usize = 8;
    pub const MOUTH_LEFT: usize = 9;
    pub const MOUTH_RIGHT: usize = 10;
    pub const LEFT_SHOULDER: usize = 11;
    pub const RIGHT_SHOULDER: usize = 12;
    pub const LEFT_ELBOW: usize = 13;
    pub const RIGHT_ELBOW: usize = 14;
    pub const LEFT_WRIST: usize = 15;
    pub const RIGHT_WRIST: usize = 16;
    pub const LEFT_PINKY: usize = 17;
    pub const RIGHT_PINKY: usize = 18;
    pub const LEFT_INDEX: usize = 19;
    pub const RIGHT_INDEX: usize = 20;
    pub const LEFT_THUMB: usize = 21;
    pub const RIGHT_THUMB: usize = 22;
    pub const LEFT_HIP: usize = 23;
    pub const RIGHT_HIP: usize = 24;
    pub const LEFT_KNEE: usize = 25;
    pub const RIGHT_KNEE: usize = 26;
    pub const LEFT_ANKLE: usize = 27;
    pub const RIGHT_ANKLE: usize = 28;
    pub const LEFT_HEEL: usize = 29;
    pub const RIGHT_HEEL: usize = 30;
    pub const LEFT_FOOT_INDEX: usize = 31;
    pub const RIGHT_FOOT_INDEX: usize = 32;
}

pub struct PoseLandmarker {
    session: Arc<ModelSession>,
}

impl PoseLandmarker {
    pub fn new(session: Arc<ModelSession>) -> Self {
        Self { session }
    }

    /// Run pose landmark estimation on a detected person ROI.
    ///
    /// `roi` is [x_min, y_min, x_max, y_max] in normalized coordinates.
    /// Returns landmarks mapped back to full-frame coordinates.
    pub fn estimate(
        &self,
        frame: &Frame,
        roi: &[f32; 4],
    ) -> Result<Option<PoseDetection>, InferenceError> {
        let (tensor, square_roi) = preprocess_roi(
            frame,
            roi[0],
            roi[1],
            roi[2],
            roi[3],
            INPUT_SIZE,
            NormalizeRange::ZeroOne,
        );

        let input =
            TensorRef::from_array_view(&tensor).map_err(|e| InferenceError::Ort(e.to_string()))?;

        #[allow(clippy::indexing_slicing)]
        let (landmarks_raw, presence_flag) = self.session.run_inference(|session| {
            let outputs = session
                .run(inputs![input])
                .map_err(|e| InferenceError::Ort(e.to_string()))?;

            debug!(
                num_outputs = outputs.len(),
                "Pose model outputs"
            );

            // Output 0: landmarks [1, 195] -> 33 x (x, y, z, visibility, presence)
            // Output 1: presence flag [1, 1]
            let lm: Vec<f32> = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| InferenceError::Ort(e.to_string()))?
                .iter()
                .copied()
                .collect();

            let flag: Vec<f32> = outputs[1]
                .try_extract_array::<f32>()
                .map_err(|e| InferenceError::Ort(e.to_string()))?
                .iter()
                .copied()
                .collect();

            // Log first landmark values for debugging
            if lm.len() >= 5 {
                trace!(
                    lm0_x = lm[0],
                    lm0_y = lm[1],
                    lm0_z = lm[2],
                    lm0_vis = lm[3],
                    lm0_pres = lm[4],
                    flag = ?flag,
                    "First pose landmark raw values"
                );
            }

            Ok((lm, flag))
        })?;

        // Apply sigmoid to presence flag
        let presence = presence_flag.first().copied().unwrap_or(0.0);
        let presence_prob = 1.0 / (1.0 + (-presence).exp());

        debug!(
            raw_presence = presence,
            presence_prob = presence_prob,
            threshold = CONFIDENCE_THRESHOLD,
            landmarks_len = landmarks_raw.len(),
            "Pose inference result"
        );

        if presence_prob < CONFIDENCE_THRESHOLD {
            debug!(presence = presence_prob, "Pose presence below threshold");
            return Ok(None);
        }

        // Convert landmarks from model output
        let roi_w = square_roi.x_max - square_roi.x_min;
        let roi_h = square_roi.y_max - square_roi.y_min;

        let mut landmarks = Vec::with_capacity(NUM_POSE_LANDMARKS);

        // Check if landmarks are in pixel space [0, 256] or normalized [0, 1]
        // by looking at the first landmark's raw values
        let first_x = landmarks_raw.first().copied().unwrap_or(0.0);
        let is_pixel_space = first_x > 2.0; // If > 2, likely pixel coordinates

        for i in 0..NUM_POSE_LANDMARKS {
            let base = i * 5;
            let raw_x = landmarks_raw.get(base).copied().unwrap_or(0.0);
            let raw_y = landmarks_raw.get(base + 1).copied().unwrap_or(0.0);
            let raw_z = landmarks_raw.get(base + 2).copied().unwrap_or(0.0);
            let raw_visibility = landmarks_raw.get(base + 3).copied().unwrap_or(0.0);
            let raw_presence = landmarks_raw.get(base + 4).copied().unwrap_or(0.0);

            // Normalize to [0, 1] if in pixel space, otherwise use directly
            let (lx, ly, lz) = if is_pixel_space {
                (
                    raw_x / INPUT_SIZE as f32,
                    raw_y / INPUT_SIZE as f32,
                    raw_z / INPUT_SIZE as f32,
                )
            } else {
                // Already normalized [0, 1]
                (raw_x, raw_y, raw_z)
            };

            // Apply sigmoid to visibility and presence
            let visibility = 1.0 / (1.0 + (-raw_visibility).exp());
            let presence = 1.0 / (1.0 + (-raw_presence).exp());

            landmarks.push(PoseLandmark {
                landmark: Landmark {
                    x: square_roi.x_min + lx * roi_w,
                    y: square_roi.y_min + ly * roi_h,
                    z: lz * roi_w,
                },
                visibility,
                presence,
            });
        }

        // Debug: log first landmark position for troubleshooting
        if let Some(first_lm) = landmarks.first() {
            debug!(
                raw_x = first_x,
                is_pixel_space = is_pixel_space,
                mapped_x = first_lm.landmark.x,
                mapped_y = first_lm.landmark.y,
                "First pose landmark"
            );
        }

        debug!(
            presence = presence_prob,
            landmarks = landmarks.len(),
            "Pose landmarks detected"
        );

        Ok(Some(PoseDetection {
            landmarks,
            confidence: presence_prob,
        }))
    }
}
