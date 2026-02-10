/// Hand landmark model wrapper.
///
/// Input: [1, 224, 224, 3] float32, normalized [0, 1]
/// Output: [1, 63] = 21 landmarks x 3 (x, y, z)
use std::sync::Arc;

use ort::inputs;
use ort::value::TensorRef;
use tracing::debug;

use crate::frame::Frame;
use crate::detection::types::{HandDetection, HandSide, Landmark};
use crate::errors::InferenceError;
use crate::inference::preprocessing::{preprocess_roi, NormalizeRange};
use crate::inference::session::ModelSession;

const INPUT_SIZE: u32 = 224;
/// Lowered to retain more hand detections - tracking handles false positives.
const CONFIDENCE_THRESHOLD: f32 = 0.15;

pub struct HandLandmarker {
    session: Arc<ModelSession>,
}

impl HandLandmarker {
    pub fn new(session: Arc<ModelSession>) -> Self {
        Self { session }
    }

    /// Run hand landmark estimation on a cropped hand ROI.
    ///
    /// `roi` is [x_min, y_min, x_max, y_max] in normalized coordinates.
    /// Returns landmarks mapped back to full-frame coordinates.
    pub fn estimate(
        &self,
        frame: &Frame,
        roi: &[f32; 4],
    ) -> Result<Option<HandDetection>, InferenceError> {
        let (tensor, square_roi) = preprocess_roi(
            frame,
            roi[0],
            roi[1],
            roi[2],
            roi[3],
            INPUT_SIZE,
            NormalizeRange::ZeroOne,
        );

        let input = TensorRef::from_array_view(&tensor)
            .map_err(|e| InferenceError::Ort(e.to_string()))?;

        #[allow(clippy::indexing_slicing)] // SessionOutputs only supports Index, not .get()
        let (landmarks_raw, confidence, handedness) = self.session.run_inference(|session| {
            let outputs = session
                .run(inputs![input])
                .map_err(|e| InferenceError::Ort(e.to_string()))?;

            // Output 0: landmarks [1, 63] -> 21 x (x, y, z)
            // Output 1: confidence [1, 1]
            // Output 2: handedness [1, 1]
            let lm: Vec<f32> = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| InferenceError::Ort(e.to_string()))?
                .iter()
                .copied()
                .collect();

            let conf: Vec<f32> = outputs[1]
                .try_extract_array::<f32>()
                .map_err(|e| InferenceError::Ort(e.to_string()))?
                .iter()
                .copied()
                .collect();

            let hand: Vec<f32> = outputs[2]
                .try_extract_array::<f32>()
                .map_err(|e| InferenceError::Ort(e.to_string()))?
                .iter()
                .copied()
                .collect();

            Ok((lm, conf, hand))
        })?;

        let conf = confidence.first().copied().unwrap_or(0.0);
        if conf < CONFIDENCE_THRESHOLD {
            debug!(confidence = conf, "Hand confidence below threshold");
            return Ok(None);
        }

        // Parse handedness (>0.5 = right, <0.5 = left).
        let side = if handedness.first().copied().unwrap_or(0.5) > 0.5 {
            Some(HandSide::Right)
        } else {
            Some(HandSide::Left)
        };

        // Convert 63 floats to 21 landmarks, mapping back to full-frame coords.
        // Use the square-adjusted ROI for remapping (preprocess_roi may have
        // expanded the shorter dimension to make the crop square in pixel space).
        let roi_w = square_roi.x_max - square_roi.x_min;
        let roi_h = square_roi.y_max - square_roi.y_min;
        let mut landmarks = [Landmark {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }; 21];

        for (i, lm) in landmarks.iter_mut().enumerate() {
            let base = i * 3;
            let Some(&raw_x) = landmarks_raw.get(base) else {
                break;
            };
            let Some(&raw_y) = landmarks_raw.get(base + 1) else {
                break;
            };
            let Some(&raw_z) = landmarks_raw.get(base + 2) else {
                break;
            };
            // Landmarks are relative to the crop (0..224), normalize to 0..1 then map to frame.
            let lx = raw_x / INPUT_SIZE as f32;
            let ly = raw_y / INPUT_SIZE as f32;
            let lz = raw_z / INPUT_SIZE as f32;

            *lm = Landmark {
                x: square_roi.x_min + lx * roi_w,
                y: square_roi.y_min + ly * roi_h,
                z: lz * roi_w, // Scale Z by ROI width.
            };
        }

        debug!(confidence = conf, ?side, "Hand landmarks detected");

        Ok(Some(HandDetection {
            side,
            landmarks,
            confidence: conf,
        }))
    }
}
