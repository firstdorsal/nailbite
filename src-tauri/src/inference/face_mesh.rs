/// Face mesh model wrapper.
///
/// Input: [1, 192, 192, 3] float32, normalized [0, 1]
/// Output: [1, 1404] = 468 landmarks x 3 (x, y, z)
use std::sync::Arc;

use ort::inputs;
use ort::value::TensorRef;
use tracing::debug;

use crate::frame::Frame;
use crate::detection::types::{FaceDetection, Landmark};
use crate::errors::InferenceError;
use crate::inference::postprocessing::sigmoid;
use crate::inference::preprocessing::{preprocess_rotated_nhwc, NormalizeRange, RotatedRoi};
use crate::inference::session::ModelSession;

const INPUT_SIZE: u32 = 192;
const CONFIDENCE_THRESHOLD: f32 = 0.5;
const NUM_LANDMARKS: usize = 468;

pub struct FaceMesher {
    session: Arc<ModelSession>,
}

impl FaceMesher {
    pub fn new(session: Arc<ModelSession>) -> Self {
        Self { session }
    }

    /// Run face mesh estimation on a rotation-normalised face crop.
    ///
    /// The face mesh model is trained on upright, eye-level face crops.
    /// Feeding it an axis-aligned bbox from face detection makes the
    /// 468 landmarks — especially the lip/jaw set — jitter on every
    /// frame-to-frame change in head tilt. Callers compute the rotation
    /// from BlazeFace's `right-eye → left-eye` keypoints and pass it in
    /// via `roi`; landmark outputs are inverse-rotated back into image
    /// coordinates.
    pub fn estimate(
        &self,
        frame: &Frame,
        roi: &RotatedRoi,
    ) -> Result<Option<FaceDetection>, InferenceError> {
        let tensor = preprocess_rotated_nhwc(
            frame,
            roi,
            INPUT_SIZE,
            NormalizeRange::ZeroOne,
        );

        let input = TensorRef::from_array_view(&tensor)
            .map_err(|e| InferenceError::Ort(e.to_string()))?;

        // Output 0: landmarks [1, 1404] -> 468 x (x, y, z)
        // Output 1: confidence [1, 1] -> logit, apply sigmoid
        #[allow(clippy::indexing_slicing)] // SessionOutputs only supports Index, not .get()
        let (landmarks_raw, confidence_raw) = self.session.run_inference(|session| {
            let outputs = session
                .run(inputs![input])
                .map_err(|e| InferenceError::Ort(e.to_string()))?;

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

            Ok((lm, conf))
        })?;

        let conf = sigmoid(confidence_raw.first().copied().unwrap_or(0.0));
        if conf < CONFIDENCE_THRESHOLD {
            debug!(confidence = conf, "Face mesh confidence below threshold");
            return Ok(None);
        }

        let mut landmarks = Vec::with_capacity(NUM_LANDMARKS);
        for i in 0..NUM_LANDMARKS {
            let base = i * 3;
            let raw_x = landmarks_raw.get(base).copied();
            let raw_y = landmarks_raw.get(base + 1).copied();
            let raw_z = landmarks_raw.get(base + 2).copied();

            let lm = match (raw_x, raw_y, raw_z) {
                (Some(rx), Some(ry), Some(rz)) => {
                    let nx = rx / INPUT_SIZE as f32;
                    let ny = ry / INPUT_SIZE as f32;
                    let nz = rz / INPUT_SIZE as f32;
                    let (img_x, img_y, img_z) =
                        roi.landmark_to_image(nx, ny, nz, frame.width, frame.height);
                    Landmark { x: img_x, y: img_y, z: img_z }
                }
                _ => Landmark {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
            };
            landmarks.push(lm);
        }

        debug!(confidence = conf, "Face mesh landmarks detected");

        Ok(Some(FaceDetection {
            landmarks,
            confidence: conf,
        }))
    }
}
