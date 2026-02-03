/// Face detection (BlazeFace short range) model wrapper.
///
/// Input: [1, 128, 128, 3] float32, normalized [-1, 1]
/// Output: Bounding boxes + 6 facial keypoints
use std::sync::Arc;

use ort::inputs;
use ort::value::TensorRef;
use tracing::debug;

use crate::camera::frame::Frame;
use crate::errors::InferenceError;
use crate::inference::postprocessing::{
    bbox_to_corners, decode_detections, expand_bbox, generate_ssd_anchors, non_max_suppression,
    remap_detection, Detection,
};
use crate::inference::preprocessing::{preprocess_frame_letterbox, NormalizeRange};
use crate::inference::session::ModelSession;

const INPUT_SIZE: u32 = 128;
const SCORE_THRESHOLD: f32 = 0.5;
const NMS_THRESHOLD: f32 = 0.3;
const NUM_KEYPOINTS: usize = 6;
/// Expand face bbox for face mesh input.
const FACE_ROI_SCALE: f32 = 1.5;

pub struct FaceDetector {
    session: Arc<ModelSession>,
    anchors: Vec<[f32; 2]>,
}

/// Result of face detection.
#[derive(Debug, Clone)]
pub struct FaceDetectionResult {
    /// Expanded bounding boxes for face mesh input.
    pub face_rois: Vec<[f32; 4]>,
    /// Raw detections.
    pub detections: Vec<Detection>,
}

impl FaceDetector {
    pub fn new(session: Arc<ModelSession>) -> Self {
        // BlazeFace short range: strides [8, 16, 16, 16] with 2 anchors.
        let anchors = generate_ssd_anchors(INPUT_SIZE, &[8, 16, 16, 16], 2);
        Self { session, anchors }
    }

    pub fn detect(&self, frame: &Frame) -> Result<FaceDetectionResult, InferenceError> {
        let (tensor, letterbox) =
            preprocess_frame_letterbox(frame, INPUT_SIZE, NormalizeRange::NegOneOne);

        let input = TensorRef::from_array_view(&tensor)
            .map_err(|e| InferenceError::Ort(e.to_string()))?;

        let anchors = &self.anchors;
        #[allow(clippy::indexing_slicing)] // SessionOutputs only supports Index, not .get()
        let (raw_boxes, raw_scores) = self.session.run_inference(|session| {
            let outputs = session
                .run(inputs![input])
                .map_err(|e| InferenceError::Ort(e.to_string()))?;

            let boxes: Vec<f32> = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| InferenceError::Ort(e.to_string()))?
                .iter()
                .copied()
                .collect();

            let scores: Vec<f32> = outputs[1]
                .try_extract_array::<f32>()
                .map_err(|e| InferenceError::Ort(e.to_string()))?
                .iter()
                .copied()
                .collect();

            Ok((boxes, scores))
        })?;

        let mut detections = decode_detections(
            &raw_boxes,
            &raw_scores,
            anchors,
            INPUT_SIZE as f32,
            NUM_KEYPOINTS,
            SCORE_THRESHOLD,
        );

        non_max_suppression(&mut detections, NMS_THRESHOLD);

        // Remap detections from letterboxed model space to image space.
        for det in &mut detections {
            remap_detection(det, &letterbox);
        }

        let face_rois: Vec<[f32; 4]> = detections
            .iter()
            .map(|d| {
                let corners = bbox_to_corners(&d.bbox);
                expand_bbox(&corners, FACE_ROI_SCALE)
            })
            .collect();

        debug!(count = face_rois.len(), "Face detections");

        Ok(FaceDetectionResult {
            face_rois,
            detections,
        })
    }
}
