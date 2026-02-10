/// Pose detection (BlazePose detector) model wrapper.
///
/// Input: [1, 224, 224, 3] float32, normalized [0, 1]
/// Output: Bounding boxes + 4 body keypoints (mid-hip, chest, mid-shoulder, face)
use std::sync::Arc;

use ort::inputs;
use ort::value::TensorRef;
use tracing::debug;

use crate::errors::InferenceError;
use crate::frame::Frame;
use crate::inference::postprocessing::{
    bbox_to_corners, decode_detections, expand_bbox, generate_ssd_anchors, non_max_suppression,
    remap_detection, Detection,
};
use crate::inference::preprocessing::{preprocess_frame_letterbox, NormalizeRange};
use crate::inference::session::ModelSession;

const INPUT_SIZE: u32 = 224;
const SCORE_THRESHOLD: f32 = 0.5;
const NMS_THRESHOLD: f32 = 0.3;
/// 4 keypoints: mid-hip, full body mid-point, mid-shoulder, face center.
const NUM_KEYPOINTS: usize = 4;
/// Expand pose bbox for landmark model input.
const POSE_ROI_SCALE: f32 = 1.25;

pub struct PoseDetector {
    session: Arc<ModelSession>,
    anchors: Vec<[f32; 2]>,
}

/// Result of pose detection.
#[derive(Debug, Clone)]
pub struct PoseDetectionResult {
    /// Expanded bounding boxes for pose landmark input [x_min, y_min, x_max, y_max].
    pub pose_rois: Vec<[f32; 4]>,
    /// Raw detections with keypoints.
    pub detections: Vec<Detection>,
}

impl PoseDetector {
    pub fn new(session: Arc<ModelSession>) -> Self {
        // BlazePose detector uses similar anchor scheme to BlazeFace.
        // Strides [8, 16, 32, 32, 32] with 2 anchors per location.
        let anchors = generate_ssd_anchors(INPUT_SIZE, &[8, 16, 32, 32, 32], 2);
        Self { session, anchors }
    }

    pub fn detect(&self, frame: &Frame) -> Result<PoseDetectionResult, InferenceError> {
        let (tensor, letterbox) =
            preprocess_frame_letterbox(frame, INPUT_SIZE, NormalizeRange::ZeroOne);

        let input =
            TensorRef::from_array_view(&tensor).map_err(|e| InferenceError::Ort(e.to_string()))?;

        let anchors = &self.anchors;
        #[allow(clippy::indexing_slicing)]
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

        let pose_rois: Vec<[f32; 4]> = detections
            .iter()
            .map(|d| {
                let corners = bbox_to_corners(&d.bbox);
                expand_bbox(&corners, POSE_ROI_SCALE)
            })
            .collect();

        debug!(count = pose_rois.len(), "Pose detections");

        Ok(PoseDetectionResult {
            pose_rois,
            detections,
        })
    }
}
