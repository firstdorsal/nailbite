/// Palm detection model wrapper.
///
/// Input: [1, 192, 192, 3] float32, normalized [0, 1]
/// Output: Bounding boxes with confidence scores + 7 palm keypoints
use std::sync::Arc;

use ort::inputs;
use ort::value::TensorRef;
use tracing::debug;

use crate::frame::Frame;
use crate::errors::InferenceError;
use crate::inference::postprocessing::{
    bbox_to_corners, decode_detections, expand_bbox, generate_ssd_anchors, non_max_suppression,
    remap_detection, Detection,
};
use crate::inference::preprocessing::{preprocess_frame_letterbox, NormalizeRange};
use crate::inference::session::ModelSession;

const INPUT_SIZE: u32 = 192;
/// Palm detection confidence threshold. Lower threshold catches more hands,
/// false positives filtered by hand landmark model.
const SCORE_THRESHOLD: f32 = 0.2;
/// Increased to allow more overlapping detections - tracker handles duplicates.
const NMS_THRESHOLD: f32 = 0.5;
const NUM_KEYPOINTS: usize = 7;
/// Expand palm bbox to capture full hand region.
const HAND_ROI_SCALE: f32 = 2.6;

pub struct PalmDetector {
    session: Arc<ModelSession>,
    anchors: Vec<[f32; 2]>,
}

/// Result of palm detection: bounding boxes for hand regions.
#[derive(Debug, Clone)]
pub struct PalmDetectionResult {
    /// Expanded bounding boxes in corner format [x_min, y_min, x_max, y_max], normalized.
    pub hand_rois: Vec<[f32; 4]>,
    /// Raw detections with scores and keypoints.
    pub detections: Vec<Detection>,
}

impl PalmDetector {
    pub fn new(session: Arc<ModelSession>) -> Self {
        // Palm detection uses strides [8, 16, 16, 16] with 2 anchors per location.
        let anchors = generate_ssd_anchors(INPUT_SIZE, &[8, 16, 16, 16], 2);
        Self { session, anchors }
    }

    /// Run palm detection on a full frame.
    pub fn detect(&self, frame: &Frame) -> Result<PalmDetectionResult, InferenceError> {
        let (tensor, letterbox) =
            preprocess_frame_letterbox(frame, INPUT_SIZE, NormalizeRange::ZeroOne);

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

        let hand_rois: Vec<[f32; 4]> = detections
            .iter()
            .map(|d| {
                let corners = bbox_to_corners(&d.bbox);
                let roi = expand_bbox(&corners, HAND_ROI_SCALE);
                debug!(
                    score = d.score,
                    bbox_cx = d.bbox[0],
                    bbox_cy = d.bbox[1],
                    bbox_w = d.bbox[2],
                    bbox_h = d.bbox[3],
                    roi_x0 = roi[0],
                    roi_y0 = roi[1],
                    roi_x1 = roi[2],
                    roi_y1 = roi[3],
                    "Palm detection ROI"
                );
                roi
            })
            .collect();

        debug!(count = hand_rois.len(), "Palm detections");

        Ok(PalmDetectionResult {
            hand_rois,
            detections,
        })
    }
}
