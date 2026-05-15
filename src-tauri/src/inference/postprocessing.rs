//! Post-processing utilities for model outputs.
//!
//! Includes non-maximum suppression (NMS), anchor decoding,
//! and coordinate mapping back to original image space.

/// A bounding box with associated confidence and optional keypoints.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Center-x, center-y, width, height (all normalized 0..1 relative to model input).
    pub bbox: [f32; 4],
    pub score: f32,
    /// Keypoints as (x, y) pairs, normalized to model input size.
    pub keypoints: Vec<[f32; 2]>,
}

/// Generate SSD anchors for palm/face detection.
///
/// Anchor configuration follows the MediaPipe SSD anchor spec.
pub fn generate_ssd_anchors(
    input_size: u32,
    strides: &[u32],
    num_anchors_per_location: u32,
) -> Vec<[f32; 2]> {
    let mut anchors = Vec::new();

    for &stride in strides {
        let grid = input_size / stride;
        for y in 0..grid {
            for x in 0..grid {
                for _ in 0..num_anchors_per_location {
                    let cx = (x as f32 + 0.5) / grid as f32;
                    let cy = (y as f32 + 0.5) / grid as f32;
                    anchors.push([cx, cy]);
                }
            }
        }
    }

    anchors
}

/// Decode raw SSD regression outputs into detections using anchors.
///
/// `raw_boxes`: shape `[N, box_size]` where first 4 are center offsets + size.
/// `raw_scores`: shape `[N, 1]` raw logits.
/// `anchors`: `[N, 2]` anchor centers.
/// `num_keypoints`: number of (x, y) keypoint pairs after the 4 box values.
pub fn decode_detections(
    raw_boxes: &[f32],
    raw_scores: &[f32],
    anchors: &[[f32; 2]],
    input_size: f32,
    num_keypoints: usize,
    score_threshold: f32,
) -> Vec<Detection> {
    let box_size = 4 + num_keypoints * 2;
    let num_anchors = anchors.len();
    let mut detections = Vec::new();

    for i in 0..num_anchors {
        let score = sigmoid(raw_scores.get(i).copied().unwrap_or(0.0));
        if score < score_threshold {
            continue;
        }

        let offset = i * box_size;
        let Some(box_data) = raw_boxes.get(offset..offset + box_size) else {
            break;
        };

        let anchor = anchors.get(i).copied().unwrap_or([0.5, 0.5]);

        // Decode box center + size relative to anchor.
        let cx = box_data.first().copied().unwrap_or(0.0) / input_size + anchor[0];
        let cy = box_data.get(1).copied().unwrap_or(0.0) / input_size + anchor[1];
        let w = box_data.get(2).copied().unwrap_or(0.0) / input_size;
        let h = box_data.get(3).copied().unwrap_or(0.0) / input_size;

        // Decode keypoints.
        let mut keypoints = Vec::with_capacity(num_keypoints);
        for k in 0..num_keypoints {
            let kp_offset = 4 + k * 2;
            let kx = box_data.get(kp_offset).copied().unwrap_or(0.0) / input_size + anchor[0];
            let ky = box_data
                .get(kp_offset + 1)
                .copied()
                .unwrap_or(0.0)
                / input_size
                + anchor[1];
            keypoints.push([kx, ky]);
        }

        detections.push(Detection {
            bbox: [cx, cy, w, h],
            score,
            keypoints,
        });
    }

    detections
}

/// Non-maximum suppression on a list of detections.
///
/// Iterates detections by score (highest first), suppressing overlapping boxes.
/// Suppression triggers when EITHER the IoU exceeds `iou_threshold`, OR the
/// intersection covers more than `IOMIN_THRESHOLD` of the smaller box. The
/// second check catches the failure mode where palm detection fires twice
/// on the same hand with nested bboxes (a tight box inside a loose one) —
/// pure IoU is dominated by the larger box's area in that case and lets
/// the duplicate through.
const IOMIN_THRESHOLD: f32 = 0.6;
#[allow(clippy::indexing_slicing)]
pub fn non_max_suppression(detections: &mut Vec<Detection>, iou_threshold: f32) {
    detections.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let mut keep = vec![true; detections.len()];

    for i in 0..detections.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..detections.len() {
            if !keep[j] {
                continue;
            }
            let a = &detections[i].bbox;
            let b = &detections[j].bbox;
            if iou(a, b) > iou_threshold || iomin(a, b) > IOMIN_THRESHOLD {
                keep[j] = false;
            }
        }
    }

    let mut idx = 0;
    detections.retain(|_| {
        let k = keep.get(idx).copied().unwrap_or(false);
        idx += 1;
        k
    });
}

/// Intersection over the smaller box's area. Robust against nested
/// bounding boxes that pure IoU under-penalizes.
fn iomin(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let a_x1 = a[0] - a[2] / 2.0;
    let a_y1 = a[1] - a[3] / 2.0;
    let a_x2 = a[0] + a[2] / 2.0;
    let a_y2 = a[1] + a[3] / 2.0;

    let b_x1 = b[0] - b[2] / 2.0;
    let b_y1 = b[1] - b[3] / 2.0;
    let b_x2 = b[0] + b[2] / 2.0;
    let b_y2 = b[1] + b[3] / 2.0;

    let inter_w = (a_x2.min(b_x2) - a_x1.max(b_x1)).max(0.0);
    let inter_h = (a_y2.min(b_y2) - a_y1.max(b_y1)).max(0.0);
    let inter_area = inter_w * inter_h;

    let a_area = a[2] * a[3];
    let b_area = b[2] * b[3];
    let min_area = a_area.min(b_area);

    if min_area <= 0.0 {
        0.0
    } else {
        inter_area / min_area
    }
}

/// Compute Intersection over Union between two center-format bounding boxes.
fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let a_x1 = a[0] - a[2] / 2.0;
    let a_y1 = a[1] - a[3] / 2.0;
    let a_x2 = a[0] + a[2] / 2.0;
    let a_y2 = a[1] + a[3] / 2.0;

    let b_x1 = b[0] - b[2] / 2.0;
    let b_y1 = b[1] - b[3] / 2.0;
    let b_x2 = b[0] + b[2] / 2.0;
    let b_y2 = b[1] + b[3] / 2.0;

    let inter_x1 = a_x1.max(b_x1);
    let inter_y1 = a_y1.max(b_y1);
    let inter_x2 = a_x2.min(b_x2);
    let inter_y2 = a_y2.min(b_y2);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let a_area = a[2] * a[3];
    let b_area = b[2] * b[3];
    let union_area = a_area + b_area - inter_area;

    if union_area <= 0.0 {
        0.0
    } else {
        inter_area / union_area
    }
}

/// Convert center-format bbox `[cx, cy, w, h]` to corner format `[x_min, y_min, x_max, y_max]`.
pub fn bbox_to_corners(bbox: &[f32; 4]) -> [f32; 4] {
    [
        bbox[0] - bbox[2] / 2.0,
        bbox[1] - bbox[3] / 2.0,
        bbox[0] + bbox[2] / 2.0,
        bbox[1] + bbox[3] / 2.0,
    ]
}

/// Expand a bounding box by a scale factor (e.g., 1.5 = 50% larger).
pub fn expand_bbox(corners: &[f32; 4], scale: f32) -> [f32; 4] {
    let cx = (corners[0] + corners[2]) / 2.0;
    let cy = (corners[1] + corners[3]) / 2.0;
    let w = (corners[2] - corners[0]) * scale;
    let h = (corners[3] - corners[1]) * scale;

    [
        (cx - w / 2.0).max(0.0),
        (cy - h / 2.0).max(0.0),
        (cx + w / 2.0).min(1.0),
        (cy + h / 2.0).min(1.0),
    ]
}

/// Remap detection coordinates from letterboxed model space to image space.
///
/// After letterbox preprocessing, model detections are in padded model space.
/// This converts them back to normalized [0, 1] image coordinates.
pub fn remap_detection(
    detection: &mut Detection,
    params: &crate::inference::preprocessing::LetterboxParams,
) {
    detection.bbox[0] = params.to_image_x_normalized(detection.bbox[0]);
    detection.bbox[1] = params.to_image_y_normalized(detection.bbox[1]);
    detection.bbox[2] = params.to_image_w(detection.bbox[2]);
    detection.bbox[3] = params.to_image_h(detection.bbox[3]);

    for kp in &mut detection.keypoints {
        kp[0] = params.to_image_x_normalized(kp[0]);
        kp[1] = params.to_image_y_normalized(kp[1]);
    }
}

/// Sigmoid activation.
pub fn sigmoid(x: f32) -> f32 {
    let x_clamped = x.clamp(-80.0, 80.0);
    1.0 / (1.0 + (-x_clamped).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_basic_values() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn sigmoid_extreme_values() {
        // Should not overflow.
        assert!((sigmoid(100.0) - 1.0).abs() < 1e-6);
        assert!(sigmoid(-100.0).abs() < 1e-6);
    }

    #[test]
    fn iou_same_box() {
        let a = [0.5, 0.5, 0.2, 0.2];
        assert!((iou(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_no_overlap() {
        let a = [0.1, 0.1, 0.1, 0.1];
        let b = [0.9, 0.9, 0.1, 0.1];
        assert!(iou(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn nms_removes_duplicates() {
        let mut dets = vec![
            Detection {
                bbox: [0.5, 0.5, 0.2, 0.2],
                score: 0.9,
                keypoints: vec![],
            },
            Detection {
                bbox: [0.51, 0.51, 0.2, 0.2],
                score: 0.8,
                keypoints: vec![],
            },
            Detection {
                bbox: [0.1, 0.1, 0.1, 0.1],
                score: 0.7,
                keypoints: vec![],
            },
        ];
        non_max_suppression(&mut dets, 0.3);
        assert_eq!(dets.len(), 2);
    }

    #[test]
    fn generate_anchors_count() {
        let anchors = generate_ssd_anchors(192, &[8, 16, 16, 16], 2);
        // 192/8 = 24 -> 24*24*2 = 1152
        // 192/16 = 12 -> 12*12*2 = 288 (x3 strides)
        // Total: 1152 + 288*3 = 2016
        assert_eq!(anchors.len(), 2016);
    }

    #[test]
    fn bbox_to_corners_correct() {
        let bbox = [0.5, 0.5, 0.4, 0.4];
        let corners = bbox_to_corners(&bbox);
        assert!((corners[0] - 0.3).abs() < 1e-6);
        assert!((corners[1] - 0.3).abs() < 1e-6);
        assert!((corners[2] - 0.7).abs() < 1e-6);
        assert!((corners[3] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn expand_bbox_clamps() {
        let corners = [0.0, 0.0, 0.1, 0.1];
        let expanded = expand_bbox(&corners, 3.0);
        assert!(expanded[0] >= 0.0);
        assert!(expanded[1] >= 0.0);
    }
}
