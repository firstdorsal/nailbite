//! Spatial analysis utilities for landmark proximity and pose classification.
//!
//! Provides functions for computing distances between landmarks,
//! normalizing by face width, and classifying hand poses.

use crate::detection::types::{
    FaceDetection, HandDetection, Landmark, FACE_LEFT_INDEX, FACE_RIGHT_INDEX, FINGERTIP_INDICES,
    FINGER_DIP_INDICES, FINGER_MCP_INDICES, INNER_LIP_INDICES, WRIST_INDEX,
};

/// Euclidean distance between two 2D landmarks (ignoring Z).
pub fn landmark_distance_2d(a: &Landmark, b: &Landmark) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

/// Compute the face width from face mesh landmarks (left-right boundary).
/// Returns `None` if the face doesn't have enough landmarks.
pub fn face_width(face: &FaceDetection) -> Option<f32> {
    let left = face.landmarks.get(FACE_LEFT_INDEX)?;
    let right = face.landmarks.get(FACE_RIGHT_INDEX)?;
    Some(landmark_distance_2d(left, right))
}

/// Compute the mouth center from inner lip landmarks.
/// Returns the average position of all inner lip landmarks.
pub fn mouth_center(face: &FaceDetection) -> Option<Landmark> {
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_z = 0.0;
    let mut count = 0;

    for &idx in &INNER_LIP_INDICES {
        if let Some(lm) = face.landmarks.get(idx) {
            sum_x += lm.x;
            sum_y += lm.y;
            sum_z += lm.z;
            count += 1;
        }
    }

    if count == 0 {
        return None;
    }

    let n = count as f32;
    Some(Landmark {
        x: sum_x / n,
        y: sum_y / n,
        z: sum_z / n,
    })
}

/// Compute the minimum distance from any fingertip of a hand to a target landmark.
/// Returns (min_distance, fingertip_index).
pub fn min_fingertip_distance(hand: &HandDetection, target: &Landmark) -> Option<(f32, usize)> {
    let mut min_dist = f32::MAX;
    let mut min_idx = 0;

    for &tip_idx in &FINGERTIP_INDICES {
        if let Some(tip) = hand.landmarks.get(tip_idx) {
            let dist = landmark_distance_2d(tip, target);
            if dist < min_dist {
                min_dist = dist;
                min_idx = tip_idx;
            }
        }
    }

    if min_dist < f32::MAX {
        Some((min_dist, min_idx))
    } else {
        None
    }
}

/// Check if a hand's fingers are curled (typical nail biting pose).
///
/// Fingers are considered curled when fingertips are closer to the wrist
/// than their MCP joints (knuckles) are. Returns the fraction of fingers
/// that are curled [0.0, 1.0].
pub fn finger_curl_ratio(hand: &HandDetection) -> f32 {
    let Some(wrist) = hand.landmarks.get(WRIST_INDEX) else {
        return 0.0;
    };

    let mut curled = 0;
    let mut total = 0;

    // Check fingers 1-4 (index through pinky). Skip thumb (different geometry).
    for i in 1..5 {
        let Some(&tip_idx) = FINGERTIP_INDICES.get(i) else {
            continue;
        };
        let Some(&mcp_idx) = FINGER_MCP_INDICES.get(i) else {
            continue;
        };

        let Some(tip) = hand.landmarks.get(tip_idx) else {
            continue;
        };
        let Some(mcp) = hand.landmarks.get(mcp_idx) else {
            continue;
        };

        let tip_to_wrist = landmark_distance_2d(tip, wrist);
        let mcp_to_wrist = landmark_distance_2d(mcp, wrist);

        total += 1;
        if tip_to_wrist < mcp_to_wrist {
            curled += 1;
        }
    }

    if total == 0 {
        0.0
    } else {
        curled as f32 / total as f32
    }
}

/// Check if fingers are extended/spread (flat hand pose).
///
/// Fingers are extended when DIP joints are farther from wrist than MCP joints.
pub fn finger_extension_ratio(hand: &HandDetection) -> f32 {
    let Some(wrist) = hand.landmarks.get(WRIST_INDEX) else {
        return 0.0;
    };

    let mut extended = 0;
    let mut total = 0;

    for i in 1..5 {
        let Some(&dip_idx) = FINGER_DIP_INDICES.get(i) else {
            continue;
        };
        let Some(&mcp_idx) = FINGER_MCP_INDICES.get(i) else {
            continue;
        };

        let Some(dip) = hand.landmarks.get(dip_idx) else {
            continue;
        };
        let Some(mcp) = hand.landmarks.get(mcp_idx) else {
            continue;
        };

        let dip_to_wrist = landmark_distance_2d(dip, wrist);
        let mcp_to_wrist = landmark_distance_2d(mcp, wrist);

        total += 1;
        if dip_to_wrist > mcp_to_wrist {
            extended += 1;
        }
    }

    if total == 0 {
        0.0
    } else {
        extended as f32 / total as f32
    }
}

/// Detect typing posture: both hands spread on a keyboard at similar Y, with
/// fingers extended downward and NOT touching across hands.
///
/// We require horizontal separation between wrists and an inter-hand
/// fingertip gap because nail picking ALSO has both hands at similar Y with
/// extended fingers — the discriminator is that picking brings the hands
/// together while typing keeps them apart. Without these checks the
/// nail-picking detector gets suppressed whenever both hands are visible.
pub fn is_typing_posture(hands: &[HandDetection]) -> bool {
    if hands.len() < 2 {
        return false;
    }

    // Both hands must have most fingers extended downward.
    let extensions: Vec<f32> = hands.iter().map(finger_extension_ratio).collect();
    let both_extended = extensions.iter().all(|&ext| ext >= 0.5);
    if !both_extended {
        return false;
    }

    let Some(w0) = hands.first().and_then(|h| h.landmarks.get(WRIST_INDEX)) else {
        return false;
    };
    let Some(w1) = hands.get(1).and_then(|h| h.landmarks.get(WRIST_INDEX)) else {
        return false;
    };

    // Wrists at similar Y level (within 10% of frame height).
    if (w0.y - w1.y).abs() >= 0.10 {
        return false;
    }

    // Wrists horizontally separated (typical keyboard span ≥ ~20% of frame
    // width). Picking brings hands close horizontally; typing keeps them
    // apart on the home row.
    if (w0.x - w1.x).abs() < 0.20 {
        return false;
    }

    // Fingertips must NOT be touching across hands — picking has at least
    // one fingertip pair within ~10% of frame width, typing never does.
    if let Some(hand0) = hands.first() {
        if let Some(hand1) = hands.get(1) {
            if let Some((dist, _, _)) = min_inter_hand_fingertip_distance(hand0, hand1) {
                if dist < 0.10 {
                    return false;
                }
            }
        }
    }

    true
}

/// Detect chin rest posture: wrist/palm near face with fingers NOT at mouth.
///
/// Returns true if the hand appears to be resting under the chin rather than
/// having fingers at the mouth.
pub fn is_chin_rest(hand: &HandDetection, face: &FaceDetection) -> bool {
    let Some(wrist) = hand.landmarks.get(WRIST_INDEX) else {
        return false;
    };
    let Some(mouth) = mouth_center(face) else {
        return false;
    };
    let Some(fw) = face_width(face) else {
        return false;
    };

    if fw <= 0.0 {
        return false;
    }

    // Wrist must be near the face (within 1.5x face width).
    let wrist_to_mouth = landmark_distance_2d(wrist, &mouth);
    let wrist_near_face = wrist_to_mouth / fw < 1.5;

    // But fingers must be extended (not curled toward mouth).
    let fingers_extended = finger_extension_ratio(hand) >= 0.5;

    wrist_near_face && fingers_extended
}

/// Minimum distance between fingertips of two hands.
/// Returns (distance, hand1_tip_idx, hand2_tip_idx).
pub fn min_inter_hand_fingertip_distance(
    hand1: &HandDetection,
    hand2: &HandDetection,
) -> Option<(f32, usize, usize)> {
    let mut min_dist = f32::MAX;
    let mut best_pair = (0, 0);

    for &tip1 in &FINGERTIP_INDICES {
        let Some(t1) = hand1.landmarks.get(tip1) else {
            continue;
        };
        for &tip2 in &FINGERTIP_INDICES {
            let Some(t2) = hand2.landmarks.get(tip2) else {
                continue;
            };
            let dist = landmark_distance_2d(t1, t2);
            if dist < min_dist {
                min_dist = dist;
                best_pair = (tip1, tip2);
            }
        }
    }

    if min_dist < f32::MAX {
        Some((min_dist, best_pair.0, best_pair.1))
    } else {
        None
    }
}

/// Check if a hand has a pinching gesture (thumb tip close to index tip).
pub fn is_pinching(hand: &HandDetection) -> bool {
    // Thumb tip = index 4, Index tip = index 8
    let Some(thumb) = hand.landmarks.get(4) else {
        return false;
    };
    let Some(index) = hand.landmarks.get(8) else {
        return false;
    };

    // Get hand scale: distance from wrist to middle finger MCP.
    let Some(wrist) = hand.landmarks.get(WRIST_INDEX) else {
        return false;
    };
    let Some(middle_mcp) = hand.landmarks.get(9) else {
        return false;
    };

    let hand_scale = landmark_distance_2d(wrist, middle_mcp);
    if hand_scale <= 0.0 {
        return false;
    }

    let pinch_distance = landmark_distance_2d(thumb, index);
    pinch_distance / hand_scale < 0.4
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_landmark(x: f32, y: f32) -> Landmark {
        Landmark { x, y, z: 0.0 }
    }

    #[test]
    fn distance_2d_same_point() {
        let a = make_landmark(0.5, 0.5);
        assert!(landmark_distance_2d(&a, &a) < 1e-6);
    }

    #[test]
    fn distance_2d_known_value() {
        let a = make_landmark(0.0, 0.0);
        let b = make_landmark(3.0, 4.0);
        assert!((landmark_distance_2d(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[allow(clippy::indexing_slicing)] // Test code with known array sizes.
    #[test]
    fn face_width_returns_value() {
        let mut landmarks = vec![Landmark { x: 0.0, y: 0.0, z: 0.0 }; 468];
        landmarks[FACE_LEFT_INDEX] = make_landmark(0.2, 0.5);
        landmarks[FACE_RIGHT_INDEX] = make_landmark(0.8, 0.5);

        let face = FaceDetection {
            landmarks,
            confidence: 1.0,
        };

        let width = face_width(&face).unwrap();
        assert!((width - 0.6).abs() < 1e-6);
    }

    #[allow(clippy::indexing_slicing)]
    #[test]
    fn mouth_center_returns_average() {
        let mut landmarks = vec![Landmark { x: 0.0, y: 0.0, z: 0.0 }; 468];
        for &idx in &INNER_LIP_INDICES {
            landmarks[idx] = make_landmark(0.5, 0.6);
        }

        let face = FaceDetection {
            landmarks,
            confidence: 1.0,
        };

        let center = mouth_center(&face).unwrap();
        assert!((center.x - 0.5).abs() < 1e-6);
        assert!((center.y - 0.6).abs() < 1e-6);
    }

    #[allow(clippy::indexing_slicing)]
    #[test]
    fn pinching_detection() {
        let mut landmarks = [Landmark { x: 0.0, y: 0.0, z: 0.0 }; 21];
        landmarks[WRIST_INDEX] = make_landmark(0.0, 0.0);
        landmarks[9] = make_landmark(0.0, 0.5);
        landmarks[4] = make_landmark(0.1, 0.3);
        landmarks[8] = make_landmark(0.11, 0.31);

        let hand = HandDetection {
            side: None,
            landmarks,
            confidence: 1.0,
        };

        assert!(is_pinching(&hand));
    }

    #[allow(clippy::indexing_slicing)]
    #[test]
    fn not_pinching_when_spread() {
        let mut landmarks = [Landmark { x: 0.0, y: 0.0, z: 0.0 }; 21];
        landmarks[WRIST_INDEX] = make_landmark(0.0, 0.0);
        landmarks[9] = make_landmark(0.0, 0.5);
        landmarks[4] = make_landmark(0.3, 0.3);
        landmarks[8] = make_landmark(-0.2, 0.5);

        let hand = HandDetection {
            side: None,
            landmarks,
            confidence: 1.0,
        };

        assert!(!is_pinching(&hand));
    }
}
