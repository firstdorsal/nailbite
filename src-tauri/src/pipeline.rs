//! Inference pipeline utilities.
//!
//! Contains face landmark smoothing for temporal consistency.

use crate::detection::types::FaceDetection;

/// Apply exponential moving average smoothing to face landmarks.
///
/// Blends the current face detection landmarks with the previous frame's
/// landmarks using `alpha` as the weight for new values. Updates `prev`
/// with the smoothed result for the next frame.
///
/// When face detection fails, uses the previous face for a grace period
/// (tracked by `miss_count`) before clearing. This prevents flickering
/// from brief detection gaps.
pub fn smooth_face_landmarks_with_grace(
    current: &mut Option<FaceDetection>,
    prev: &mut Option<FaceDetection>,
    miss_count: &mut u8,
    alpha: f32,
    grace_frames: u8,
) {
    let one_minus_alpha = 1.0 - alpha;

    match (current.as_mut(), prev.as_ref()) {
        (Some(cur), Some(prv)) if cur.landmarks.len() == prv.landmarks.len() => {
            // Face detected and matches previous; blend with EMA
            for (c, p) in cur.landmarks.iter_mut().zip(&prv.landmarks) {
                c.x = alpha.mul_add(c.x, one_minus_alpha * p.x);
                c.y = alpha.mul_add(c.y, one_minus_alpha * p.y);
                c.z = alpha.mul_add(c.z, one_minus_alpha * p.z);
            }
            *prev = Some(cur.clone());
            *miss_count = 0;
        }
        (Some(cur), _) => {
            // No previous frame or landmark count changed; adopt current as-is.
            *prev = Some(cur.clone());
            *miss_count = 0;
        }
        (None, Some(prv)) if *miss_count < grace_frames => {
            // No face detected but within grace period; use previous face
            *current = Some(prv.clone());
            *miss_count += 1;
        }
        (None, _) => {
            // No face detected and grace period expired; clear previous
            *prev = None;
            *miss_count = 0;
        }
    }
}

/// Apply exponential moving average smoothing to face landmarks.
///
/// Simplified version without grace period - clears immediately when face is lost.
pub fn smooth_face_landmarks(
    current: &mut Option<FaceDetection>,
    prev: &mut Option<FaceDetection>,
    alpha: f32,
) {
    let mut miss_count = 0;
    smooth_face_landmarks_with_grace(current, prev, &mut miss_count, alpha, 0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detection::types::Landmark;

    #[test]
    fn smooth_face_landmarks_blends_with_previous() {
        let lm = |x, y, z| Landmark { x, y, z };
        let mut current = Some(FaceDetection {
            landmarks: vec![lm(1.0, 1.0, 1.0)],
            confidence: 0.9,
        });
        let mut prev = Some(FaceDetection {
            landmarks: vec![lm(0.0, 0.0, 0.0)],
            confidence: 0.9,
        });

        smooth_face_landmarks(&mut current, &mut prev, 0.5);

        let cur = current.as_ref().unwrap();
        assert!((cur.landmarks.first().unwrap().x - 0.5).abs() < 0.001);
        assert!((cur.landmarks.first().unwrap().y - 0.5).abs() < 0.001);
        assert!((cur.landmarks.first().unwrap().z - 0.5).abs() < 0.001);
    }

    #[test]
    fn smooth_face_landmarks_adopts_first_frame() {
        let lm = |x, y, z| Landmark { x, y, z };
        let mut current = Some(FaceDetection {
            landmarks: vec![lm(0.5, 0.5, 0.5)],
            confidence: 0.9,
        });
        let mut prev = None;

        smooth_face_landmarks(&mut current, &mut prev, 0.5);

        let cur = current.as_ref().unwrap();
        assert!((cur.landmarks.first().unwrap().x - 0.5).abs() < 0.001);
        assert!(prev.is_some());
    }

    #[test]
    fn smooth_face_landmarks_clears_on_no_face() {
        let lm = |x, y, z| Landmark { x, y, z };
        let mut current = None;
        let mut prev = Some(FaceDetection {
            landmarks: vec![lm(0.5, 0.5, 0.5)],
            confidence: 0.9,
        });

        smooth_face_landmarks(&mut current, &mut prev, 0.5);

        assert!(prev.is_none());
    }
}
