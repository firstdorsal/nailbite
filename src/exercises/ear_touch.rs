//! Ear touch deflection exercise (Moritz decoupling protocol).
//!
//! 10 reps. Hand moves toward face then deflects to touch the ear.

use std::time::Duration;

use crate::detection::analyzer::min_fingertip_distance;
use crate::detection::types::{
    BfrbType, FaceDetection, HandDetection, EAR_LEFT_INDEX, EAR_RIGHT_INDEX,
};
use crate::exercises::types::{Exercise, ExerciseCategory, VerificationResult};

pub struct EarTouch;

impl Exercise for EarTouch {
    fn id(&self) -> &str {
        "ear_touch"
    }

    fn name(&self) -> &str {
        "Ear Touch Deflection"
    }

    fn instructions(&self) -> &str {
        "Move your hand toward your face, then deflect it to touch your ear. \
         Repeat 10 times. This retrains the hand-to-face movement pattern."
    }

    fn category(&self) -> ExerciseCategory {
        ExerciseCategory::Repetitions
    }

    fn hold_duration(&self) -> Duration {
        Duration::ZERO
    }

    fn target_reps(&self) -> u32 {
        10
    }

    fn applicable_to(&self) -> &[BfrbType] {
        &[BfrbType::NailBiting, BfrbType::LipBiting]
    }

    fn verify(
        &self,
        hands: &[HandDetection],
        face: &Option<FaceDetection>,
    ) -> VerificationResult {
        let Some(face) = face else {
            return VerificationResult {
                pose_correct: false,
                confidence: 0.0,
                feedback: "Face not visible. Look at the camera.".to_string(),
            };
        };

        if hands.is_empty() {
            return VerificationResult {
                pose_correct: false,
                confidence: 0.0,
                feedback: "Move your hand toward your ear.".to_string(),
            };
        }

        // Check if any fingertip is near either ear landmark.
        let ear_left = face.landmarks.get(EAR_LEFT_INDEX);
        let ear_right = face.landmarks.get(EAR_RIGHT_INDEX);

        let mut min_ear_dist = f32::MAX;

        for hand in hands {
            for ear in [ear_left, ear_right].into_iter().flatten() {
                if let Some((dist, _)) = min_fingertip_distance(hand, ear) {
                    if dist < min_ear_dist {
                        min_ear_dist = dist;
                    }
                }
            }
        }

        // Ear touch threshold: fingertip within 0.1 normalized distance of ear.
        if min_ear_dist < 0.1 {
            let confidence = 1.0 - (min_ear_dist / 0.1);
            VerificationResult {
                pose_correct: true,
                confidence,
                feedback: "Good ear touch! Now lower your hand and repeat.".to_string(),
            }
        } else {
            VerificationResult {
                pose_correct: false,
                confidence: 0.0,
                feedback: "Touch your ear with your fingertips.".to_string(),
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::detection::types::Landmark;

    #[test]
    fn rejects_without_face() {
        let exercise = EarTouch;
        let hand = HandDetection {
            side: None,
            landmarks: [Landmark { x: 0.0, y: 0.0, z: 0.0 }; 21],
            confidence: 1.0,
        };
        let result = exercise.verify(&[hand], &None);
        assert!(!result.pose_correct);
    }

    #[test]
    fn applicable_to_nail_biting() {
        let exercise = EarTouch;
        assert!(exercise.applicable_to().contains(&BfrbType::NailBiting));
    }
}
