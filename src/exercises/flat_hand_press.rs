//! Flat hand press exercise (HRT competing response).
//!
//! 60s hold. Both hands flat on desk, fingers spread wide.

use std::time::Duration;

use crate::detection::analyzer::finger_extension_ratio;
use crate::detection::types::{BfrbType, FaceDetection, HandDetection};
use crate::exercises::types::{Exercise, ExerciseCategory, VerificationResult};

pub struct FlatHandPress;

impl Exercise for FlatHandPress {
    fn id(&self) -> &str {
        "flat_hand_press"
    }

    fn name(&self) -> &str {
        "Flat Hand Press"
    }

    fn instructions(&self) -> &str {
        "Place both hands flat on a surface with fingers spread wide. \
         Press down firmly and hold for 60 seconds."
    }

    fn category(&self) -> ExerciseCategory {
        ExerciseCategory::TimedHold
    }

    fn hold_duration(&self) -> Duration {
        Duration::from_secs(60)
    }

    fn target_reps(&self) -> u32 {
        0
    }

    fn applicable_to(&self) -> &[BfrbType] {
        &[
            BfrbType::NailBiting,
            BfrbType::NailPicking,
            BfrbType::SkinPicking,
        ]
    }

    fn verify(
        &self,
        hands: &[HandDetection],
        _face: &Option<FaceDetection>,
    ) -> VerificationResult {
        if hands.len() < 2 {
            return VerificationResult {
                pose_correct: false,
                confidence: 0.0,
                feedback: "Show both hands flat to the camera.".to_string(),
            };
        }

        let mut total_ext = 0.0;
        for hand in hands {
            total_ext += finger_extension_ratio(hand);
        }
        let avg_ext = total_ext / hands.len() as f32;

        if avg_ext >= 0.75 {
            VerificationResult {
                pose_correct: true,
                confidence: avg_ext,
                feedback: "Good! Keep your hands flat and pressed down.".to_string(),
            }
        } else {
            VerificationResult {
                pose_correct: false,
                confidence: avg_ext,
                feedback: "Spread your fingers wide and press flat.".to_string(),
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::detection::types::Landmark;

    fn make_flat_hand(x: f32) -> HandDetection {
        let mut landmarks = [Landmark { x: 0.0, y: 0.0, z: 0.0 }; 21];
        landmarks[0] = Landmark { x, y: 0.8, z: 0.0 };
        for &idx in &[1, 5, 9, 13, 17] {
            landmarks[idx] = Landmark { x, y: 0.6, z: 0.0 };
        }
        // DIP joints farther from wrist than MCPs = extended.
        for &idx in &[3, 7, 11, 15, 19] {
            landmarks[idx] = Landmark { x, y: 0.5, z: 0.0 };
        }
        for &idx in &[4, 8, 12, 16, 20] {
            landmarks[idx] = Landmark { x, y: 0.4, z: 0.0 };
        }
        HandDetection { side: None, landmarks, confidence: 1.0 }
    }

    #[test]
    fn accepts_two_flat_hands() {
        let exercise = FlatHandPress;
        let result = exercise.verify(&[make_flat_hand(0.3), make_flat_hand(0.7)], &None);
        assert!(result.pose_correct, "feedback: {}", result.feedback);
    }

    #[test]
    fn rejects_one_hand() {
        let exercise = FlatHandPress;
        let result = exercise.verify(&[make_flat_hand(0.5)], &None);
        assert!(!result.pose_correct);
    }
}
