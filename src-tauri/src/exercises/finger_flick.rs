//! Finger extension flick exercise (Moritz decoupling protocol).
//!
//! 10 reps per hand. Rapid curl-to-extend finger motion away from body.

use std::time::Duration;

use crate::detection::analyzer::finger_extension_ratio;
use crate::detection::types::{BfrbType, FaceDetection, HandDetection};
use crate::exercises::types::{Exercise, ExerciseCategory, VerificationResult};

pub struct FingerFlick;

impl Exercise for FingerFlick {
    fn id(&self) -> &str {
        "finger_flick"
    }

    fn name(&self) -> &str {
        "Finger Extension Flick"
    }

    fn instructions(&self) -> &str {
        "Rapidly extend your fingers outward from a curled position. \
         Flick your fingers open with force. Repeat 10 times per hand."
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
        &[BfrbType::NailPicking, BfrbType::SkinPicking]
    }

    fn verify(
        &self,
        hands: &[HandDetection],
        _face: &Option<FaceDetection>,
    ) -> VerificationResult {
        if hands.is_empty() {
            return VerificationResult {
                pose_correct: false,
                confidence: 0.0,
                feedback: "Show your hand to the camera.".to_string(),
            };
        }

        // Detect extension (flicked open). A rep counts when fingers go from
        // curled to fully extended. We check for the extended state here;
        // the verification loop tracks transitions.
        let max_ext = hands
            .iter()
            .map(finger_extension_ratio)
            .fold(0.0_f32, f32::max);

        if max_ext >= 0.75 {
            VerificationResult {
                pose_correct: true,
                confidence: max_ext,
                feedback: "Good flick! Now curl and flick again.".to_string(),
            }
        } else {
            VerificationResult {
                pose_correct: false,
                confidence: max_ext,
                feedback: "Extend your fingers fully outward.".to_string(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applicable_to_nail_picking() {
        let exercise = FingerFlick;
        assert!(exercise.applicable_to().contains(&BfrbType::NailPicking));
    }

    #[test]
    fn is_repetition_exercise() {
        let exercise = FingerFlick;
        assert_eq!(exercise.category(), ExerciseCategory::Repetitions);
        assert_eq!(exercise.target_reps(), 10);
    }
}
