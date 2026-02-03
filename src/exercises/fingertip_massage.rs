//! Fingertip-on-palm massage exercise (Moritz decoupling protocol).
//!
//! 30s per hand. Rubbing fingertips gently on the opposite palm.

use std::time::Duration;

use crate::detection::analyzer::min_inter_hand_fingertip_distance;
use crate::detection::types::{BfrbType, FaceDetection, HandDetection};
use crate::exercises::types::{Exercise, ExerciseCategory, VerificationResult};

pub struct FingertipMassage;

impl Exercise for FingertipMassage {
    fn id(&self) -> &str {
        "fingertip_massage"
    }

    fn name(&self) -> &str {
        "Fingertip-on-Palm Massage"
    }

    fn instructions(&self) -> &str {
        "Gently rub your fingertips across the palm of your other hand. \
         30 seconds per hand. Focus on the sensory experience."
    }

    fn category(&self) -> ExerciseCategory {
        ExerciseCategory::TimedHold
    }

    fn hold_duration(&self) -> Duration {
        Duration::from_secs(60) // 30s per hand.
    }

    fn target_reps(&self) -> u32 {
        0
    }

    fn applicable_to(&self) -> &[BfrbType] {
        &[BfrbType::NailPicking, BfrbType::SkinPicking]
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
                feedback: "Show both hands. Place fingertips on your other palm.".to_string(),
            };
        }

        let Some(hand1) = hands.first() else {
            return VerificationResult {
                pose_correct: false,
                confidence: 0.0,
                feedback: "Show both hands.".to_string(),
            };
        };
        let Some(hand2) = hands.get(1) else {
            return VerificationResult {
                pose_correct: false,
                confidence: 0.0,
                feedback: "Show both hands.".to_string(),
            };
        };

        // Hands should be close (fingertips on palm).
        let Some((dist, _, _)) = min_inter_hand_fingertip_distance(hand1, hand2) else {
            return VerificationResult {
                pose_correct: false,
                confidence: 0.0,
                feedback: "Bring your hands closer together.".to_string(),
            };
        };

        if dist < 0.12 {
            let confidence = 1.0 - (dist / 0.12).min(1.0);
            VerificationResult {
                pose_correct: true,
                confidence,
                feedback: "Good! Keep massaging your fingertips on your palm.".to_string(),
            }
        } else {
            VerificationResult {
                pose_correct: false,
                confidence: 0.0,
                feedback: "Place your fingertips on your other palm and rub gently.".to_string(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applicable_to_nail_picking() {
        let exercise = FingertipMassage;
        assert!(exercise.applicable_to().contains(&BfrbType::NailPicking));
    }

    #[test]
    fn is_timed_hold() {
        let exercise = FingertipMassage;
        assert_eq!(exercise.category(), ExerciseCategory::TimedHold);
    }
}
