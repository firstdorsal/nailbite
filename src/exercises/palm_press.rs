//! Palm-to-palm press exercise (HRT competing response).
//!
//! 60s hold. Both palms pressed together in front of chest.

use std::time::Duration;

use crate::detection::analyzer::{landmark_distance_2d, min_inter_hand_fingertip_distance};
use crate::detection::types::{BfrbType, FaceDetection, HandDetection, WRIST_INDEX};
use crate::exercises::types::{Exercise, ExerciseCategory, VerificationResult};

pub struct PalmPress;

impl Exercise for PalmPress {
    fn id(&self) -> &str {
        "palm_press"
    }

    fn name(&self) -> &str {
        "Palm-to-Palm Press"
    }

    fn instructions(&self) -> &str {
        "Press your palms firmly together in front of your chest \
         (prayer position). Hold for 60 seconds."
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
            BfrbType::HairPulling,
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
                feedback: "Press both palms together in front of the camera.".to_string(),
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

        // Check wrists are close (hands facing each other).
        let wrist1 = hand1.landmarks.get(WRIST_INDEX);
        let wrist2 = hand2.landmarks.get(WRIST_INDEX);

        let wrists_close = match (wrist1, wrist2) {
            (Some(w1), Some(w2)) => landmark_distance_2d(w1, w2) < 0.2,
            _ => false,
        };

        // Check fingertips are close (palms pressed).
        let tips_close = min_inter_hand_fingertip_distance(hand1, hand2)
            .map(|(dist, _, _)| dist < 0.08)
            .unwrap_or(false);

        if tips_close && wrists_close {
            VerificationResult {
                pose_correct: true,
                confidence: 0.9,
                feedback: "Good! Keep pressing your palms together.".to_string(),
            }
        } else if tips_close {
            VerificationResult {
                pose_correct: true,
                confidence: 0.7,
                feedback: "Palms are touching. Press more firmly.".to_string(),
            }
        } else {
            VerificationResult {
                pose_correct: false,
                confidence: 0.0,
                feedback: "Press your palms flat against each other.".to_string(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_timed_hold() {
        let exercise = PalmPress;
        assert_eq!(exercise.category(), ExerciseCategory::TimedHold);
        assert_eq!(exercise.hold_duration(), Duration::from_secs(60));
    }

    #[test]
    fn applicable_to_all_major_bfrbs() {
        let exercise = PalmPress;
        let types = exercise.applicable_to();
        assert!(types.contains(&BfrbType::NailBiting));
        assert!(types.contains(&BfrbType::NailPicking));
    }
}
