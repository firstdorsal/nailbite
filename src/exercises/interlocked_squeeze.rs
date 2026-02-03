//! Interlocked finger squeeze exercise (HRT competing response).
//!
//! 60s hold. Both hands clasped together with fingers interleaved.

use std::time::Duration;

use crate::detection::analyzer::min_inter_hand_fingertip_distance;
use crate::detection::types::{BfrbType, FaceDetection, HandDetection};
use crate::exercises::types::{Exercise, ExerciseCategory, VerificationResult};

pub struct InterlockedSqueeze;

impl Exercise for InterlockedSqueeze {
    fn id(&self) -> &str {
        "interlocked_squeeze"
    }

    fn name(&self) -> &str {
        "Interlocked Squeeze"
    }

    fn instructions(&self) -> &str {
        "Clasp your hands together with fingers interlocked. \
         Squeeze firmly and hold for 60 seconds."
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
            BfrbType::HairPulling,
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
                feedback: "Clasp both hands together in front of the camera.".to_string(),
            };
        }

        // Check that hands are close together (interlocked).
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

        let Some((dist, _, _)) = min_inter_hand_fingertip_distance(hand1, hand2) else {
            return VerificationResult {
                pose_correct: false,
                confidence: 0.0,
                feedback: "Bring your hands closer together.".to_string(),
            };
        };

        // Interlocked hands have very close fingertips (< 0.1 normalized).
        if dist < 0.1 {
            let confidence = 1.0 - (dist / 0.1);
            VerificationResult {
                pose_correct: true,
                confidence,
                feedback: "Good! Keep your fingers interlocked and squeeze.".to_string(),
            }
        } else {
            VerificationResult {
                pose_correct: false,
                confidence: 0.0,
                feedback: "Interlock your fingers and squeeze your hands together.".to_string(),
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::detection::types::Landmark;

    fn make_interlocked_hand(x: f32) -> HandDetection {
        let mut landmarks = [Landmark { x: 0.0, y: 0.0, z: 0.0 }; 21];
        landmarks[0] = Landmark { x, y: 0.7, z: 0.0 };
        for &idx in &[1, 5, 9, 13, 17] {
            landmarks[idx] = Landmark { x, y: 0.55, z: 0.0 };
        }
        // Fingertips at the same position (interlocked).
        for &idx in &[4, 8, 12, 16, 20] {
            landmarks[idx] = Landmark { x: 0.5, y: 0.5, z: 0.0 };
        }
        HandDetection { side: None, landmarks, confidence: 1.0 }
    }

    #[test]
    fn accepts_interlocked_hands() {
        let exercise = InterlockedSqueeze;
        let result = exercise.verify(
            &[make_interlocked_hand(0.45), make_interlocked_hand(0.55)],
            &None,
        );
        assert!(result.pose_correct, "feedback: {}", result.feedback);
    }
}
