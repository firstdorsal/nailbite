//! Fist clench exercise (HRT competing response).
//!
//! 60s hold. All fingertips collapsed to palm, thumb across fingers.

use std::time::Duration;

use crate::detection::analyzer::finger_curl_ratio;
use crate::detection::types::{BfrbType, FaceDetection, HandDetection};
use crate::exercises::types::{Exercise, ExerciseCategory, VerificationResult};

pub struct FistClench;

impl Exercise for FistClench {
    fn id(&self) -> &str {
        "fist_clench"
    }

    fn name(&self) -> &str {
        "Fist Clench"
    }

    fn instructions(&self) -> &str {
        "Make tight fists with both hands. Squeeze firmly and hold for 60 seconds. \
         Keep your thumbs wrapped over your fingers."
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
        if hands.is_empty() {
            return VerificationResult {
                pose_correct: false,
                confidence: 0.0,
                feedback: "Show your hands to the camera.".to_string(),
            };
        }

        let mut total_curl = 0.0;
        for hand in hands {
            total_curl += finger_curl_ratio(hand);
        }
        let avg_curl = total_curl / hands.len() as f32;

        // Fist: all fingers curled (curl ratio > 0.75).
        if avg_curl >= 0.75 {
            VerificationResult {
                pose_correct: true,
                confidence: avg_curl,
                feedback: "Good! Keep holding your fists tight.".to_string(),
            }
        } else if avg_curl >= 0.5 {
            VerificationResult {
                pose_correct: false,
                confidence: avg_curl,
                feedback: "Curl your fingers tighter into fists.".to_string(),
            }
        } else {
            VerificationResult {
                pose_correct: false,
                confidence: avg_curl,
                feedback: "Make fists by curling all fingers into your palms.".to_string(),
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::detection::types::Landmark;

    fn make_fist() -> HandDetection {
        let mut landmarks = [Landmark { x: 0.0, y: 0.0, z: 0.0 }; 21];
        // Wrist at origin.
        landmarks[0] = Landmark { x: 0.5, y: 0.8, z: 0.0 };
        // MCP joints extended.
        for &idx in &[1, 5, 9, 13, 17] {
            landmarks[idx] = Landmark { x: 0.5, y: 0.6, z: 0.0 };
        }
        // Fingertips curled back toward wrist (close to palm).
        for &idx in &[4, 8, 12, 16, 20] {
            landmarks[idx] = Landmark { x: 0.5, y: 0.72, z: 0.0 };
        }
        HandDetection { side: None, landmarks, confidence: 1.0 }
    }

    #[test]
    fn accepts_clenched_fist() {
        let exercise = FistClench;
        let result = exercise.verify(&[make_fist()], &None);
        assert!(result.pose_correct, "feedback: {}", result.feedback);
    }

    #[test]
    fn rejects_no_hands() {
        let exercise = FistClench;
        let result = exercise.verify(&[], &None);
        assert!(!result.pose_correct);
    }
}
