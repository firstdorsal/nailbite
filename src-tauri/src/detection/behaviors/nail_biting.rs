//! Nail biting behavior detector.
//!
//! Detects when fingertips are in close proximity to the mouth region,
//! with hand pose filtering to reduce false positives from chin resting
//! and typing.

use std::time::Duration;

use tracing::debug;

use crate::config::BehaviorConfig;
use crate::detection::analyzer::{
    face_width, finger_curl_ratio, is_chin_rest, is_typing_posture, min_fingertip_distance,
    mouth_center,
};
use crate::detection::behaviors::BehaviorDetector;
use crate::detection::types::{BfrbType, FrameAnalysis};

pub struct NailBitingDetector {
    proximity_threshold: f32,
    min_sustained_ms: u64,
    confidence_threshold: f32,
    chin_rest_suppression: bool,
    typing_suppression: bool,
}

impl NailBitingDetector {
    pub fn new(
        config: &BehaviorConfig,
        chin_rest_suppression: bool,
        typing_suppression: bool,
    ) -> Self {
        Self {
            proximity_threshold: config.proximity_threshold,
            min_sustained_ms: config.min_sustained_ms,
            confidence_threshold: config.confidence_threshold,
            chin_rest_suppression,
            typing_suppression,
        }
    }
}

impl BehaviorDetector for NailBitingDetector {
    fn bfrb_type(&self) -> BfrbType {
        BfrbType::NailBiting
    }

    fn name(&self) -> &str {
        "Nail Biting"
    }

    fn analyze_frame(&self, analysis: &FrameAnalysis) -> Option<f32> {
        // Require at least one hand and a face with mouth landmarks.
        if analysis.hands.is_empty() {
            debug!("NailBiting: no hands detected");
            return None;
        }
        let face = analysis.face.as_ref()?;
        let mouth = mouth_center(face)?;
        let fw = face_width(face)?;

        if fw <= 0.0 {
            debug!(face_width = fw, "NailBiting: invalid face width");
            return None;
        }

        // Check typing suppression.
        if self.typing_suppression && is_typing_posture(&analysis.hands) {
            debug!("NailBiting: typing posture suppression");
            return Some(0.0);
        }

        let mut max_confidence = 0.0_f32;

        for (hand_idx, hand) in analysis.hands.iter().enumerate() {
            // Find closest fingertip to mouth.
            let Some((distance, tip_idx)) = min_fingertip_distance(hand, &mouth) else {
                continue;
            };

            // Normalize distance by face width.
            let normalized_dist = distance / fw;

            // Suppress chin rest: hand near face but fingers extended AND
            // fingertips not close to mouth. If a fingertip IS near the mouth,
            // it's likely a BFRB, not a chin rest.
            if self.chin_rest_suppression
                && normalized_dist > self.proximity_threshold * 0.8
                && is_chin_rest(hand, face)
            {
                debug!(hand = hand_idx, "NailBiting: chin rest suppression");
                continue;
            }

            let curl = finger_curl_ratio(hand);

            debug!(
                hand = hand_idx,
                tip = tip_idx,
                raw_dist = distance,
                face_width = fw,
                norm_dist = normalized_dist,
                threshold = self.proximity_threshold,
                curl = curl,
                "NailBiting: hand analysis"
            );

            if normalized_dist > self.proximity_threshold {
                continue;
            }

            // Compute confidence: closer to mouth = higher confidence.
            // At proximity_threshold -> 0.0, at 0 distance -> 1.0.
            let proximity_score = 1.0 - (normalized_dist / self.proximity_threshold);

            // Use proximity score directly. Curl only adds a small bonus
            // to avoid penalizing valid poses with noisy curl estimates.
            let confidence = proximity_score * (0.8 + 0.2 * curl);
            max_confidence = max_confidence.max(confidence);

            debug!(
                hand = hand_idx,
                proximity_score = proximity_score,
                curl = curl,
                confidence = confidence,
                "NailBiting: confidence computed"
            );
        }

        Some(max_confidence)
    }

    fn min_sustained_duration(&self) -> Duration {
        Duration::from_millis(self.min_sustained_ms)
    }

    fn confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }

    fn requires_face(&self) -> bool {
        true
    }

    fn requires_hands(&self) -> bool {
        true
    }
}

#[cfg(test)]
#[allow(clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::detection::types::{FaceDetection, HandDetection, Landmark};
    use std::sync::Arc;
    use std::time::Instant;

    fn default_config() -> BehaviorConfig {
        BehaviorConfig {
            enabled: true,
            proximity_threshold: 0.35,
            min_sustained_ms: 1500,
            confidence_threshold: 0.3,
        }
    }

    fn make_face_with_mouth(mouth_x: f32, mouth_y: f32) -> FaceDetection {
        let mut landmarks = vec![Landmark { x: 0.0, y: 0.0, z: 0.0 }; 468];
        landmarks[234] = Landmark {
            x: mouth_x - 0.15,
            y: mouth_y,
            z: 0.0,
        };
        landmarks[454] = Landmark {
            x: mouth_x + 0.15,
            y: mouth_y,
            z: 0.0,
        };
        for &idx in &crate::detection::types::INNER_LIP_INDICES {
            landmarks[idx] = Landmark {
                x: mouth_x,
                y: mouth_y,
                z: 0.0,
            };
        }
        FaceDetection {
            landmarks,
            confidence: 1.0,
        }
    }

    fn make_curled_hand_near_mouth(mouth_x: f32, mouth_y: f32) -> HandDetection {
        let mut landmarks = [Landmark {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }; 21];
        // Wrist below the mouth.
        landmarks[0] = Landmark {
            x: mouth_x,
            y: mouth_y + 0.25,
            z: 0.0,
        };
        // MCP joints (knuckles) extended toward mouth direction.
        for &idx in &[1, 5, 9, 13, 17] {
            landmarks[idx] = Landmark {
                x: mouth_x,
                y: mouth_y + 0.10,
                z: 0.0,
            };
        }
        // Index fingertip near mouth (the finger being bitten).
        landmarks[8] = Landmark {
            x: mouth_x + 0.01,
            y: mouth_y + 0.01,
            z: 0.0,
        };
        // Thumb tip near mouth too.
        landmarks[4] = Landmark {
            x: mouth_x - 0.01,
            y: mouth_y + 0.02,
            z: 0.0,
        };
        // Other fingertips curled back toward wrist/palm (middle, ring, pinky).
        for &idx in &[12, 16, 20] {
            landmarks[idx] = Landmark {
                x: mouth_x,
                y: mouth_y + 0.18,
                z: 0.0,
            };
        }
        // DIP joints for curled fingers (between MCP and curled tips).
        for &idx in &[3, 7, 11, 15, 19] {
            landmarks[idx] = Landmark {
                x: mouth_x,
                y: mouth_y + 0.14,
                z: 0.0,
            };
        }
        HandDetection {
            side: None,
            landmarks,
            confidence: 1.0,
        }
    }

    #[test]
    fn detects_nail_biting() {
        let detector = NailBitingDetector::new(&default_config(), true, true);
        let analysis = FrameAnalysis {
            timestamp: Instant::now(),
            camera_id: Arc::from("test"),
            hands: vec![make_curled_hand_near_mouth(0.5, 0.4)],
            face: Some(make_face_with_mouth(0.5, 0.4)),
            raw_frame: None,
        };

        let confidence = detector.analyze_frame(&analysis);
        assert!(confidence.is_some());
        assert!(
            confidence.unwrap() > 0.5,
            "confidence was {}",
            confidence.unwrap()
        );
    }

    #[test]
    fn no_detection_without_face() {
        let detector = NailBitingDetector::new(&default_config(), true, true);
        let analysis = FrameAnalysis {
            timestamp: Instant::now(),
            camera_id: Arc::from("test"),
            hands: vec![make_curled_hand_near_mouth(0.5, 0.4)],
            face: None,
            raw_frame: None,
        };

        assert!(detector.analyze_frame(&analysis).is_none());
    }

    #[test]
    fn no_detection_without_hands() {
        let detector = NailBitingDetector::new(&default_config(), true, true);
        let analysis = FrameAnalysis {
            timestamp: Instant::now(),
            camera_id: Arc::from("test"),
            hands: vec![],
            face: Some(make_face_with_mouth(0.5, 0.4)),
            raw_frame: None,
        };

        assert!(detector.analyze_frame(&analysis).is_none());
    }

    #[test]
    fn no_detection_when_hand_far_from_mouth() {
        let detector = NailBitingDetector::new(&default_config(), true, true);
        let mut hand = make_curled_hand_near_mouth(0.5, 0.4);
        for lm in &mut hand.landmarks {
            lm.y += 0.5;
        }
        let analysis = FrameAnalysis {
            timestamp: Instant::now(),
            camera_id: Arc::from("test"),
            hands: vec![hand],
            face: Some(make_face_with_mouth(0.5, 0.4)),
            raw_frame: None,
        };

        let confidence = detector.analyze_frame(&analysis).unwrap();
        assert!(confidence < 0.1, "confidence was {confidence}");
    }

    /// Regression test: chin rest suppression must NOT trigger when a
    /// fingertip is close to the mouth. Previously, the check ran before
    /// the fingertip distance check, suppressing valid nail biting poses
    /// where some fingers appeared "extended" toward the mouth.
    #[test]
    fn chin_rest_suppression_does_not_block_fingertip_at_mouth() {
        let detector = NailBitingDetector::new(&default_config(), true, true);

        // Create a hand where index finger is at the mouth, middle is
        // slightly extended (would trigger chin rest by extension ratio),
        // but ring and pinky are curled toward the palm.
        let mouth_x = 0.5;
        let mouth_y = 0.4;
        let mut landmarks = [Landmark { x: 0.0, y: 0.0, z: 0.0 }; 21];

        // Wrist below the face.
        landmarks[0] = Landmark { x: mouth_x, y: mouth_y + 0.25, z: 0.0 };
        // MCP joints (knuckles).
        for &idx in &[1, 5, 9, 13, 17] {
            landmarks[idx] = Landmark { x: mouth_x, y: mouth_y + 0.12, z: 0.0 };
        }
        // Index fingertip at mouth (the finger being bitten).
        landmarks[8] = Landmark { x: mouth_x, y: mouth_y + 0.01, z: 0.0 };
        // Thumb near mouth too.
        landmarks[4] = Landmark { x: mouth_x - 0.02, y: mouth_y + 0.03, z: 0.0 };
        // Middle finger extended (tip far from wrist, farther than MCP).
        landmarks[12] = Landmark { x: mouth_x + 0.05, y: mouth_y + 0.05, z: 0.0 };
        // Ring and pinky CURLED (tips closer to wrist than MCPs).
        landmarks[16] = Landmark { x: mouth_x, y: mouth_y + 0.20, z: 0.0 };
        landmarks[20] = Landmark { x: mouth_x, y: mouth_y + 0.22, z: 0.0 };
        // DIP joints: middle extended, ring/pinky closer to wrist.
        landmarks[3] = Landmark { x: mouth_x, y: mouth_y + 0.07, z: 0.0 };
        landmarks[7] = Landmark { x: mouth_x, y: mouth_y + 0.07, z: 0.0 };
        landmarks[11] = Landmark { x: mouth_x, y: mouth_y + 0.07, z: 0.0 };
        landmarks[15] = Landmark { x: mouth_x, y: mouth_y + 0.17, z: 0.0 };
        landmarks[19] = Landmark { x: mouth_x, y: mouth_y + 0.19, z: 0.0 };

        let hand = HandDetection {
            side: None,
            landmarks,
            confidence: 1.0,
        };

        let analysis = FrameAnalysis {
            timestamp: Instant::now(),
            camera_id: Arc::from("test"),
            hands: vec![hand],
            face: Some(make_face_with_mouth(mouth_x, mouth_y)),
            raw_frame: None,
        };

        let confidence = detector.analyze_frame(&analysis);
        assert!(confidence.is_some());
        assert!(
            confidence.unwrap() > 0.3,
            "Expected detection despite extended fingers, got {}",
            confidence.unwrap()
        );
    }
}
