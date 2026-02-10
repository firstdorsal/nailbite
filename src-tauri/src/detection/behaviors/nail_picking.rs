//! Nail picking behavior detector.
//!
//! Detects when fingertips of one hand are in close proximity to
//! fingertips/nails of the other hand, with a pinching gesture check.

use std::time::Duration;

use tracing::debug;

use crate::config::BehaviorConfig;
use crate::detection::analyzer::{
    is_pinching, is_typing_posture, landmark_distance_2d, min_inter_hand_fingertip_distance,
};
use crate::detection::behaviors::BehaviorDetector;
use crate::detection::types::{BfrbType, FrameAnalysis, HandDetection, WRIST_INDEX};

pub struct NailPickingDetector {
    proximity_threshold: f32,
    min_sustained_ms: u64,
    confidence_threshold: f32,
    typing_suppression: bool,
}

impl NailPickingDetector {
    pub fn new(config: &BehaviorConfig, typing_suppression: bool) -> Self {
        Self {
            proximity_threshold: config.proximity_threshold,
            min_sustained_ms: config.min_sustained_ms,
            confidence_threshold: config.confidence_threshold,
            typing_suppression,
        }
    }
}

impl BehaviorDetector for NailPickingDetector {
    fn bfrb_type(&self) -> BfrbType {
        BfrbType::NailPicking
    }

    fn name(&self) -> &str {
        "Nail Picking"
    }

    fn analyze_frame(&self, analysis: &FrameAnalysis) -> Option<f32> {
        // Require at least two hands.
        if analysis.hands.len() < 2 {
            debug!(hands = analysis.hands.len(), "NailPicking: need 2 hands");
            return None;
        }

        // Check typing suppression.
        if self.typing_suppression && is_typing_posture(&analysis.hands) {
            debug!("NailPicking: typing posture suppression");
            return Some(0.0);
        }

        let mut max_confidence = 0.0_f32;

        // Check all pairs of hands.
        for i in 0..analysis.hands.len() {
            for j in (i + 1)..analysis.hands.len() {
                let Some(hand1) = analysis.hands.get(i) else {
                    continue;
                };
                let Some(hand2) = analysis.hands.get(j) else {
                    continue;
                };

                // Compute hand scale: average wrist-to-middle-MCP distance.
                let scale = hand_scale(hand1, hand2);
                if scale <= 0.0 {
                    continue;
                }

                // Find minimum inter-hand fingertip distance.
                let Some((dist, _tip1, _tip2)) =
                    min_inter_hand_fingertip_distance(hand1, hand2)
                else {
                    continue;
                };

                let normalized_dist = dist / scale;
                if normalized_dist > self.proximity_threshold {
                    continue;
                }

                // Bonus confidence if one hand is pinching.
                let pinch_bonus = if is_pinching(hand1) || is_pinching(hand2) {
                    0.2
                } else {
                    0.0
                };

                let proximity_score = 1.0 - (normalized_dist / self.proximity_threshold);
                let confidence = (proximity_score + pinch_bonus).min(1.0);

                debug!(
                    raw_dist = dist,
                    normalized_dist = normalized_dist,
                    threshold = self.proximity_threshold,
                    proximity_score = proximity_score,
                    pinch_bonus = pinch_bonus,
                    confidence = confidence,
                    "NailPicking: hand pair analysis"
                );

                max_confidence = max_confidence.max(confidence);
            }
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
        false
    }

    fn requires_hands(&self) -> bool {
        true
    }
}

/// Average hand scale from two hands (wrist to middle finger MCP distance).
fn hand_scale(hand1: &HandDetection, hand2: &HandDetection) -> f32 {
    let scale1 = single_hand_scale(hand1);
    let scale2 = single_hand_scale(hand2);

    match (scale1, scale2) {
        (Some(s1), Some(s2)) => (s1 + s2) / 2.0,
        (Some(s), None) | (None, Some(s)) => s,
        (None, None) => 0.0,
    }
}

fn single_hand_scale(hand: &HandDetection) -> Option<f32> {
    let wrist = hand.landmarks.get(WRIST_INDEX)?;
    let middle_mcp = hand.landmarks.get(9)?;
    let dist = landmark_distance_2d(wrist, middle_mcp);
    if dist > 0.0 {
        Some(dist)
    } else {
        None
    }
}

#[cfg(test)]
#[allow(clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::detection::types::Landmark;
    use std::sync::Arc;
    use std::time::Instant;

    fn default_config() -> BehaviorConfig {
        BehaviorConfig {
            enabled: true,
            proximity_threshold: 0.15,
            min_sustained_ms: 1500,
            confidence_threshold: 0.3,
        }
    }

    fn make_hand_at(x: f32, y: f32) -> HandDetection {
        let mut landmarks = [Landmark {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }; 21];
        // Wrist
        landmarks[0] = Landmark {
            x,
            y: y + 0.2,
            z: 0.0,
        };
        // Middle MCP (for scale)
        landmarks[9] = Landmark { x, y, z: 0.0 };
        // Set fingertips near center
        for &idx in &[4, 8, 12, 16, 20] {
            landmarks[idx] = Landmark {
                x,
                y: y - 0.05,
                z: 0.0,
            };
        }
        // MCP joints
        for &idx in &[1, 5, 13, 17] {
            landmarks[idx] = Landmark {
                x,
                y: y + 0.05,
                z: 0.0,
            };
        }
        // DIP joints: set between MCP and wrist (semi-curled) to avoid
        // triggering typing suppression (which checks finger extension ratio).
        for &idx in &[3, 7, 11, 15, 19] {
            landmarks[idx] = Landmark {
                x,
                y: y + 0.12,
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
    fn detects_nail_picking_when_hands_close() {
        let detector = NailPickingDetector::new(&default_config(), true);
        let hand1 = make_hand_at(0.5, 0.5);
        let hand2 = make_hand_at(0.501, 0.5);

        let analysis = FrameAnalysis {
            timestamp: Instant::now(),
            camera_id: Arc::from("test"),
            hands: vec![hand1, hand2],
            face: None,
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
    fn no_detection_with_one_hand() {
        let detector = NailPickingDetector::new(&default_config(), true);
        let analysis = FrameAnalysis {
            timestamp: Instant::now(),
            camera_id: Arc::from("test"),
            hands: vec![make_hand_at(0.5, 0.5)],
            face: None,
            raw_frame: None,
        };

        assert!(detector.analyze_frame(&analysis).is_none());
    }

    #[test]
    fn no_detection_when_hands_far_apart() {
        let detector = NailPickingDetector::new(&default_config(), true);
        let hand1 = make_hand_at(0.2, 0.5);
        let hand2 = make_hand_at(0.8, 0.5);

        let analysis = FrameAnalysis {
            timestamp: Instant::now(),
            camera_id: Arc::from("test"),
            hands: vec![hand1, hand2],
            face: None,
            raw_frame: None,
        };

        let confidence = detector.analyze_frame(&analysis).unwrap();
        assert!(confidence < 0.1, "confidence was {confidence}");
    }
}
