//! Nail picking behavior detector.
//!
//! Detects nail picking in two forms:
//! 1. Inter-hand: fingertips of one hand near fingertips/nails of the other hand
//! 2. Intra-hand (same-hand): thumb picking at other fingernails on the same hand

use std::time::Duration;

use tracing::debug;

use crate::config::BehaviorConfig;
use crate::detection::analyzer::{
    is_pinching, is_typing_posture, landmark_distance_2d, min_inter_hand_fingertip_distance,
};
use crate::detection::behaviors::BehaviorDetector;
use crate::detection::types::{BfrbType, FrameAnalysis, HandDetection, FINGERTIP_INDICES, WRIST_INDEX};

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

    /// Check for same-hand nail picking (thumb picking at other fingernails).
    ///
    /// Detects when thumb tip is very close to another fingertip on the same hand,
    /// indicating a picking motion.
    fn check_same_hand_picking(&self, hand: &HandDetection) -> Option<f32> {
        // Get thumb tip (index 4)
        let thumb_tip = hand.landmarks.get(4)?;

        // Get hand scale for normalization
        let scale = single_hand_scale(hand)?;
        if scale <= 0.0 {
            return None;
        }

        // Same-hand picking threshold is tighter since fingers are naturally closer
        // Use half the inter-hand threshold
        let same_hand_threshold = self.proximity_threshold * 0.5;

        let mut max_confidence = 0.0_f32;

        // Check distance from thumb tip to other fingertips (index=8, middle=12, ring=16, pinky=20)
        for &tip_idx in &FINGERTIP_INDICES[1..] {
            // Skip thumb itself
            let Some(other_tip) = hand.landmarks.get(tip_idx) else {
                continue;
            };

            let dist = landmark_distance_2d(thumb_tip, other_tip);
            let normalized_dist = dist / scale;

            if normalized_dist < same_hand_threshold {
                // Additionally check that the hand is in a picking posture:
                // - The target finger should be somewhat extended (not fully curled)
                // - This distinguishes picking from a normal closed fist
                if self.is_picking_posture(hand, tip_idx) {
                    let proximity_score = 1.0 - (normalized_dist / same_hand_threshold);
                    let confidence = proximity_score.min(1.0);

                    debug!(
                        thumb_to_finger = tip_idx,
                        raw_dist = dist,
                        normalized_dist = normalized_dist,
                        threshold = same_hand_threshold,
                        confidence = confidence,
                        "NailPicking: same-hand thumb-to-finger"
                    );

                    max_confidence = max_confidence.max(confidence);
                }
            }
        }

        if max_confidence > 0.0 {
            Some(max_confidence)
        } else {
            None
        }
    }

    /// Check if the hand is in a picking posture for same-hand detection.
    ///
    /// The target finger should be partially extended (not fully curled into palm),
    /// and thumb should be approaching from the side/top.
    fn is_picking_posture(&self, hand: &HandDetection, target_tip_idx: usize) -> bool {
        let Some(wrist) = hand.landmarks.get(WRIST_INDEX) else {
            return false;
        };
        let Some(target_tip) = hand.landmarks.get(target_tip_idx) else {
            return false;
        };

        // Get the MCP (knuckle) for the target finger
        let mcp_idx = match target_tip_idx {
            8 => 5,   // Index finger MCP
            12 => 9,  // Middle finger MCP
            16 => 13, // Ring finger MCP
            20 => 17, // Pinky MCP
            _ => return false,
        };

        let Some(mcp) = hand.landmarks.get(mcp_idx) else {
            return false;
        };

        // The target finger should be at least partially extended:
        // tip should be farther from wrist than MCP, or at similar distance
        let tip_to_wrist = landmark_distance_2d(target_tip, wrist);
        let mcp_to_wrist = landmark_distance_2d(mcp, wrist);

        // Allow if tip is at least 70% as far as MCP (partially extended or fully extended)
        tip_to_wrist >= mcp_to_wrist * 0.7
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
        // Need at least one hand
        if analysis.hands.is_empty() {
            debug!(hands = 0, "NailPicking: need at least 1 hand");
            return None;
        }

        // Check typing suppression (only applies when 2+ hands visible).
        if analysis.hands.len() >= 2
            && self.typing_suppression
            && is_typing_posture(&analysis.hands)
        {
            debug!("NailPicking: typing posture suppression");
            return Some(0.0);
        }

        let mut max_confidence = 0.0_f32;

        // 1. Check SAME-HAND picking (thumb picking at other fingernails)
        for hand in &analysis.hands {
            if let Some(conf) = self.check_same_hand_picking(hand) {
                debug!(
                    confidence = conf,
                    "NailPicking: same-hand picking detected"
                );
                max_confidence = max_confidence.max(conf);
            }
        }

        // 2. Check INTER-HAND picking (fingertips of different hands close together)
        if analysis.hands.len() >= 2 {
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
                        "NailPicking: inter-hand analysis"
                    );

                    max_confidence = max_confidence.max(confidence);
                }
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

    /// Create a hand with fingers spread apart (no same-hand picking).
    fn make_spread_hand_at(x: f32, y: f32) -> HandDetection {
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
        // Spread fingertips horizontally so thumb is far from other fingers
        // Thumb tip far to the left
        landmarks[4] = Landmark {
            x: x - 0.15,
            y: y - 0.05,
            z: 0.0,
        };
        // Other fingertips spread to the right
        landmarks[8] = Landmark {
            x: x + 0.02,
            y: y - 0.08,
            z: 0.0,
        };
        landmarks[12] = Landmark {
            x: x + 0.04,
            y: y - 0.09,
            z: 0.0,
        };
        landmarks[16] = Landmark {
            x: x + 0.06,
            y: y - 0.08,
            z: 0.0,
        };
        landmarks[20] = Landmark {
            x: x + 0.08,
            y: y - 0.06,
            z: 0.0,
        };
        // MCP joints
        for &idx in &[1, 5, 13, 17] {
            landmarks[idx] = Landmark {
                x,
                y: y + 0.05,
                z: 0.0,
            };
        }
        // DIP joints
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
    fn no_detection_with_no_hands() {
        let detector = NailPickingDetector::new(&default_config(), true);
        let analysis = FrameAnalysis {
            timestamp: Instant::now(),
            camera_id: Arc::from("test"),
            hands: vec![],
            face: None,
            raw_frame: None,
        };

        assert!(detector.analyze_frame(&analysis).is_none());
    }

    #[test]
    fn detects_same_hand_picking() {
        let detector = NailPickingDetector::new(&default_config(), true);

        // Create a hand with thumb tip close to index fingertip (picking posture)
        let mut hand = make_hand_at(0.5, 0.5);
        // Position thumb tip (4) very close to index tip (8)
        hand.landmarks[4] = Landmark {
            x: 0.5,
            y: 0.44, // Very close to index tip
            z: 0.0,
        };
        hand.landmarks[8] = Landmark {
            x: 0.5,
            y: 0.45,
            z: 0.0,
        };
        // Make sure index finger is extended (tip farther from wrist than MCP)
        hand.landmarks[5] = Landmark {
            x: 0.5,
            y: 0.55, // Index MCP closer to wrist
            z: 0.0,
        };

        let analysis = FrameAnalysis {
            timestamp: Instant::now(),
            camera_id: Arc::from("test"),
            hands: vec![hand],
            face: None,
            raw_frame: None,
        };

        let confidence = detector.analyze_frame(&analysis);
        assert!(confidence.is_some());
        assert!(
            confidence.unwrap() > 0.3,
            "same-hand picking confidence was {}",
            confidence.unwrap()
        );
    }

    #[test]
    fn no_detection_when_hands_far_apart() {
        let detector = NailPickingDetector::new(&default_config(), true);
        // Use spread hands so same-hand picking isn't triggered
        let hand1 = make_spread_hand_at(0.2, 0.5);
        let hand2 = make_spread_hand_at(0.8, 0.5);

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
