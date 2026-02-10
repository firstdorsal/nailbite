//! Simplified hand tracking across frames.
//!
//! Tracks up to 2 hands for a single person (the user).
//! Provides temporal consistency:
//! - Consistent left/right hand identity
//! - Smooth tracking when detection briefly drops
//! - Filters duplicate detections

#![allow(clippy::indexing_slicing)]

use std::time::{Duration, Instant};

use tracing::debug;

use crate::detection::types::{HandDetection, HandSide, Landmark, WRIST_INDEX};

/// How long to keep tracking after last detection (grace period).
const TRACKING_TIMEOUT: Duration = Duration::from_millis(1000);

/// Maximum distance (normalized) for matching a detection to a tracked hand.
const MAX_MATCH_DISTANCE: f32 = 0.35;

/// Minimum distance between two hands to consider them distinct.
const MIN_HAND_SEPARATION: f32 = 0.10;

/// Smoothing factor for landmark updates (0 = no smoothing, 1 = no update).
const SMOOTHING_ALPHA: f32 = 0.4;

/// A tracked hand.
#[derive(Debug, Clone)]
pub struct TrackedHand {
    /// Assigned side (left or right).
    pub side: HandSide,
    /// Smoothed landmarks (all 21 points).
    pub smoothed_landmarks: [Landmark; 21],
    /// Last raw detection confidence.
    pub confidence: f32,
    /// Last time this hand was detected.
    pub last_seen: Instant,
    /// Whether this hand is currently visible.
    pub visible: bool,
}

impl TrackedHand {
    fn new(side: HandSide, detection: HandDetection, now: Instant) -> Self {
        Self {
            side,
            smoothed_landmarks: detection.landmarks,
            confidence: detection.confidence,
            last_seen: now,
            visible: true,
        }
    }

    fn update(&mut self, detection: HandDetection, now: Instant) {
        // Smooth landmarks
        for (i, new_lm) in detection.landmarks.iter().enumerate() {
            let old = &mut self.smoothed_landmarks[i];
            old.x = SMOOTHING_ALPHA * new_lm.x + (1.0 - SMOOTHING_ALPHA) * old.x;
            old.y = SMOOTHING_ALPHA * new_lm.y + (1.0 - SMOOTHING_ALPHA) * old.y;
            old.z = SMOOTHING_ALPHA * new_lm.z + (1.0 - SMOOTHING_ALPHA) * old.z;
        }
        self.confidence = detection.confidence;
        self.last_seen = now;
        self.visible = true;
    }

    fn wrist(&self) -> &Landmark {
        &self.smoothed_landmarks[WRIST_INDEX]
    }

    fn distance_to(&self, detection: &HandDetection) -> f32 {
        let wrist = &detection.landmarks[WRIST_INDEX];
        let tracked = self.wrist();
        let dx = wrist.x - tracked.x;
        let dy = wrist.y - tracked.y;
        (dx * dx + dy * dy).sqrt()
    }

    fn is_expired(&self, now: Instant) -> bool {
        now.duration_since(self.last_seen) > TRACKING_TIMEOUT
    }

    fn to_detection(&self) -> HandDetection {
        HandDetection {
            side: Some(self.side),
            landmarks: self.smoothed_landmarks,
            confidence: self.confidence,
        }
    }
}

/// Simple hand tracker for a single person.
#[derive(Debug, Default)]
pub struct HandTracker {
    /// Left hand (if tracked).
    left: Option<TrackedHand>,
    /// Right hand (if tracked).
    right: Option<TrackedHand>,
}

impl HandTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update tracking with new detections.
    /// Returns smoothed hand detections.
    pub fn update(&mut self, detections: Vec<HandDetection>) -> Vec<HandDetection> {
        let now = Instant::now();

        // Mark hands as not visible
        if let Some(ref mut h) = self.left {
            h.visible = false;
        }
        if let Some(ref mut h) = self.right {
            h.visible = false;
        }

        // Remove expired hands
        if self.left.as_ref().is_some_and(|h| h.is_expired(now)) {
            debug!("Left hand expired");
            self.left = None;
        }
        if self.right.as_ref().is_some_and(|h| h.is_expired(now)) {
            debug!("Right hand expired");
            self.right = None;
        }

        // Sort by confidence
        let mut sorted = detections;
        sorted.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Track which detections are used
        let mut used = vec![false; sorted.len()];

        // Phase 1: Match detections to existing tracked hands
        // Find the BEST match for each detection (closest hand)
        for (i, det) in sorted.iter().enumerate() {
            if used[i] {
                continue;
            }

            // Calculate distance to both hands, find the closest
            let left_dist = self.left.as_ref()
                .filter(|h| !h.visible)
                .map(|h| h.distance_to(det));
            let right_dist = self.right.as_ref()
                .filter(|h| !h.visible)
                .map(|h| h.distance_to(det));

            // Match to the closest hand if within threshold
            match (left_dist, right_dist) {
                (Some(ld), Some(rd)) => {
                    if ld <= rd && ld < MAX_MATCH_DISTANCE {
                        self.left.as_mut().unwrap().update(det.clone(), now);
                        used[i] = true;
                        debug!(side = "left", distance = ld, "Matched to closest hand");
                    } else if rd < ld && rd < MAX_MATCH_DISTANCE {
                        self.right.as_mut().unwrap().update(det.clone(), now);
                        used[i] = true;
                        debug!(side = "right", distance = rd, "Matched to closest hand");
                    }
                }
                (Some(ld), None) if ld < MAX_MATCH_DISTANCE => {
                    self.left.as_mut().unwrap().update(det.clone(), now);
                    used[i] = true;
                    debug!(side = "left", distance = ld, "Matched to left hand");
                }
                (None, Some(rd)) if rd < MAX_MATCH_DISTANCE => {
                    self.right.as_mut().unwrap().update(det.clone(), now);
                    used[i] = true;
                    debug!(side = "right", distance = rd, "Matched to right hand");
                }
                _ => {}
            }
        }

        // Phase 2: Add unmatched detections as new hands
        for (i, det) in sorted.iter().enumerate() {
            if used[i] {
                continue;
            }

            let wrist = &det.landmarks[WRIST_INDEX];

            // Check if too close to an existing visible hand (duplicate)
            let too_close = [&self.left, &self.right].iter().any(|h| {
                h.as_ref().is_some_and(|tracked| {
                    if !tracked.visible {
                        return false;
                    }
                    let dx = wrist.x - tracked.wrist().x;
                    let dy = wrist.y - tracked.wrist().y;
                    (dx * dx + dy * dy).sqrt() < MIN_HAND_SEPARATION
                })
            });

            if too_close {
                debug!(wrist_x = wrist.x, "Detection too close to existing hand");
                continue;
            }

            // Determine which slot to use for this hand.
            // Priority: detection's side > opposite of occupied slot > x-position heuristic.
            let preferred_side = det.side.unwrap_or(
                // Use x-position heuristic (camera mirror: left side of screen = right hand)
                if wrist.x < 0.5 {
                    HandSide::Right
                } else {
                    HandSide::Left
                },
            );

            // Check if preferred slot is available
            let left_available = self.left.as_ref().is_none_or(|h| !h.visible);
            let right_available = self.right.as_ref().is_none_or(|h| !h.visible);

            let side = match preferred_side {
                HandSide::Left if left_available => HandSide::Left,
                HandSide::Right if right_available => HandSide::Right,
                // Preferred slot occupied; try the other slot if this is clearly a different hand
                HandSide::Left if right_available => HandSide::Right,
                HandSide::Right if left_available => HandSide::Left,
                // Both slots occupied, keep preferred (will fail `can_add` check below)
                _ => preferred_side,
            };

            // Add the hand - can add if:
            // 1. Slot is empty, OR
            // 2. Slot has invisible hand that is far from this detection (wouldn't match anyway)
            match side {
                HandSide::Left => {
                    let can_add = self.left.as_ref().is_none_or(|h| {
                        !h.visible && h.distance_to(det) > MAX_MATCH_DISTANCE
                    });

                    if can_add {
                        debug!(
                            side = "left",
                            wrist_x = wrist.x,
                            confidence = det.confidence,
                            "Added new hand"
                        );
                        self.left = Some(TrackedHand::new(side, det.clone(), now));
                        used[i] = true;
                    }
                }
                HandSide::Right => {
                    let can_add = self.right.as_ref().is_none_or(|h| {
                        !h.visible && h.distance_to(det) > MAX_MATCH_DISTANCE
                    });

                    if can_add {
                        debug!(
                            side = "right",
                            wrist_x = wrist.x,
                            confidence = det.confidence,
                            "Added new hand"
                        );
                        self.right = Some(TrackedHand::new(side, det.clone(), now));
                        used[i] = true;
                    }
                }
            }
        }

        // Return visible hands
        let mut result = Vec::with_capacity(2);
        if let Some(ref h) = self.left {
            if h.visible {
                result.push(h.to_detection());
            }
        }
        if let Some(ref h) = self.right {
            if h.visible {
                result.push(h.to_detection());
            }
        }

        result
    }

    /// Get all visible hands.
    pub fn hands(&self) -> Vec<HandDetection> {
        let mut result = Vec::with_capacity(2);
        if let Some(ref h) = self.left {
            if h.visible {
                result.push(h.to_detection());
            }
        }
        if let Some(ref h) = self.right {
            if h.visible {
                result.push(h.to_detection());
            }
        }
        result
    }

    /// Whether we have two visible hands.
    pub fn has_two_hands(&self) -> bool {
        self.left.as_ref().is_some_and(|h| h.visible)
            && self.right.as_ref().is_some_and(|h| h.visible)
    }

    /// Clear all tracking state.
    pub fn clear(&mut self) {
        self.left = None;
        self.right = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hand(x: f32, y: f32, side: Option<HandSide>, confidence: f32) -> HandDetection {
        let mut landmarks = [Landmark { x: 0.0, y: 0.0, z: 0.0 }; 21];
        landmarks[WRIST_INDEX] = Landmark { x, y, z: 0.0 };
        for (i, lm) in landmarks.iter_mut().enumerate() {
            if i != WRIST_INDEX {
                lm.x = x + (i as f32 * 0.01);
                lm.y = y - (i as f32 * 0.01);
            }
        }
        HandDetection {
            side,
            landmarks,
            confidence,
        }
    }

    #[test]
    fn tracks_single_hand() {
        let mut tracker = HandTracker::new();

        let det = make_hand(0.3, 0.5, None, 0.9);
        let result = tracker.update(vec![det]);

        assert_eq!(result.len(), 1);
        assert!(result[0].side.is_some());
    }

    #[test]
    fn tracks_two_hands() {
        let mut tracker = HandTracker::new();

        let left = make_hand(0.6, 0.5, None, 0.9);
        let right = make_hand(0.3, 0.5, None, 0.9);
        let result = tracker.update(vec![left, right]);

        assert_eq!(result.len(), 2);
        assert!(tracker.has_two_hands());
    }

    #[test]
    fn maintains_identity_across_frames() {
        let mut tracker = HandTracker::new();

        // Frame 1
        let det1 = make_hand(0.3, 0.5, None, 0.9);
        let result1 = tracker.update(vec![det1]);
        let first_side = result1[0].side;

        // Frame 2 - slightly moved
        let det2 = make_hand(0.32, 0.52, None, 0.9);
        let result2 = tracker.update(vec![det2]);

        assert_eq!(result2[0].side, first_side);
    }

    #[test]
    fn rejects_duplicate_same_position() {
        let mut tracker = HandTracker::new();

        let det1 = make_hand(0.5, 0.5, None, 0.9);
        let det2 = make_hand(0.51, 0.51, None, 0.8);
        let result = tracker.update(vec![det1, det2]);

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn accepts_both_hands_when_model_misclassifies() {
        let mut tracker = HandTracker::new();

        // Model says both are Left, but they're clearly different hands
        let hand1 = make_hand(0.6, 0.5, Some(HandSide::Left), 0.9);
        let hand2 = make_hand(0.3, 0.5, Some(HandSide::Left), 0.85);
        let result = tracker.update(vec![hand1, hand2]);

        assert_eq!(result.len(), 2, "Should have 2 hands");
        assert!(tracker.has_two_hands());

        let sides: Vec<_> = result.iter().map(|h| h.side).collect();
        assert!(sides.contains(&Some(HandSide::Left)));
        assert!(sides.contains(&Some(HandSide::Right)));
    }

    #[test]
    fn maintains_side_when_model_flips() {
        let mut tracker = HandTracker::new();

        // Frame 1: Hand detected
        let det1 = make_hand(0.4, 0.5, Some(HandSide::Right), 0.9);
        let result1 = tracker.update(vec![det1]);
        let initial_side = result1[0].side;

        // Frame 2: Same hand, model says Left now
        let det2 = make_hand(0.42, 0.52, Some(HandSide::Left), 0.9);
        let result2 = tracker.update(vec![det2]);

        assert_eq!(result2[0].side, initial_side, "Side should remain stable");
    }

    #[test]
    fn revives_invisible_hand() {
        let mut tracker = HandTracker::new();

        // Frame 1: Hand detected
        let det1 = make_hand(0.5, 0.5, Some(HandSide::Left), 0.9);
        let result1 = tracker.update(vec![det1]);
        assert_eq!(result1.len(), 1);

        // Frame 2: No detection
        let result2 = tracker.update(vec![]);
        assert_eq!(result2.len(), 0);

        // Frame 3: Hand back
        let det3 = make_hand(0.52, 0.52, Some(HandSide::Right), 0.9);
        let result3 = tracker.update(vec![det3]);
        assert_eq!(result3.len(), 1);
        // Should keep original side
        assert_eq!(result3[0].side, result1[0].side);
    }

    #[test]
    fn hands_far_apart_still_tracked() {
        let mut tracker = HandTracker::new();

        // Hands very far apart (like arms spread wide)
        let left = make_hand(0.1, 0.5, None, 0.9);
        let right = make_hand(0.9, 0.5, None, 0.9);
        let result = tracker.update(vec![left, right]);

        assert_eq!(result.len(), 2, "Should track both hands even far apart");
        assert!(tracker.has_two_hands());
    }
}
