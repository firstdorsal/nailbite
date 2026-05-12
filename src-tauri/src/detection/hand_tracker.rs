//! Simplified hand tracking across frames.
//!
//! Tracks up to 2 hands for a single person (the user).
//! Provides temporal consistency:
//! - Consistent left/right hand identity
//! - Smooth tracking when detection briefly drops
//! - Grace period to prevent flickering when detection momentarily drops
//! - Confirmation delay to prevent single-frame false positives
//! - Confidence hysteresis for stable hand presence
//! - Filters duplicate detections

#![allow(clippy::indexing_slicing)]

use std::time::{Duration, Instant};

use tracing::debug;

use crate::detection::types::{HandDetection, HandSide, Landmark, WRIST_INDEX};

/// How long to keep tracking after last detection (absolute timeout).
/// After this duration without any match, the hand slot is fully cleared.
const TRACKING_TIMEOUT: Duration = Duration::from_millis(1500);

/// Maximum distance (normalized) for matching a detection to a tracked hand.
const MAX_MATCH_DISTANCE: f32 = 0.35;

/// Minimum distance between two hands to consider them distinct.
const MIN_HAND_SEPARATION: f32 = 0.10;

/// Default smoothing factor for landmark updates (0 = no smoothing, 1 = no update).
/// Lower values = smoother/more stable but less responsive.
const DEFAULT_SMOOTHING_ALPHA: f32 = 0.3;

/// Default number of frames a hand persists without detection before going invisible.
/// At 8 FPS, 3 frames = ~375ms grace period.
const DEFAULT_GRACE_FRAMES: u8 = 3;

/// Default number of consecutive detection frames required before a new hand becomes visible.
/// At 8 FPS, 2 frames = ~250ms confirmation delay.
const DEFAULT_CONFIRMATION_FRAMES: u8 = 2;

/// Default confidence threshold for accepting a NEW hand (higher = more selective).
const DEFAULT_NEW_HAND_CONFIDENCE: f32 = 0.30;

/// Default confidence threshold for keeping an EXISTING tracked hand (lower = more stable).
const DEFAULT_EXISTING_HAND_CONFIDENCE: f32 = 0.15;

/// Configuration for hand tracking behavior.
#[derive(Debug, Clone)]
pub struct TrackingConfig {
    /// Smoothing factor for landmark EMA (0 = no smoothing, 1 = no update).
    pub smoothing_alpha: f32,
    /// Number of unmatched frames before a hand goes invisible.
    pub grace_frames: u8,
    /// Number of consecutive detections required before a new hand becomes visible.
    pub confirmation_frames: u8,
    /// Confidence threshold for accepting a new hand.
    pub new_hand_confidence: f32,
    /// Confidence threshold for keeping an existing tracked hand.
    pub existing_hand_confidence: f32,
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self {
            smoothing_alpha: DEFAULT_SMOOTHING_ALPHA,
            grace_frames: DEFAULT_GRACE_FRAMES,
            confirmation_frames: DEFAULT_CONFIRMATION_FRAMES,
            new_hand_confidence: DEFAULT_NEW_HAND_CONFIDENCE,
            existing_hand_confidence: DEFAULT_EXISTING_HAND_CONFIDENCE,
        }
    }
}

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
    /// Whether this hand is currently visible (confirmed and within grace).
    pub visible: bool,
    /// Number of consecutive frames this hand was NOT matched (for grace period).
    pub miss_count: u8,
    /// Number of consecutive frames this hand was detected (for confirmation delay).
    pub consecutive_detections: u8,
}

impl TrackedHand {
    fn new(side: HandSide, detection: HandDetection, confirmation_frames: u8, now: Instant) -> Self {
        // First detection counts as 1; visible immediately only if confirmation_frames <= 1
        let visible = confirmation_frames <= 1;
        Self {
            side,
            smoothed_landmarks: detection.landmarks,
            confidence: detection.confidence,
            last_seen: now,
            visible,
            miss_count: 0,
            consecutive_detections: 1,
        }
    }

    fn update(&mut self, detection: HandDetection, alpha: f32, confirmation_frames: u8, now: Instant) {
        let one_minus_alpha = 1.0 - alpha;
        // Smooth landmarks
        for (i, new_lm) in detection.landmarks.iter().enumerate() {
            let old = &mut self.smoothed_landmarks[i];
            old.x = alpha.mul_add(new_lm.x, one_minus_alpha * old.x);
            old.y = alpha.mul_add(new_lm.y, one_minus_alpha * old.y);
            old.z = alpha.mul_add(new_lm.z, one_minus_alpha * old.z);
        }
        // Smooth confidence to reduce flickering
        self.confidence = alpha.mul_add(detection.confidence, one_minus_alpha * self.confidence);
        self.last_seen = now;
        self.miss_count = 0;

        // Increment consecutive detections (saturating to avoid overflow)
        self.consecutive_detections = self.consecutive_detections.saturating_add(1);

        // Confirmation delay: only become visible after enough consecutive detections
        if self.consecutive_detections >= confirmation_frames {
            self.visible = true;
        }

        // If detection has a pose-corrected side that differs, log but don't change slot assignment
        // The slot (left/right in HandTracker) determines identity, not this field
        // This field is informational for the output
        if let Some(pose_side) = detection.side {
            if pose_side != self.side {
                debug!(
                    tracked_side = ?self.side,
                    detection_side = ?pose_side,
                    "Detection side differs from tracked slot (slot assignment stable)"
                );
            }
        }
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
    /// Tracking configuration.
    config: TrackingConfig,
}

impl HandTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a hand tracker with custom configuration.
    pub fn with_config(config: TrackingConfig) -> Self {
        Self {
            left: None,
            right: None,
            config,
        }
    }

    /// Update tracking with new detections.
    /// Returns smoothed hand detections.
    pub fn update(&mut self, detections: Vec<HandDetection>) -> Vec<HandDetection> {
        let now = Instant::now();
        let alpha = self.config.smoothing_alpha;
        let confirmation_frames = self.config.confirmation_frames;
        let grace_frames = self.config.grace_frames;
        let new_hand_confidence = self.config.new_hand_confidence;
        let existing_hand_confidence = self.config.existing_hand_confidence;

        // Remove expired hands (absolute timeout)
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

        // Filter detections below existing_hand_confidence early
        // (the lowest threshold we'll ever accept)
        sorted.retain(|d| d.confidence >= existing_hand_confidence);

        // Track which detections are used and which hands were matched
        let mut used = vec![false; sorted.len()];
        let mut left_matched = false;
        let mut right_matched = false;

        // Phase 1: Match detections to existing tracked hands
        // When a detection has a pose-corrected side, prefer matching to that side's slot.
        // This ensures pose estimation overrides position-based heuristics.
        for (i, det) in sorted.iter().enumerate() {
            if used[i] {
                continue;
            }

            // Calculate distance to both hands (only consider unmatched slots)
            let left_dist = self.left.as_ref()
                .filter(|_| !left_matched)
                .map(|h| h.distance_to(det));
            let right_dist = self.right.as_ref()
                .filter(|_| !right_matched)
                .map(|h| h.distance_to(det));

            // If detection has a pose-corrected side, prefer that side
            if let Some(pose_side) = det.side {
                match pose_side {
                    HandSide::Left => {
                        if let Some(ld) = left_dist {
                            if ld < MAX_MATCH_DISTANCE {
                                self.left.as_mut().unwrap().update(det.clone(), alpha, confirmation_frames, now);
                                used[i] = true;
                                left_matched = true;
                                debug!(side = "left", distance = ld, pose_corrected = true, "Matched to pose-preferred side");
                                continue;
                            }
                        }
                    }
                    HandSide::Right => {
                        if let Some(rd) = right_dist {
                            if rd < MAX_MATCH_DISTANCE {
                                self.right.as_mut().unwrap().update(det.clone(), alpha, confirmation_frames, now);
                                used[i] = true;
                                right_matched = true;
                                debug!(side = "right", distance = rd, pose_corrected = true, "Matched to pose-preferred side");
                                continue;
                            }
                        }
                    }
                }
                // Pose-corrected side slot not available or too far, try the other slot
                // but only if it's reasonably close (tighter threshold since it's a side mismatch)
                let cross_threshold = MAX_MATCH_DISTANCE * 0.5;
                match pose_side {
                    HandSide::Left => {
                        if let Some(rd) = right_dist {
                            if rd < cross_threshold {
                                debug!(
                                    pose_side = "left",
                                    matching_slot = "right",
                                    distance = rd,
                                    "Cross-matching due to slot unavailability"
                                );
                                self.right.as_mut().unwrap().update(det.clone(), alpha, confirmation_frames, now);
                                used[i] = true;
                                right_matched = true;
                            }
                        }
                    }
                    HandSide::Right => {
                        if let Some(ld) = left_dist {
                            if ld < cross_threshold {
                                debug!(
                                    pose_side = "right",
                                    matching_slot = "left",
                                    distance = ld,
                                    "Cross-matching due to slot unavailability"
                                );
                                self.left.as_mut().unwrap().update(det.clone(), alpha, confirmation_frames, now);
                                used[i] = true;
                                left_matched = true;
                            }
                        }
                    }
                }
                continue;
            }

            // No pose-corrected side - fall back to distance-based matching
            match (left_dist, right_dist) {
                (Some(ld), Some(rd)) => {
                    if ld <= rd && ld < MAX_MATCH_DISTANCE {
                        self.left.as_mut().unwrap().update(det.clone(), alpha, confirmation_frames, now);
                        used[i] = true;
                        left_matched = true;
                        debug!(side = "left", distance = ld, "Matched to closest hand");
                    } else if rd < ld && rd < MAX_MATCH_DISTANCE {
                        self.right.as_mut().unwrap().update(det.clone(), alpha, confirmation_frames, now);
                        used[i] = true;
                        right_matched = true;
                        debug!(side = "right", distance = rd, "Matched to closest hand");
                    }
                }
                (Some(ld), None) if ld < MAX_MATCH_DISTANCE => {
                    self.left.as_mut().unwrap().update(det.clone(), alpha, confirmation_frames, now);
                    used[i] = true;
                    left_matched = true;
                    debug!(side = "left", distance = ld, "Matched to left hand");
                }
                (None, Some(rd)) if rd < MAX_MATCH_DISTANCE => {
                    self.right.as_mut().unwrap().update(det.clone(), alpha, confirmation_frames, now);
                    used[i] = true;
                    right_matched = true;
                    debug!(side = "right", distance = rd, "Matched to right hand");
                }
                _ => {}
            }
        }

        // Grace period: handle unmatched hands
        // Instead of immediately going invisible, increment miss_count and keep
        // visibility during grace period, returning stale smoothed landmarks.
        if !left_matched {
            if let Some(ref mut h) = self.left {
                h.miss_count = h.miss_count.saturating_add(1);
                if h.miss_count > grace_frames {
                    h.visible = false;
                    h.consecutive_detections = 0;
                }
                // During grace period, visible stays as-is (preserving previous state)
            }
        }
        if !right_matched {
            if let Some(ref mut h) = self.right {
                h.miss_count = h.miss_count.saturating_add(1);
                if h.miss_count > grace_frames {
                    h.visible = false;
                    h.consecutive_detections = 0;
                }
            }
        }

        // Phase 2: Add unmatched detections as new hands
        // Apply higher confidence threshold for new hands (hysteresis)
        for (i, det) in sorted.iter().enumerate() {
            if used[i] {
                continue;
            }

            // New hands require higher confidence to prevent flickering
            if det.confidence < new_hand_confidence {
                debug!(
                    confidence = det.confidence,
                    threshold = new_hand_confidence,
                    "New hand rejected: confidence below new-hand threshold"
                );
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
                            "Added new hand (pending confirmation)"
                        );
                        self.left = Some(TrackedHand::new(side, det.clone(), confirmation_frames, now));
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
                            "Added new hand (pending confirmation)"
                        );
                        self.right = Some(TrackedHand::new(side, det.clone(), confirmation_frames, now));
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

    /// Get the tracking config (for telemetry).
    pub fn config(&self) -> &TrackingConfig {
        &self.config
    }

    /// Get miss count for left hand (for telemetry).
    pub fn left_miss_count(&self) -> Option<u8> {
        self.left.as_ref().map(|h| h.miss_count)
    }

    /// Get miss count for right hand (for telemetry).
    pub fn right_miss_count(&self) -> Option<u8> {
        self.right.as_ref().map(|h| h.miss_count)
    }

    /// Get consecutive detections for left hand (for telemetry).
    pub fn left_consecutive_detections(&self) -> Option<u8> {
        self.left.as_ref().map(|h| h.consecutive_detections)
    }

    /// Get consecutive detections for right hand (for telemetry).
    pub fn right_consecutive_detections(&self) -> Option<u8> {
        self.right.as_ref().map(|h| h.consecutive_detections)
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

    /// Helper: create a tracker with confirmation_frames=1 for legacy tests
    /// that expect immediate visibility (like the original behavior).
    fn immediate_tracker() -> HandTracker {
        HandTracker::with_config(TrackingConfig {
            confirmation_frames: 1,
            ..TrackingConfig::default()
        })
    }

    #[test]
    fn new_hand_not_immediately_visible() {
        let mut tracker = HandTracker::new(); // default: confirmation_frames=2

        let det = make_hand(0.3, 0.5, None, 0.9);
        let result = tracker.update(vec![det]);

        // First frame: hand should not yet be visible (needs confirmation)
        assert_eq!(result.len(), 0, "New hand should not be visible on first frame");
    }

    #[test]
    fn new_hand_visible_after_confirmation() {
        let mut tracker = HandTracker::new(); // default: confirmation_frames=2

        let det = make_hand(0.3, 0.5, None, 0.9);

        // Frame 1: not visible yet
        let result1 = tracker.update(vec![det.clone()]);
        assert_eq!(result1.len(), 0);

        // Frame 2: confirmed, now visible
        let result2 = tracker.update(vec![det]);
        assert_eq!(result2.len(), 1);
        assert!(result2[0].side.is_some());
    }

    #[test]
    fn tracks_single_hand() {
        let mut tracker = immediate_tracker();

        let det = make_hand(0.3, 0.5, None, 0.9);
        let result = tracker.update(vec![det]);

        assert_eq!(result.len(), 1);
        assert!(result[0].side.is_some());
    }

    #[test]
    fn tracks_two_hands() {
        let mut tracker = immediate_tracker();

        let left = make_hand(0.6, 0.5, None, 0.9);
        let right = make_hand(0.3, 0.5, None, 0.9);
        let result = tracker.update(vec![left.clone(), right.clone()]);

        assert_eq!(result.len(), 2);
        assert!(tracker.has_two_hands());
    }

    #[test]
    fn maintains_identity_across_frames() {
        let mut tracker = immediate_tracker();

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
        let mut tracker = immediate_tracker();

        let det1 = make_hand(0.5, 0.5, None, 0.9);
        let det2 = make_hand(0.51, 0.51, None, 0.8);
        let result = tracker.update(vec![det1, det2]);

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn accepts_both_hands_when_model_misclassifies() {
        let mut tracker = immediate_tracker();

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
        let mut tracker = immediate_tracker();

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
    fn hand_persists_during_grace_period() {
        let mut tracker = immediate_tracker(); // grace_frames=3

        // Frame 1: Hand detected and confirmed
        let det = make_hand(0.5, 0.5, Some(HandSide::Left), 0.9);
        let result1 = tracker.update(vec![det]);
        assert_eq!(result1.len(), 1);

        // Frames 2-4: No detection — hand should persist during grace period
        for frame in 2..=4 {
            let result = tracker.update(vec![]);
            assert_eq!(result.len(), 1, "Hand should persist during grace frame {frame}");
        }

        // Frame 5: No detection — grace period exceeded, hand should disappear
        let result5 = tracker.update(vec![]);
        assert_eq!(result5.len(), 0, "Hand should disappear after grace period");
    }

    #[test]
    fn revives_invisible_hand() {
        let mut tracker = immediate_tracker();

        // Frame 1: Hand detected
        let det1 = make_hand(0.5, 0.5, Some(HandSide::Left), 0.9);
        let result1 = tracker.update(vec![det1]);
        assert_eq!(result1.len(), 1);

        // Frames 2-5: No detection — exhaust grace period
        for _ in 0..4 {
            tracker.update(vec![]);
        }
        let result_gone = tracker.update(vec![]);
        assert_eq!(result_gone.len(), 0, "Hand should be gone after grace");

        // Frame 6: Hand back in same area — should match to existing slot
        let det3 = make_hand(0.52, 0.52, Some(HandSide::Right), 0.9);
        let result3 = tracker.update(vec![det3]);
        assert_eq!(result3.len(), 1);
        // Should keep original side
        assert_eq!(result3[0].side, result1[0].side);
    }

    #[test]
    fn hands_far_apart_still_tracked() {
        let mut tracker = immediate_tracker();

        // Hands very far apart (like arms spread wide)
        let left = make_hand(0.1, 0.5, None, 0.9);
        let right = make_hand(0.9, 0.5, None, 0.9);
        let result = tracker.update(vec![left, right]);

        assert_eq!(result.len(), 2, "Should track both hands even far apart");
        assert!(tracker.has_two_hands());
    }

    #[test]
    fn single_frame_false_positive_blocked() {
        let mut tracker = HandTracker::new(); // default: confirmation_frames=2

        // Frame 1: Hand appears
        let det = make_hand(0.5, 0.5, None, 0.9);
        let result1 = tracker.update(vec![det]);
        assert_eq!(result1.len(), 0, "Not confirmed yet");

        // Frame 2: Hand disappears — single frame spike
        let result2 = tracker.update(vec![]);
        assert_eq!(result2.len(), 0, "Single frame detection should not appear");
    }

    #[test]
    fn existing_hand_survives_low_confidence() {
        // Once a hand is confirmed, it should survive at lower confidence
        // (hysteresis: existing_hand_confidence=0.15 < new_hand_confidence=0.30)
        let mut tracker = immediate_tracker();

        // Frame 1: High confidence hand confirmed
        let det1 = make_hand(0.5, 0.5, None, 0.9);
        let result1 = tracker.update(vec![det1]);
        assert_eq!(result1.len(), 1);

        // Frame 2: Same hand at low confidence (0.20 — below new_hand but above existing_hand)
        let det2 = make_hand(0.51, 0.51, None, 0.20);
        let result2 = tracker.update(vec![det2]);
        assert_eq!(result2.len(), 1, "Existing hand should survive at low confidence");
    }

    #[test]
    fn new_hand_rejected_at_low_confidence() {
        let mut tracker = immediate_tracker();

        // Try to add a new hand at confidence below new_hand_confidence (0.30)
        let det = make_hand(0.5, 0.5, None, 0.20);
        let result = tracker.update(vec![det]);
        assert_eq!(result.len(), 0, "New hand should be rejected at low confidence");
    }

    #[test]
    fn grace_period_revives_with_confirmation_bypass() {
        // A hand that was previously confirmed should not need re-confirmation
        // after a brief gap within the grace period
        let mut tracker = immediate_tracker(); // grace_frames=3

        // Frame 1: Hand confirmed
        let det = make_hand(0.5, 0.5, None, 0.9);
        let result1 = tracker.update(vec![det.clone()]);
        assert_eq!(result1.len(), 1);

        // Frame 2: No detection (miss_count=1, still in grace)
        let result2 = tracker.update(vec![]);
        assert_eq!(result2.len(), 1, "Should persist in grace");

        // Frame 3: Hand back — should immediately be visible (already confirmed)
        let result3 = tracker.update(vec![make_hand(0.51, 0.51, None, 0.9)]);
        assert_eq!(result3.len(), 1, "Should be immediately visible after grace match");
    }
}
