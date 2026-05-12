//! Temporal tracking state machine for behavior detection.
//!
//! Uses a sliding window of per-frame confidence values to confirm
//! sustained behavior before triggering alerts. Each BFRB type gets
//! its own tracker instance.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use tracing::{debug, info};

use crate::detection::types::{BfrbType, DetectionEvent, DetectionExplanation};

/// Phase of the temporal tracking state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackingPhase {
    /// No detection.
    Idle,
    /// Detection started, accumulating evidence.
    Accumulating,
    /// Sustained detection confirmed, alert should be triggered.
    Confirmed,
    /// Alert is active, waiting for exercise completion or dismissal.
    Alerting,
    /// Cooldown after exercise to prevent immediate re-trigger.
    Cooldown,
}

/// A single confidence sample with its timestamp.
#[derive(Debug, Clone, Copy)]
struct ConfidenceSample {
    timestamp: Instant,
    confidence: f32,
}

/// Temporal tracker for a single behavior type.
///
/// Maintains a sliding window of confidence samples and transitions
/// through tracking phases based on the positive ratio threshold.
pub struct BehaviorTracker {
    bfrb_type: BfrbType,
    phase: TrackingPhase,
    /// Sliding window of recent confidence samples.
    window: VecDeque<ConfidenceSample>,
    /// Duration of the sliding window.
    window_duration: Duration,
    /// Fraction of samples that must exceed the confidence threshold.
    positive_ratio: f32,
    /// Per-behavior confidence threshold.
    confidence_threshold: f32,
    /// Cooldown duration after alert is dismissed/completed.
    cooldown_duration: Duration,
    /// When the current phase started.
    phase_started_at: Instant,
    /// Camera ID for the most recent detection (for event construction).
    last_camera_id: String,
    /// The highest confidence seen during the current accumulation period.
    peak_confidence: f32,
    /// Explanation captured at the peak-confidence frame, used to explain
    /// the alert to the user. None until a positive sample has been seen.
    peak_explanation: Option<DetectionExplanation>,
    /// Timestamp of last info-level log during Alerting (rate-limited to 1/sec).
    last_alerting_log: Instant,
}

impl BehaviorTracker {
    pub fn new(
        bfrb_type: BfrbType,
        window_duration: Duration,
        positive_ratio: f32,
        confidence_threshold: f32,
        cooldown_duration: Duration,
    ) -> Self {
        Self {
            bfrb_type,
            phase: TrackingPhase::Idle,
            window: VecDeque::new(),
            window_duration,
            positive_ratio,
            confidence_threshold,
            cooldown_duration,
            phase_started_at: Instant::now(),
            last_camera_id: String::new(),
            peak_confidence: 0.0,
            peak_explanation: None,
            last_alerting_log: Instant::now(),
        }
    }

    /// Current tracking phase.
    pub fn phase(&self) -> TrackingPhase {
        self.phase
    }

    /// The BFRB type this tracker handles.
    pub fn bfrb_type(&self) -> BfrbType {
        self.bfrb_type
    }

    /// Feed a new confidence value from a frame analysis.
    ///
    /// Pass `Some(confidence)` when the detector had enough data to produce
    /// a result. Pass `None` when the detector had insufficient data (e.g.,
    /// no hand detected). `None` values are skipped entirely and do not
    /// count as negative samples — "no data" is not the same as "no behavior".
    ///
    /// Returns `Some(DetectionEvent)` if the tracker just transitioned
    /// to the `Confirmed` phase (first confirmation only, not repeated).
    pub fn update(
        &mut self,
        confidence: Option<f32>,
        timestamp: Instant,
        camera_id: &str,
    ) -> Option<DetectionEvent> {
        self.update_with_explanation(confidence, None, timestamp, camera_id)
    }

    /// Like `update`, but also records the per-frame explanation used to
    /// produce `confidence`. The tracker keeps the explanation from the
    /// peak-confidence sample so the resulting `DetectionEvent` can be
    /// rendered with full context.
    pub fn update_with_explanation(
        &mut self,
        confidence: Option<f32>,
        explanation: Option<DetectionExplanation>,
        timestamp: Instant,
        camera_id: &str,
    ) -> Option<DetectionEvent> {
        // Only clone camera_id if it changed (TECH-14)
        if self.last_camera_id != camera_id {
            self.last_camera_id = camera_id.to_string();
        }

        // Prune old samples outside the sliding window.
        self.prune_window(timestamp);

        // Only add a sample when the detector had enough data.
        // None = no data (e.g., hand not detected) → skip entirely.
        if let Some(conf) = confidence {
            self.window.push_back(ConfidenceSample {
                timestamp,
                confidence: conf,
            });
        }

        // Handle cooldown expiry even without new samples.
        if self.phase == TrackingPhase::Cooldown {
            let elapsed = timestamp.duration_since(self.phase_started_at);
            if elapsed >= self.cooldown_duration {
                self.transition_to(TrackingPhase::Idle, timestamp);
            }
            return None;
        }

        // Compute the positive ratio within the window.
        let current_ratio = self.compute_positive_ratio();
        let window_mature = self.is_window_mature();

        debug!(
            bfrb = %self.bfrb_type,
            phase = ?self.phase,
            confidence = ?confidence,
            positive_ratio = current_ratio,
            required_ratio = self.positive_ratio,
            conf_threshold = self.confidence_threshold,
            window_size = self.window.len(),
            mature = window_mature,
            "Tracker update"
        );

        match self.phase {
            TrackingPhase::Idle => {
                let conf = confidence.unwrap_or(0.0);
                if window_mature && current_ratio >= self.positive_ratio {
                    self.transition_to(TrackingPhase::Confirmed, timestamp);
                    Some(self.make_event(timestamp))
                } else if conf >= self.confidence_threshold {
                    self.transition_to(TrackingPhase::Accumulating, timestamp);
                    self.peak_confidence = conf;
                    self.peak_explanation = explanation.clone();
                    None
                } else {
                    None
                }
            }
            TrackingPhase::Accumulating => {
                let conf = confidence.unwrap_or(0.0);
                if conf >= self.confidence_threshold && conf >= self.peak_confidence {
                    self.peak_confidence = conf;
                    if explanation.is_some() {
                        self.peak_explanation = explanation.clone();
                    }
                }
                if window_mature && current_ratio >= self.positive_ratio {
                    self.transition_to(TrackingPhase::Confirmed, timestamp);
                    Some(self.make_event(timestamp))
                } else if current_ratio == 0.0 {
                    // All samples in window are below threshold, back to idle.
                    self.transition_to(TrackingPhase::Idle, timestamp);
                    None
                } else {
                    None
                }
            }
            TrackingPhase::Confirmed => {
                // Immediately transition to Alerting.
                // The consumer should have already received the event.
                self.transition_to(TrackingPhase::Alerting, timestamp);
                None
            }
            TrackingPhase::Alerting => {
                let conf = confidence.unwrap_or(0.0);
                if conf >= self.confidence_threshold && conf >= self.peak_confidence {
                    self.peak_confidence = conf;
                    if explanation.is_some() {
                        self.peak_explanation = explanation.clone();
                    }
                }

                // Rate-limited info logging: once per second during active alert.
                if timestamp.duration_since(self.last_alerting_log) >= Duration::from_secs(1) {
                    self.last_alerting_log = timestamp;
                    info!(
                        bfrb = %self.bfrb_type,
                        positive_ratio = current_ratio,
                        peak_confidence = self.peak_confidence,
                        window_size = self.window.len(),
                        "Alert active"
                    );
                }

                // Auto-stop: transition to Idle when behavior ceases.
                // Uses Idle (not Cooldown) so re-detection is immediate.
                if window_mature && current_ratio < self.positive_ratio {
                    info!(
                        bfrb = %self.bfrb_type,
                        positive_ratio = current_ratio,
                        required_ratio = self.positive_ratio,
                        "Behavior stopped, ending alert"
                    );
                    self.transition_to(TrackingPhase::Idle, timestamp);
                }
                None
            }
            // Cooldown handled above.
            TrackingPhase::Cooldown => None,
        }
    }

    /// Dismiss the current alert (e.g., false positive hotkey or exercise completed).
    /// Transitions to Cooldown to prevent immediate re-trigger.
    pub fn dismiss(&mut self) {
        if self.phase == TrackingPhase::Alerting || self.phase == TrackingPhase::Confirmed {
            self.transition_to(TrackingPhase::Cooldown, Instant::now());
            self.window.clear();
        }
    }

    /// Reset the tracker to Idle (e.g., when pausing detection).
    pub fn reset(&mut self) {
        self.phase = TrackingPhase::Idle;
        self.window.clear();
        self.peak_confidence = 0.0;
        self.peak_explanation = None;
    }

    /// Remove samples outside the sliding window.
    fn prune_window(&mut self, now: Instant) {
        let cutoff = now.checked_sub(self.window_duration).unwrap_or(now);
        while self
            .window
            .front()
            .is_some_and(|s| s.timestamp < cutoff)
        {
            self.window.pop_front();
        }
    }

    /// Check if the window has enough data to make a meaningful decision.
    ///
    /// Requires samples spanning at least a third of the window duration to
    /// prevent a single sample from causing an immediate confirmation (1/1 = 100%)
    /// while remaining forgiving with intermittent palm detection at ~8 FPS.
    fn is_window_mature(&self) -> bool {
        if self.window.len() < 2 {
            return false;
        }
        let Some(first) = self.window.front() else {
            return false;
        };
        let Some(last) = self.window.back() else {
            return false;
        };
        let span = last.timestamp.duration_since(first.timestamp);
        span >= self.window_duration / 3
    }

    /// Compute the fraction of samples in the window that exceed the confidence threshold.
    fn compute_positive_ratio(&self) -> f32 {
        if self.window.is_empty() {
            return 0.0;
        }
        let positive_count = self
            .window
            .iter()
            .filter(|s| s.confidence >= self.confidence_threshold)
            .count();
        positive_count as f32 / self.window.len() as f32
    }

    fn transition_to(&mut self, phase: TrackingPhase, timestamp: Instant) {
        debug!(
            bfrb = %self.bfrb_type,
            from = ?self.phase,
            to = ?phase,
            "Tracker phase transition"
        );
        self.phase = phase;
        self.phase_started_at = timestamp;
        if phase == TrackingPhase::Idle {
            self.peak_confidence = 0.0;
            self.peak_explanation = None;
        }
    }

    fn make_event(&self, timestamp: Instant) -> DetectionEvent {
        let duration = timestamp.duration_since(self.phase_started_at);
        DetectionEvent {
            bfrb_type: self.bfrb_type,
            confidence: self.peak_confidence,
            started_at: self.phase_started_at,
            duration,
            camera_id: self.last_camera_id.clone(),
            explanation: self.peak_explanation.clone(),
        }
    }
}

/// Manages temporal trackers for all enabled behavior types.
pub struct DetectionTracker {
    trackers: Vec<BehaviorTracker>,
}

impl DetectionTracker {
    pub fn new(trackers: Vec<BehaviorTracker>) -> Self {
        Self { trackers }
    }

    /// Feed a frame's per-behavior confidence values.
    ///
    /// `results` is a list of `(BfrbType, Option<confidence>)` pairs from all active
    /// detectors. `None` means the detector had insufficient data (e.g., no hand
    /// detected) and should not count as either positive or negative.
    /// Returns any newly confirmed detection events.
    pub fn update(
        &mut self,
        results: &[(BfrbType, Option<f32>)],
        timestamp: Instant,
        camera_id: &str,
    ) -> Vec<DetectionEvent> {
        let mut events = Vec::new();

        for tracker in &mut self.trackers {
            // Find the confidence for this tracker's behavior type.
            let confidence = results
                .iter()
                .find(|(bfrb, _)| *bfrb == tracker.bfrb_type())
                .and_then(|(_, conf)| *conf);

            if let Some(event) = tracker.update(confidence, timestamp, camera_id) {
                events.push(event);
            }
        }

        events
    }

    /// Like `update`, but also accepts a per-detector explanation that will
    /// be carried into the resulting `DetectionEvent`.
    pub fn update_with_explanations(
        &mut self,
        results: &[(BfrbType, Option<f32>, Option<DetectionExplanation>)],
        timestamp: Instant,
        camera_id: &str,
    ) -> Vec<DetectionEvent> {
        let mut events = Vec::new();

        for tracker in &mut self.trackers {
            let entry = results.iter().find(|(bfrb, _, _)| *bfrb == tracker.bfrb_type());
            let (confidence, explanation) = match entry {
                Some((_, c, e)) => (*c, e.clone()),
                None => (None, None),
            };

            if let Some(event) =
                tracker.update_with_explanation(confidence, explanation, timestamp, camera_id)
            {
                events.push(event);
            }
        }

        events
    }

    /// Dismiss all active alerts and transition them to cooldown.
    pub fn dismiss_all(&mut self) {
        for tracker in &mut self.trackers {
            tracker.dismiss();
        }
    }

    /// Dismiss alert for a specific behavior type.
    pub fn dismiss(&mut self, bfrb_type: BfrbType) {
        for tracker in &mut self.trackers {
            if tracker.bfrb_type() == bfrb_type {
                tracker.dismiss();
            }
        }
    }

    /// Reset all trackers to idle.
    pub fn reset_all(&mut self) {
        for tracker in &mut self.trackers {
            tracker.reset();
        }
    }

    /// Get the current phase of a specific behavior tracker.
    pub fn phase_of(&self, bfrb_type: BfrbType) -> Option<TrackingPhase> {
        self.trackers
            .iter()
            .find(|t| t.bfrb_type() == bfrb_type)
            .map(|t| t.phase())
    }

    /// Check if any tracker is in the Alerting or Confirmed phase.
    pub fn any_alerting(&self) -> bool {
        self.trackers.iter().any(|t| {
            t.phase() == TrackingPhase::Alerting || t.phase() == TrackingPhase::Confirmed
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tracker() -> BehaviorTracker {
        BehaviorTracker::new(
            BfrbType::NailBiting,
            Duration::from_millis(1000),
            0.7,  // 70% positive ratio
            0.5,  // confidence threshold
            Duration::from_secs(5),
        )
    }

    #[test]
    fn starts_in_idle() {
        let tracker = make_tracker();
        assert_eq!(tracker.phase(), TrackingPhase::Idle);
    }

    #[test]
    fn transitions_to_accumulating_on_first_positive() {
        let mut tracker = make_tracker();
        let now = Instant::now();

        let event = tracker.update(Some(0.8), now, "cam0");
        assert!(event.is_none());
        assert_eq!(tracker.phase(), TrackingPhase::Accumulating);
    }

    #[test]
    fn confirms_after_sustained_detection() {
        let mut tracker = make_tracker();
        let start = Instant::now();

        // Feed enough positive samples to exceed the 70% ratio.
        // With a 1s window, sending 10 frames at 100ms intervals (all positive)
        // should trigger confirmation.
        let mut event = None;
        for i in 0..10 {
            let ts = start + Duration::from_millis(i * 100);
            event = tracker.update(Some(0.8), ts, "cam0");
            if event.is_some() {
                break;
            }
        }

        assert!(event.is_some(), "Expected a detection event");
        let event = event.unwrap();
        assert_eq!(event.bfrb_type, BfrbType::NailBiting);
        assert!(event.confidence > 0.0);
    }

    #[test]
    fn returns_to_idle_when_positive_ratio_drops() {
        let mut tracker = make_tracker();
        let start = Instant::now();

        // One positive sample to get to Accumulating.
        tracker.update(Some(0.8), start, "cam0");
        assert_eq!(tracker.phase(), TrackingPhase::Accumulating);

        // Then many negative samples to push the ratio below threshold.
        for i in 1..20 {
            let ts = start + Duration::from_millis(i * 100);
            tracker.update(Some(0.0), ts, "cam0");
        }

        assert_eq!(tracker.phase(), TrackingPhase::Idle);
    }

    #[test]
    fn dismiss_transitions_to_cooldown() {
        let mut tracker = make_tracker();
        let start = Instant::now();

        // Trigger confirmation.
        for i in 0..10 {
            let ts = start + Duration::from_millis(i * 100);
            tracker.update(Some(0.8), ts, "cam0");
        }

        // Should be in Confirmed or Alerting at this point. Feed one more
        // to advance from Confirmed -> Alerting.
        tracker.update(Some(0.8), start + Duration::from_millis(1000), "cam0");
        assert_eq!(tracker.phase(), TrackingPhase::Alerting);

        tracker.dismiss();
        assert_eq!(tracker.phase(), TrackingPhase::Cooldown);
    }

    #[test]
    fn cooldown_expires_and_returns_to_idle() {
        let mut tracker = BehaviorTracker::new(
            BfrbType::NailBiting,
            Duration::from_millis(500),
            0.7,
            0.5,
            Duration::from_millis(100), // Short cooldown for testing.
        );
        let start = Instant::now();

        // Trigger alert.
        for i in 0..10 {
            let ts = start + Duration::from_millis(i * 50);
            tracker.update(Some(0.8), ts, "cam0");
        }
        // Advance to Alerting.
        tracker.update(Some(0.8), start + Duration::from_millis(600), "cam0");
        tracker.dismiss();
        assert_eq!(tracker.phase(), TrackingPhase::Cooldown);

        // Feed a sample after cooldown expires.
        let after_cooldown = start + Duration::from_millis(800);
        tracker.update(Some(0.0), after_cooldown, "cam0");
        assert_eq!(tracker.phase(), TrackingPhase::Idle);
    }

    #[test]
    fn no_confirmation_with_low_confidence() {
        let mut tracker = make_tracker();
        let start = Instant::now();

        // Feed samples just below threshold.
        for i in 0..20 {
            let ts = start + Duration::from_millis(i * 100);
            tracker.update(Some(0.4), ts, "cam0");
        }

        assert_eq!(tracker.phase(), TrackingPhase::Idle);
    }

    #[test]
    fn none_samples_do_not_count_as_negative() {
        let mut tracker = make_tracker();
        let start = Instant::now();

        // 5 positive samples, then 10 None samples (no hand detected).
        // The None samples should NOT dilute the positive ratio.
        for i in 0..5 {
            let ts = start + Duration::from_millis(i * 100);
            tracker.update(Some(0.8), ts, "cam0");
        }
        for i in 5..15 {
            let ts = start + Duration::from_millis(i * 100);
            tracker.update(None, ts, "cam0");
        }

        // Window should still have 5 positive samples out of 5 total = 100%.
        // Window spans 400ms (samples at 0..400ms) which is < 500ms maturity.
        // But the window is only the 5 actual samples, not the None ones.
        assert_ne!(tracker.phase(), TrackingPhase::Idle);
    }

    #[test]
    fn detection_tracker_manages_multiple_behaviors() {
        let mut dt = DetectionTracker::new(vec![
            BehaviorTracker::new(
                BfrbType::NailBiting,
                Duration::from_millis(500),
                0.7,
                0.5,
                Duration::from_secs(5),
            ),
            BehaviorTracker::new(
                BfrbType::NailPicking,
                Duration::from_millis(500),
                0.7,
                0.5,
                Duration::from_secs(5),
            ),
        ]);

        let start = Instant::now();

        // Only nail biting has high confidence.
        for i in 0..10 {
            let ts = start + Duration::from_millis(i * 50);
            let events = dt.update(
                &[
                    (BfrbType::NailBiting, Some(0.9)),
                    (BfrbType::NailPicking, Some(0.1)),
                ],
                ts,
                "cam0",
            );
            if !events.is_empty() {
                assert_eq!(events.len(), 1);
                assert_eq!(events[0].bfrb_type, BfrbType::NailBiting);
                return;
            }
        }

        panic!("Expected nail biting event");
    }

    #[test]
    fn dismiss_all_puts_alerting_trackers_in_cooldown() {
        let mut dt = DetectionTracker::new(vec![BehaviorTracker::new(
            BfrbType::NailBiting,
            Duration::from_millis(500),
            0.7,
            0.5,
            Duration::from_secs(5),
        )]);

        let start = Instant::now();

        // Trigger alert.
        for i in 0..20 {
            let ts = start + Duration::from_millis(i * 50);
            dt.update(&[(BfrbType::NailBiting, Some(0.9))], ts, "cam0");
        }

        assert!(dt.any_alerting());
        dt.dismiss_all();
        assert!(!dt.any_alerting());
        assert_eq!(
            dt.phase_of(BfrbType::NailBiting),
            Some(TrackingPhase::Cooldown)
        );
    }

    #[test]
    fn reset_clears_all_state() {
        let mut tracker = make_tracker();
        let now = Instant::now();

        tracker.update(Some(0.8), now, "cam0");
        assert_eq!(tracker.phase(), TrackingPhase::Accumulating);

        tracker.reset();
        assert_eq!(tracker.phase(), TrackingPhase::Idle);
    }

    #[test]
    fn alerting_auto_stops_when_behavior_ceases() {
        let mut tracker = make_tracker();
        let start = Instant::now();

        // Trigger confirmation with sustained positive samples.
        for i in 0..10 {
            let ts = start + Duration::from_millis(i * 100);
            tracker.update(Some(0.8), ts, "cam0");
        }

        // Advance from Confirmed → Alerting.
        tracker.update(Some(0.8), start + Duration::from_millis(1000), "cam0");
        assert_eq!(tracker.phase(), TrackingPhase::Alerting);

        // Now feed negative samples to push positive_ratio below threshold.
        // The window is 1000ms, so old positive samples age out as new
        // negatives fill the window.
        for i in 0..20 {
            let ts = start + Duration::from_millis(1100 + i * 100);
            tracker.update(Some(0.0), ts, "cam0");
        }

        // Should have auto-transitioned to Idle (not Cooldown) for instant re-detection.
        assert_eq!(
            tracker.phase(),
            TrackingPhase::Idle,
            "Alerting should auto-stop to Idle when behavior ceases"
        );
    }

    #[test]
    fn peak_explanation_is_carried_into_event() {
        use crate::detection::types::{HandSignal, SuppressionReason};
        let mut tracker = make_tracker();
        let start = Instant::now();

        let mk_exp = |conf: f32| DetectionExplanation {
            bfrb_type: BfrbType::NailBiting,
            hands: vec![HandSignal {
                hand_index: 0,
                side: None,
                normalized_distance: 0.10,
                distance_threshold: 0.35,
                contributing_fingertip: Some(8),
                partner_fingertip: None,
                curl: Some(0.7),
                bonus: 0.0,
                confidence: conf,
            }],
            suppressions: Vec::<SuppressionReason>::new(),
            frame_confidence: conf,
        };

        // Feed positive samples, with the highest confidence in the middle.
        let mut event = None;
        for (i, conf) in [0.6_f32, 0.95, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
            .iter()
            .copied()
            .enumerate()
        {
            let ts = start + Duration::from_millis(i as u64 * 100);
            event = tracker.update_with_explanation(
                Some(conf),
                Some(mk_exp(conf)),
                ts,
                "cam0",
            );
            if event.is_some() {
                break;
            }
        }

        let event = event.expect("event should fire");
        let exp = event.explanation.expect("event must carry explanation");
        // The peak (0.95) should be the one preserved.
        assert!(
            (exp.frame_confidence - 0.95).abs() < 1e-6,
            "got peak frame_confidence {}",
            exp.frame_confidence
        );
        assert_eq!(exp.hands.len(), 1);
        assert_eq!(exp.hands[0].contributing_fingertip, Some(8));
    }

    #[test]
    fn peak_explanation_clears_on_idle() {
        let mut tracker = make_tracker();
        let start = Instant::now();

        let exp = DetectionExplanation::empty(BfrbType::NailBiting);
        tracker.update_with_explanation(Some(0.8), Some(exp), start, "cam0");
        assert!(tracker.peak_explanation.is_some());

        // Drop below threshold for long enough → goes to Idle and clears.
        for i in 1..30 {
            let ts = start + Duration::from_millis(i * 100);
            tracker.update_with_explanation(Some(0.0), None, ts, "cam0");
        }
        assert_eq!(tracker.phase(), TrackingPhase::Idle);
        assert!(tracker.peak_explanation.is_none());
    }

    #[test]
    fn alerting_stays_while_behavior_continues() {
        let mut tracker = make_tracker();
        let start = Instant::now();

        // Trigger confirmation.
        for i in 0..10 {
            let ts = start + Duration::from_millis(i * 100);
            tracker.update(Some(0.8), ts, "cam0");
        }

        // Advance to Alerting.
        tracker.update(Some(0.8), start + Duration::from_millis(1000), "cam0");
        assert_eq!(tracker.phase(), TrackingPhase::Alerting);

        // Continue feeding positive samples — should stay in Alerting.
        for i in 0..10 {
            let ts = start + Duration::from_millis(1100 + i * 100);
            tracker.update(Some(0.8), ts, "cam0");
        }

        assert_eq!(
            tracker.phase(),
            TrackingPhase::Alerting,
            "Alerting should persist while behavior continues"
        );
    }
}
