pub mod nail_biting;
pub mod nail_picking;

use crate::detection::types::{BfrbType, DetectionExplanation, FrameAnalysis};
use std::time::Duration;

/// Trait for implementing BFRB behavior detectors.
///
/// Each detector analyzes frame landmarks for a specific behavior pattern.
/// New BFRBs are added by implementing this trait.
pub trait BehaviorDetector: Send + Sync {
    /// The type of BFRB this detector handles.
    fn bfrb_type(&self) -> BfrbType;

    /// Human-readable name for this detector.
    fn name(&self) -> &str;

    /// Analyze a single frame's landmarks and return a raw confidence [0.0, 1.0]
    /// that the behavior is occurring.
    ///
    /// Returns `None` if insufficient data is available (e.g., no hands detected
    /// for a hand-based detector).
    fn analyze_frame(&self, analysis: &FrameAnalysis) -> Option<f32> {
        self.analyze_frame_explained(analysis)
            .map(|(c, _)| c)
    }

    /// Like `analyze_frame`, but additionally returns the contributing signals
    /// that led to the confidence value. Used to explain detections to the
    /// user, persist per-frame state, and feed label-driven threshold tuning.
    ///
    /// The default implementation returns an empty explanation alongside
    /// `analyze_frame`'s value, so existing detectors keep compiling. Real
    /// detectors should override this and have `analyze_frame` defer to it.
    fn analyze_frame_explained(
        &self,
        analysis: &FrameAnalysis,
    ) -> Option<(f32, DetectionExplanation)>;

    /// Minimum sustained duration before triggering an alert.
    /// Prevents false positives from brief touches.
    fn min_sustained_duration(&self) -> Duration;

    /// Confidence threshold above which a single frame is considered positive.
    fn confidence_threshold(&self) -> f32;

    /// Whether this detector requires face mesh data.
    fn requires_face(&self) -> bool;

    /// Whether this detector requires hand landmark data.
    fn requires_hands(&self) -> bool;
}
