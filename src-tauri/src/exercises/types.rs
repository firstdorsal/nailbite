use crate::detection::types::{BfrbType, FaceDetection, HandDetection};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Category of decoupling exercise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExerciseCategory {
    /// Hold a position for a duration (fist clench, flat hand press, etc.)
    TimedHold,
    /// Perform a number of repetitions (ear touch, finger flick)
    Repetitions,
}

/// Current phase of an exercise session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExercisePhase {
    /// Showing instructions, waiting for user to start.
    Instruction,
    /// User is performing the exercise, camera is verifying.
    Active,
    /// Exercise completed successfully.
    Completed,
    /// Exercise timed out or user gave up.
    TimedOut,
}

/// Result of verifying the user's pose for a single frame during exercise.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether the correct pose is being held in this frame.
    pub pose_correct: bool,
    /// Confidence that the pose matches [0.0, 1.0].
    pub confidence: f32,
    /// Feedback message for the user.
    pub feedback: String,
}

/// Trait for camera-verifiable decoupling/competing response exercises.
///
/// Each exercise defines its instructions, verification logic, and
/// applicable BFRB types. The camera pipeline provides hand and face
/// landmarks that the exercise uses to verify correct pose.
pub trait Exercise: Send + Sync {
    /// Unique identifier for this exercise.
    fn id(&self) -> &str;

    /// Human-readable name.
    fn name(&self) -> &str;

    /// Detailed instruction text shown to the user.
    fn instructions(&self) -> &str;

    /// Category (timed hold vs repetitions).
    fn category(&self) -> ExerciseCategory;

    /// For `TimedHold`: how long the pose must be held.
    fn hold_duration(&self) -> Duration;

    /// For `Repetitions`: how many reps are needed.
    fn target_reps(&self) -> u32;

    /// Which BFRB types this exercise is appropriate for.
    fn applicable_to(&self) -> &[BfrbType];

    /// Verify the user's pose from hand/face landmarks.
    fn verify(
        &self,
        hands: &[HandDetection],
        face: &Option<FaceDetection>,
    ) -> VerificationResult;

    /// Maximum time allowed before exercise times out.
    fn timeout(&self) -> Duration {
        Duration::from_secs(120)
    }
}
