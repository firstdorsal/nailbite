//! Camera-based exercise verification loop.
//!
//! Runs during an exercise session, checking each frame's landmarks
//! against the exercise's verification criteria. Tracks hold time
//! for timed exercises and rep counts for repetition exercises.

use std::time::{Duration, Instant};

use crate::detection::types::FrameAnalysis;
use crate::exercises::types::{Exercise, ExerciseCategory, ExercisePhase, VerificationResult};

/// Manages the verification state for a single exercise session.
pub struct ExerciseSession {
    phase: ExercisePhase,
    started_at: Option<Instant>,
    /// Accumulated time where the pose was correct (for timed holds).
    compliant_duration: Duration,
    /// Number of completed repetitions.
    reps_completed: u32,
    /// Whether the pose was correct in the previous frame (for edge detection).
    was_correct: bool,
    /// Total frames checked.
    total_frames: u32,
    /// Frames where pose was correct.
    correct_frames: u32,
    /// Maximum session time before timeout.
    timeout: Duration,
    /// Compliance ratio required.
    compliance_ratio: f32,
    /// Last verification result for UI feedback.
    last_result: Option<VerificationResult>,
}

impl ExerciseSession {
    pub fn new(timeout: Duration, compliance_ratio: f32) -> Self {
        Self {
            phase: ExercisePhase::Instruction,
            started_at: None,
            compliant_duration: Duration::ZERO,
            reps_completed: 0,
            was_correct: false,
            total_frames: 0,
            correct_frames: 0,
            timeout,
            compliance_ratio,
            last_result: None,
        }
    }

    /// Current exercise phase.
    pub fn phase(&self) -> ExercisePhase {
        self.phase
    }

    /// Start the active exercise phase.
    pub fn start(&mut self) {
        self.phase = ExercisePhase::Active;
        self.started_at = Some(Instant::now());
    }

    /// Number of completed reps.
    pub fn reps_completed(&self) -> u32 {
        self.reps_completed
    }

    /// Accumulated compliant hold duration.
    pub fn compliant_duration(&self) -> Duration {
        self.compliant_duration
    }

    /// Last verification feedback for UI display.
    pub fn last_feedback(&self) -> Option<&str> {
        self.last_result.as_ref().map(|r| r.feedback.as_str())
    }

    /// The overall compliance ratio so far.
    pub fn current_compliance_ratio(&self) -> f32 {
        if self.total_frames == 0 {
            return 0.0;
        }
        self.correct_frames as f32 / self.total_frames as f32
    }

    /// Process a frame and update the session state.
    ///
    /// Returns the current phase after processing.
    pub fn process_frame(
        &mut self,
        exercise: &dyn Exercise,
        analysis: &FrameAnalysis,
    ) -> ExercisePhase {
        if self.phase != ExercisePhase::Active {
            return self.phase;
        }

        // Check timeout.
        if let Some(started) = self.started_at {
            if started.elapsed() >= self.timeout {
                self.phase = ExercisePhase::TimedOut;
                return self.phase;
            }
        }

        // Verify pose.
        let result = exercise.verify(&analysis.hands, &analysis.face);
        self.total_frames += 1;

        if result.pose_correct {
            self.correct_frames += 1;
        }

        match exercise.category() {
            ExerciseCategory::TimedHold => {
                self.process_timed_hold(exercise, &result);
            }
            ExerciseCategory::Repetitions => {
                self.process_repetitions(exercise, &result);
            }
        }

        self.last_result = Some(result);
        self.phase
    }

    fn process_timed_hold(&mut self, exercise: &dyn Exercise, result: &VerificationResult) {
        if result.pose_correct {
            // Estimate frame duration as the interval between frames.
            // At 8 inference FPS, each frame is ~125ms.
            let frame_duration = Duration::from_millis(125);
            self.compliant_duration += frame_duration;
        }

        // Check if the hold duration is met with sufficient compliance.
        if self.compliant_duration >= exercise.hold_duration()
            && self.current_compliance_ratio() >= self.compliance_ratio
        {
            self.phase = ExercisePhase::Completed;
        }
    }

    fn process_repetitions(&mut self, exercise: &dyn Exercise, result: &VerificationResult) {
        // Count a rep on the rising edge (transition from incorrect to correct).
        if result.pose_correct && !self.was_correct {
            self.reps_completed += 1;
        }
        self.was_correct = result.pose_correct;

        if self.reps_completed >= exercise.target_reps() {
            self.phase = ExercisePhase::Completed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detection::types::HandDetection;
    use crate::exercises::fist_clench::FistClench;
    use crate::exercises::finger_flick::FingerFlick;
    use std::sync::Arc;
    use std::time::Instant;

    fn make_analysis(hands: Vec<HandDetection>) -> FrameAnalysis {
        FrameAnalysis {
            timestamp: Instant::now(),
            camera_id: Arc::from("test"),
            hands,
            face: None,
            raw_frame: None,
        }
    }

    #[test]
    fn starts_in_instruction_phase() {
        let session = ExerciseSession::new(Duration::from_secs(120), 0.8);
        assert_eq!(session.phase(), ExercisePhase::Instruction);
    }

    #[test]
    fn transitions_to_active_on_start() {
        let mut session = ExerciseSession::new(Duration::from_secs(120), 0.8);
        session.start();
        assert_eq!(session.phase(), ExercisePhase::Active);
    }

    #[test]
    fn times_out_after_timeout() {
        let mut session = ExerciseSession::new(Duration::from_millis(1), 0.8);
        session.start();

        // Wait for timeout.
        std::thread::sleep(Duration::from_millis(10));

        let exercise = FistClench;
        let analysis = make_analysis(vec![]);
        let phase = session.process_frame(&exercise, &analysis);
        assert_eq!(phase, ExercisePhase::TimedOut);
    }

    #[test]
    fn rep_counting_on_rising_edge() {
        let mut session = ExerciseSession::new(Duration::from_secs(120), 0.8);
        session.start();

        // Simulate repetition detection via was_correct toggling.
        // For finger flick: extended = correct, curled = incorrect.
        // We simulate rising edges by alternating.
        assert_eq!(session.reps_completed(), 0);

        // Simulate "incorrect" then "correct" = 1 rep.
        session.was_correct = false;
        let result_correct = VerificationResult {
            pose_correct: true,
            confidence: 0.9,
            feedback: "good".to_string(),
        };
        let exercise = FingerFlick;

        // Manually call the repetitions logic.
        session.process_repetitions(&exercise, &result_correct);
        assert_eq!(session.reps_completed(), 1);

        // Same pose again (still correct) = no new rep.
        session.process_repetitions(&exercise, &result_correct);
        assert_eq!(session.reps_completed(), 1);

        // Reset to incorrect.
        let result_incorrect = VerificationResult {
            pose_correct: false,
            confidence: 0.2,
            feedback: "curl".to_string(),
        };
        session.process_repetitions(&exercise, &result_incorrect);
        assert_eq!(session.reps_completed(), 1);

        // New rising edge = rep 2.
        session.process_repetitions(&exercise, &result_correct);
        assert_eq!(session.reps_completed(), 2);
    }
}
