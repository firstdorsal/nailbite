//! Exercise guidance window.
//!
//! Shows exercise instructions and live verification feedback
//! using `egui`. Displays whether the user's pose is correct
//! and tracks progress (time held / reps completed).

use std::time::Duration;

use crossbeam_channel::Receiver;
use egui::{Align, Color32, Layout, RichText};

use crate::actions::popup::PopupMessage;
use crate::detection::types::{DetectionEvent, FrameAnalysis};
use crate::exercises::registry::ExerciseRegistry;
use crate::exercises::types::{ExerciseCategory, ExercisePhase};
use crate::exercises::verification::ExerciseSession;

/// State for the exercise guidance window.
pub struct ExerciseWindowState {
    /// Whether the window should be visible.
    pub visible: bool,
    /// The current detection event that triggered the exercise.
    pub current_event: Option<DetectionEvent>,
    /// The currently selected exercise ID.
    pub exercise_id: Option<String>,
    /// The exercise session tracker.
    pub session: Option<ExerciseSession>,
    /// Feedback text from the last verification.
    pub feedback: String,
    /// Channel for receiving popup messages from the action system.
    popup_rx: Receiver<PopupMessage>,
}

impl ExerciseWindowState {
    pub fn new(popup_rx: Receiver<PopupMessage>) -> Self {
        Self {
            visible: false,
            current_event: None,
            exercise_id: None,
            session: None,
            feedback: String::new(),
            popup_rx,
        }
    }

    /// Check for popup messages and update state.
    pub fn poll(&mut self, registry: &ExerciseRegistry) {
        while let Ok(msg) = self.popup_rx.try_recv() {
            match msg {
                PopupMessage::ShowExercise(event) => {
                    self.show_exercise(&event, registry);
                }
                PopupMessage::HideExercise => {
                    self.hide();
                }
            }
        }
    }

    /// Start showing an exercise for the given detection event.
    fn show_exercise(&mut self, event: &DetectionEvent, registry: &ExerciseRegistry) {
        if let Some(exercise) = registry.select(event.bfrb_type) {
            let mut session = ExerciseSession::new(Duration::from_secs(120), 0.8);
            session.start();

            self.exercise_id = Some(exercise.id().to_string());
            self.session = Some(session);
            self.current_event = Some(event.clone());
            self.feedback = exercise.instructions().to_string();
            self.visible = true;
        }
    }

    /// Hide the exercise window and reset state.
    pub fn hide(&mut self) {
        self.visible = false;
        self.current_event = None;
        self.exercise_id = None;
        self.session = None;
        self.feedback.clear();
    }

    /// Process a new frame analysis for exercise verification.
    ///
    /// Returns `Some(phase)` if the exercise phase changed.
    pub fn process_frame(
        &mut self,
        analysis: &FrameAnalysis,
        registry: &ExerciseRegistry,
    ) -> Option<ExercisePhase> {
        let session = self.session.as_mut()?;
        let exercise_id = self.exercise_id.as_ref()?;

        let exercise = registry
            .all()
            .iter()
            .find(|e| e.id() == exercise_id)?;

        let phase = session.process_frame(exercise.as_ref(), analysis);

        if let Some(feedback) = session.last_feedback() {
            self.feedback = feedback.to_string();
        }

        Some(phase)
    }

    /// Whether the exercise was completed.
    pub fn is_completed(&self) -> bool {
        self.session
            .as_ref()
            .map(|s| s.phase() == ExercisePhase::Completed)
            .unwrap_or(false)
    }
}

/// Render the exercise guidance UI in an egui context.
pub fn render_exercise_ui(
    ui: &mut egui::Ui,
    state: &ExerciseWindowState,
    registry: &ExerciseRegistry,
) {
    let Some(exercise_id) = &state.exercise_id else {
        ui.label("No exercise active.");
        return;
    };

    let exercise = registry
        .all()
        .iter()
        .find(|e| e.id() == exercise_id);

    let Some(exercise) = exercise else {
        ui.label("Exercise not found.");
        return;
    };

    ui.with_layout(Layout::top_down(Align::Center), |ui| {
        ui.add_space(10.0);

        // Exercise name.
        ui.label(RichText::new(exercise.name()).size(24.0).strong());
        ui.add_space(8.0);

        // Instructions.
        ui.label(RichText::new(exercise.instructions()).size(14.0));
        ui.add_space(16.0);

        // Phase indicator.
        if let Some(session) = &state.session {
            let phase = session.phase();
            let phase_color = match phase {
                ExercisePhase::Instruction => Color32::YELLOW,
                ExercisePhase::Active => Color32::GREEN,
                ExercisePhase::Completed => Color32::from_rgb(100, 200, 100),
                ExercisePhase::TimedOut => Color32::RED,
            };

            let phase_text = match phase {
                ExercisePhase::Instruction => "Get Ready",
                ExercisePhase::Active => "Active",
                ExercisePhase::Completed => "Completed!",
                ExercisePhase::TimedOut => "Timed Out",
            };

            ui.label(RichText::new(phase_text).size(20.0).color(phase_color));
            ui.add_space(12.0);

            // Progress.
            match exercise.category() {
                ExerciseCategory::TimedHold => {
                    let held = session.compliant_duration().as_secs();
                    let target = exercise.hold_duration().as_secs();
                    let ratio = if target > 0 {
                        held as f32 / target as f32
                    } else {
                        0.0
                    };
                    ui.label(
                        RichText::new(format!("Held: {held}s / {target}s"))
                            .size(18.0)
                            .monospace(),
                    );
                    let progress_bar = egui::ProgressBar::new(ratio.min(1.0))
                        .show_percentage();
                    ui.add(progress_bar);
                }
                ExerciseCategory::Repetitions => {
                    let reps = session.reps_completed();
                    let target = exercise.target_reps();
                    let ratio = if target > 0 {
                        reps as f32 / target as f32
                    } else {
                        0.0
                    };
                    ui.label(
                        RichText::new(format!("Reps: {reps} / {target}"))
                            .size(18.0)
                            .monospace(),
                    );
                    let progress_bar = egui::ProgressBar::new(ratio.min(1.0))
                        .show_percentage();
                    ui.add(progress_bar);
                }
            }

            ui.add_space(12.0);

            // Compliance ratio.
            let compliance = session.current_compliance_ratio();
            ui.label(
                RichText::new(format!("Compliance: {:.0}%", compliance * 100.0))
                    .size(14.0),
            );
        }

        ui.add_space(12.0);

        // Feedback text.
        let feedback_color = if state
            .session
            .as_ref()
            .and_then(|s| s.last_feedback())
            .is_some()
        {
            Color32::WHITE
        } else {
            Color32::GRAY
        };

        ui.label(
            RichText::new(&state.feedback)
                .size(16.0)
                .color(feedback_color),
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exercise_window_starts_hidden() {
        let (_tx, rx) = crossbeam_channel::unbounded();
        let state = ExerciseWindowState::new(rx);
        assert!(!state.visible);
        assert!(state.exercise_id.is_none());
    }

    #[test]
    fn hide_resets_state() {
        let (_tx, rx) = crossbeam_channel::unbounded();
        let mut state = ExerciseWindowState::new(rx);
        state.visible = true;
        state.feedback = "test".to_string();
        state.hide();
        assert!(!state.visible);
        assert!(state.feedback.is_empty());
    }
}
