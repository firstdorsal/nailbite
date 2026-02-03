//! Popup action that bridges to the exercise UI.
//!
//! Signals the UI thread to open the exercise guidance window
//! via a crossbeam channel.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crossbeam_channel::Sender;
use tracing::debug;

use crate::actions::types::Action;
use crate::detection::types::DetectionEvent;
use crate::errors::ActionError;

/// Message sent to the UI thread to trigger exercise display.
#[derive(Debug, Clone)]
pub enum PopupMessage {
    /// Show the exercise window for the given detection event.
    ShowExercise(DetectionEvent),
    /// Hide/close the exercise window.
    HideExercise,
}

pub struct PopupAction {
    sender: Sender<PopupMessage>,
    active: Arc<AtomicBool>,
}

impl PopupAction {
    pub fn new(sender: Sender<PopupMessage>) -> Self {
        Self {
            sender,
            active: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl Action for PopupAction {
    fn start(&mut self, event: &DetectionEvent) -> Result<(), ActionError> {
        if self.active.load(Ordering::Relaxed) {
            return Ok(());
        }

        debug!(bfrb_type = %event.bfrb_type, "Requesting exercise popup");

        self.sender
            .send(PopupMessage::ShowExercise(event.clone()))
            .map_err(|e| ActionError::Sound(format!("Failed to send popup message: {e}")))?;

        self.active.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn stop(&mut self) -> Result<(), ActionError> {
        if !self.active.load(Ordering::Relaxed) {
            return Ok(());
        }

        debug!("Hiding exercise popup");

        let _ = self.sender.send(PopupMessage::HideExercise);
        self.active.store(false, Ordering::Relaxed);
        Ok(())
    }

    fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn popup_action_starts_inactive() {
        let (sender, _receiver) = crossbeam_channel::unbounded();
        let action = PopupAction::new(sender);
        assert!(!action.is_active());
    }
}
