//! Desktop notification action using notify-rust.
//!
//! Sends a D-Bus notification when a BFRB is detected.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use notify_rust::Notification;
use tracing::debug;

use crate::actions::types::Action;
use crate::detection::types::DetectionEvent;
use crate::errors::ActionError;

pub struct NotificationAction {
    active: Arc<AtomicBool>,
}

impl Default for NotificationAction {
    fn default() -> Self {
        Self {
            active: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl NotificationAction {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Action for NotificationAction {
    fn start(&mut self, event: &DetectionEvent) -> Result<(), ActionError> {
        if self.active.load(Ordering::Relaxed) {
            return Ok(());
        }

        debug!(bfrb_type = %event.bfrb_type, "Sending desktop notification");

        Notification::new()
            .summary("Nailbite: BFRB Detected")
            .body(&format!(
                "{} detected (confidence: {:.0}%). Please complete a decoupling exercise.",
                event.bfrb_type,
                event.confidence * 100.0
            ))
            .icon("dialog-warning")
            .urgency(notify_rust::Urgency::Critical)
            .timeout(0) // Persistent until dismissed.
            .show()
            .map_err(|e| ActionError::Notification(e.to_string()))?;

        self.active.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn stop(&mut self) -> Result<(), ActionError> {
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
    fn notification_action_starts_inactive() {
        let action = NotificationAction::new();
        assert!(!action.is_active());
    }
}
