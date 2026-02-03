use crate::detection::types::DetectionEvent;
use crate::errors::ActionError;

/// Trait for actions triggered when a BFRB is detected.
///
/// Actions are started when a detection is confirmed and stopped
/// when the user completes an exercise or dismisses the alert.
pub trait Action: Send + Sync {
    /// Start the alert (called when BFRB is confirmed).
    fn start(&mut self, event: &DetectionEvent) -> Result<(), ActionError>;

    /// Stop the alert (called when exercise is completed or user dismisses).
    fn stop(&mut self) -> Result<(), ActionError>;

    /// Whether the action is currently active.
    fn is_active(&self) -> bool;
}
