//! Global hotkey listener using the `global-hotkey` crate.
//!
//! Captures system-wide keyboard shortcuts for:
//! - Dismiss / mark false positive (default: F9)
//! - Mark missed event / false negative (default: F10)
//! - Pause / resume detection (default: F11)
//!
//! On Linux, requires X11 (Wayland is not supported by global-hotkey).

use crossbeam_channel::Sender;
use global_hotkey::hotkey::HotKey;
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};
use tracing::{debug, error};

use crate::config::HotkeysConfig;

/// Events emitted by the hotkey listener.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotkeyAction {
    /// User dismissed the alert (marks as false positive).
    DismissFalsePositive,
    /// User flagged a missed event (false negative).
    MarkMissedEvent,
    /// Toggle pause/resume detection.
    PauseResume,
}

/// Manages global hotkey registration and event forwarding.
pub struct HotkeyListener {
    _manager: GlobalHotKeyManager,
    dismiss_id: u32,
    missed_id: u32,
    pause_id: u32,
    action_tx: Sender<HotkeyAction>,
}

impl HotkeyListener {
    /// Create and register global hotkeys from configuration.
    ///
    /// Must be called from a thread with an event loop (GTK on Linux).
    pub fn new(config: &HotkeysConfig, action_tx: Sender<HotkeyAction>) -> Result<Self, String> {
        let manager = GlobalHotKeyManager::new().map_err(|e| e.to_string())?;

        let dismiss_hk = parse_hotkey_config(&config.dismiss_false_positive)?;
        let missed_hk = parse_hotkey_config(&config.mark_missed_event)?;
        let pause_hk = parse_hotkey_config(&config.pause_resume)?;

        manager
            .register(dismiss_hk)
            .map_err(|e| format!("Failed to register dismiss hotkey: {e}"))?;
        manager
            .register(missed_hk)
            .map_err(|e| format!("Failed to register missed-event hotkey: {e}"))?;
        manager
            .register(pause_hk)
            .map_err(|e| format!("Failed to register pause hotkey: {e}"))?;

        debug!(
            dismiss = %config.dismiss_false_positive,
            missed = %config.mark_missed_event,
            pause = %config.pause_resume,
            "Global hotkeys registered"
        );

        Ok(Self {
            _manager: manager,
            dismiss_id: dismiss_hk.id(),
            missed_id: missed_hk.id(),
            pause_id: pause_hk.id(),
            action_tx,
        })
    }

    /// Poll for hotkey events. Non-blocking.
    ///
    /// Should be called from the event loop thread.
    pub fn poll_events(&self) {
        if let Ok(event) = GlobalHotKeyEvent::receiver().try_recv() {
            // Only react to key press, not release.
            if event.state() != HotKeyState::Pressed {
                return;
            }

            let action = if event.id() == self.dismiss_id {
                Some(HotkeyAction::DismissFalsePositive)
            } else if event.id() == self.missed_id {
                Some(HotkeyAction::MarkMissedEvent)
            } else if event.id() == self.pause_id {
                Some(HotkeyAction::PauseResume)
            } else {
                None
            };

            if let Some(action) = action {
                debug!(?action, "Hotkey triggered");
                if self.action_tx.send(action).is_err() {
                    error!("Failed to send hotkey action: receiver dropped");
                }
            }
        }
    }
}

/// Parse a hotkey string from config (e.g. "F9", "Ctrl+F9") into a `HotKey`.
fn parse_hotkey_config(key_str: &str) -> Result<HotKey, String> {
    key_str
        .parse::<HotKey>()
        .map_err(|e| format!("Invalid hotkey '{key_str}': {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_function_keys() {
        let hk = parse_hotkey_config("F9").unwrap();
        assert_eq!(hk.key, global_hotkey::hotkey::Code::F9);

        let hk = parse_hotkey_config("F10").unwrap();
        assert_eq!(hk.key, global_hotkey::hotkey::Code::F10);

        let hk = parse_hotkey_config("F11").unwrap();
        assert_eq!(hk.key, global_hotkey::hotkey::Code::F11);
    }

    #[test]
    fn parses_modifier_combos() {
        let hk = parse_hotkey_config("Ctrl+F9").unwrap();
        assert!(hk.mods.contains(global_hotkey::hotkey::Modifiers::CONTROL));
        assert_eq!(hk.key, global_hotkey::hotkey::Code::F9);
    }

    #[test]
    fn rejects_invalid_key() {
        let result = parse_hotkey_config("NotAKey");
        assert!(result.is_err());
    }
}
