//! System tray icon and context menu.
//!
//! Uses `tray-icon` crate for cross-platform tray support.
//! On Linux, requires GTK and libappindicator/libayatana-appindicator.

use crossbeam_channel::Sender;
use tracing::warn;
use tray_icon::menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem};
use tray_icon::{Icon, TrayIconBuilder};

use crate::errors::UiError;

/// Commands sent from the tray menu to the application.
#[derive(Debug, Clone)]
pub enum TrayCommand {
    /// Toggle camera preview window visibility.
    TogglePreview,
    /// Pause or resume detection.
    PauseResume,
    /// Quit the application.
    Quit,
}

/// Manages the system tray icon and context menu.
pub struct SystemTray {
    tray: tray_icon::TrayIcon,
    preview_item_id: tray_icon::menu::MenuId,
    pause_item_id: tray_icon::menu::MenuId,
    quit_item_id: tray_icon::menu::MenuId,
    command_tx: Sender<TrayCommand>,
    current_state: TrayIconState,
}

impl SystemTray {
    /// Create the system tray icon with context menu.
    ///
    /// Must be called from the same thread as the GTK event loop on Linux.
    pub fn new(command_tx: Sender<TrayCommand>) -> Result<Self, UiError> {
        let menu = Menu::new();

        let preview_item = MenuItem::new("Toggle Preview", true, None);
        let pause_item = MenuItem::new("Pause Detection", true, None);
        let quit_item = MenuItem::new("Quit", true, None);

        let preview_item_id = preview_item.id().clone();
        let pause_item_id = pause_item.id().clone();
        let quit_item_id = quit_item.id().clone();

        menu.append_items(&[
            &preview_item,
            &PredefinedMenuItem::separator(),
            &pause_item,
            &PredefinedMenuItem::separator(),
            &quit_item,
        ])
        .map_err(|e| UiError::TrayIcon(e.to_string()))?;

        let icon = generate_tray_icon(TrayIconState::Normal)?;

        let tray = TrayIconBuilder::new()
            .with_menu(Box::new(menu))
            .with_tooltip("Nailbite - BFRB Detection")
            .with_icon(icon)
            .build()
            .map_err(|e| UiError::TrayIcon(e.to_string()))?;

        Ok(Self {
            tray,
            preview_item_id,
            pause_item_id,
            quit_item_id,
            command_tx,
            current_state: TrayIconState::Normal,
        })
    }

    /// Update the tray icon to reflect a new state.
    ///
    /// Only re-renders the icon if the state actually changed.
    /// Also updates the tooltip, which forces a visual refresh on Linux
    /// AppIndicator backends that cache icons aggressively.
    pub fn set_state(&mut self, state: TrayIconState) {
        if self.current_state == state {
            return;
        }
        self.current_state = state;

        let tooltip = match state {
            TrayIconState::Normal => "Nailbite - Monitoring",
            TrayIconState::Alert => "Nailbite - BFRB Detected!",
            TrayIconState::Paused => "Nailbite - Paused",
        };

        // Update tooltip first — on some Linux AppIndicator implementations,
        // a property change (tooltip) triggers the icon refresh.
        if let Err(e) = self.tray.set_tooltip(Some(tooltip)) {
            warn!(error = %e, ?state, "Failed to update tray tooltip");
        }

        let icon = match generate_tray_icon(state) {
            Ok(i) => i,
            Err(e) => {
                warn!(error = %e, ?state, "Failed to generate tray icon");
                return;
            }
        };
        if let Err(e) = self.tray.set_icon(Some(icon)) {
            warn!(error = %e, ?state, "Failed to update tray icon");
        }
    }

    /// Poll for menu events and dispatch commands. Non-blocking.
    ///
    /// Should be called from the event loop thread.
    pub fn poll_events(&self) {
        if let Ok(event) = MenuEvent::receiver().try_recv() {
            if event.id == self.preview_item_id {
                let _ = self.command_tx.send(TrayCommand::TogglePreview);
            } else if event.id == self.pause_item_id {
                let _ = self.command_tx.send(TrayCommand::PauseResume);
            } else if event.id == self.quit_item_id {
                let _ = self.command_tx.send(TrayCommand::Quit);
            }
        }
    }
}

/// Visual state of the tray icon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrayIconState {
    /// Normal operation (monitoring).
    Normal,
    /// BFRB detected, alert active.
    Alert,
    /// Detection paused.
    Paused,
}

/// Generate a circular colored tray icon (32x32 RGBA).
///
/// Draws an anti-aliased filled circle on a transparent background.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::indexing_slicing)]
fn generate_tray_icon(state: TrayIconState) -> Result<Icon, UiError> {
    let (r, g, b) = match state {
        TrayIconState::Normal => (100, 200, 100), // Green
        TrayIconState::Alert => (220, 60, 60),     // Red
        TrayIconState::Paused => (180, 180, 180),  // Grey
    };

    let size = 32_u32;
    let center = size as f32 / 2.0;
    let radius = center - 1.5; // Slight inset for anti-aliased edge.

    // 32 * 32 * 4 = 4096, cannot overflow u32.
    let byte_count = (size * size * 4) as usize;
    let mut rgba = vec![0u8; byte_count];

    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 + 0.5 - center;
            let dy = y as f32 + 0.5 - center;
            let dist = dx.hypot(dy);

            // Anti-alias: smooth transition over ~1px at the circle edge.
            let alpha = (radius - dist + 0.5).clamp(0.0, 1.0);
            if alpha > 0.0 {
                let idx = ((y * size + x) * 4) as usize;
                rgba[idx] = r;
                rgba[idx + 1] = g;
                rgba[idx + 2] = b;
                rgba[idx + 3] = (alpha * 255.0) as u8;
            }
        }
    }

    Icon::from_rgba(rgba, size, size).map_err(|e| UiError::TrayIcon(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_icon_for_all_states() {
        for state in &[TrayIconState::Normal, TrayIconState::Alert, TrayIconState::Paused] {
            let icon = generate_tray_icon(*state);
            assert!(icon.is_ok(), "Failed to generate icon for state {state:?}");
        }
    }
}
