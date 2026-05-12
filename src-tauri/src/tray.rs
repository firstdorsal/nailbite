//! Dynamic tray icon generation.
//!
//! Creates colored circle icons for different app states:
//! - Green: Ready/normal operation
//! - Red: Detection active (BFRB detected)
//! - Yellow: Not ready (camera unavailable, etc.)

use tauri::image::Image;
use tauri::AppHandle;
use tracing::warn;

/// Tray state for icon color selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrayState {
    /// Ready and monitoring (green).
    Ready,
    /// BFRB detected, alert active (red).
    Detecting,
    /// Not ready - missing camera or other issue (yellow).
    NotReady,
    /// Detection paused by the user (gray/amber, distinct from NotReady).
    Paused,
}

/// Apply the given state to the system tray (icon + tooltip). The optional
/// `today_count` is reflected in the tooltip text only — the icon stays a
/// clean colored disc so it reads at every tray-shelf size.
pub fn apply_tray_state(app: &AppHandle, state: TrayState, today_count: Option<u32>) {
    if let Some(tray) = app.tray_by_id("nailbite-tray") {
        let icon = generate_tray_icon(state);
        if let Err(e) = tray.set_icon(Some(icon)) {
            warn!(error = %e, "Failed to update tray icon");
        }
        let base = match state {
            TrayState::Ready => "Nailbite - Monitoring",
            TrayState::Detecting => "Nailbite - BFRB Detected!",
            TrayState::NotReady => "Nailbite - Camera not started",
            TrayState::Paused => "Nailbite - Detection paused",
        };
        let tooltip = match today_count {
            Some(n) => format!("{base} ({n} today)"),
            None => base.to_string(),
        };
        let _ = tray.set_tooltip(Some(tooltip));
    }
}

/// Generate a 32×32 colored circle icon for the tray.
#[must_use]
pub fn generate_tray_icon(state: TrayState) -> Image<'static> {
    const SIZE: u32 = 32;
    const RADIUS: f32 = 14.0;
    const CENTER: f32 = 16.0;

    // These match the in-app StatusIndicator colors so the tray and the
    // sidebar status dot agree: Tailwind green-500 / red-500 / yellow-500 /
    // gray-500. Update both this map AND `StatusIndicator.tsx` together.
    let (r, g, b) = match state {
        TrayState::Ready => (0x22, 0xc5, 0x5e),     // green-500 (Monitoring)
        TrayState::Detecting => (0xef, 0x44, 0x44), // red-500 (Alert)
        TrayState::Paused => (0xea, 0xb3, 0x08),    // yellow-500 (Paused)
        TrayState::NotReady => (0x6b, 0x72, 0x80),  // gray-500 (Offline)
    };

    let mut pixels = vec![0_u8; (SIZE * SIZE * 4) as usize];

    for y in 0..SIZE {
        for x in 0..SIZE {
            let dx = x as f32 - CENTER;
            let dy = y as f32 - CENTER;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist <= RADIUS {
                let alpha = if dist > RADIUS - 1.0 {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let a = ((RADIUS - dist) * 255.0) as u8;
                    a
                } else {
                    255
                };
                let idx = ((y * SIZE + x) * 4) as usize;
                if let Some(slot) = pixels.get_mut(idx..idx + 4) {
                    slot.copy_from_slice(&[r, g, b, alpha]);
                }
            }
        }
    }

    Image::new_owned(pixels, SIZE, SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_correct_size() {
        let icon = generate_tray_icon(TrayState::Ready);
        assert_eq!(icon.width(), 32);
        assert_eq!(icon.height(), 32);
    }

    #[test]
    fn all_states_produce_icons() {
        let _ = generate_tray_icon(TrayState::Ready);
        let _ = generate_tray_icon(TrayState::Detecting);
        let _ = generate_tray_icon(TrayState::NotReady);
        let _ = generate_tray_icon(TrayState::Paused);
    }
}
