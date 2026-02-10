//! Dynamic tray icon generation.
//!
//! Creates colored circle icons for different app states:
//! - Green: Ready/normal operation
//! - Red: Detection active (BFRB detected)
//! - Yellow: Not ready (camera unavailable, etc.)

use tauri::image::Image;

/// Tray state for icon color selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrayState {
    /// Ready and monitoring (green).
    Ready,
    /// BFRB detected, alert active (red).
    Detecting,
    /// Not ready - missing camera or other issue (yellow).
    NotReady,
}

/// Generate a 32x32 colored circle icon for the tray.
///
/// Returns RGBA pixel data as a vector.
#[must_use]
pub fn generate_tray_icon(state: TrayState) -> Image<'static> {
    const SIZE: u32 = 32;
    const RADIUS: f32 = 14.0;
    const CENTER: f32 = 16.0;

    let (r, g, b) = match state {
        TrayState::Ready => (0x22, 0xc5, 0x5e),     // Green
        TrayState::Detecting => (0xef, 0x44, 0x44), // Red
        TrayState::NotReady => (0xea, 0xb3, 0x08),  // Yellow/amber
    };

    let mut pixels = Vec::with_capacity((SIZE * SIZE * 4) as usize);

    for y in 0..SIZE {
        for x in 0..SIZE {
            let dx = x as f32 - CENTER;
            let dy = y as f32 - CENTER;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist <= RADIUS {
                // Inside circle - use the color with anti-aliasing at edge
                let alpha = if dist > RADIUS - 1.0 {
                    // Edge anti-aliasing
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let a = ((RADIUS - dist) * 255.0) as u8;
                    a
                } else {
                    255
                };
                pixels.extend_from_slice(&[r, g, b, alpha]);
            } else {
                // Outside circle - transparent
                pixels.extend_from_slice(&[0, 0, 0, 0]);
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
    }
}
