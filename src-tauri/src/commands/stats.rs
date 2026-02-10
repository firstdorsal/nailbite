//! Statistics and control commands.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use tauri::{Emitter, State};
use tracing::info;

use crate::errors::NailbiteError;
use crate::state::AppState;
use crate::stats::session_log::SessionStats;

/// Get session statistics.
#[tauri::command]
pub fn get_stats(state: State<'_, Arc<AppState>>) -> Result<SessionStats, NailbiteError> {
    Ok(state.session_log.read_stats())
}

/// Toggle pause state.
#[tauri::command]
pub fn toggle_pause(state: State<'_, Arc<AppState>>) -> Result<bool, NailbiteError> {
    let was_paused = state.paused.load(Ordering::Relaxed);
    let now_paused = !was_paused;
    state.paused.store(now_paused, Ordering::Relaxed);

    info!(paused = now_paused, "Detection pause toggled");

    // Emit tray state change
    let _ = state.app_handle.emit(
        "tray-state",
        serde_json::json!({
            "state": if now_paused { "paused" } else { "normal" }
        }),
    );

    Ok(now_paused)
}

/// Dismiss the current alert.
#[tauri::command]
pub fn dismiss_alert(state: State<'_, Arc<AppState>>) -> Result<(), NailbiteError> {
    if state.alert_active.load(Ordering::Relaxed) {
        info!("Alert dismissed by user");

        // Get the current BFRB type before clearing
        let bfrb_type = state.current_bfrb.lock().take();

        // Clear state
        *state.current_confidence.lock() = None;
        state.alert_active.store(false, Ordering::Relaxed);

        // Reset tracker
        {
            let mut tracker = state.tracker.write();
            tracker.reset_all();
        }

        // Emit event
        let _ = state.app_handle.emit("alert-ended", serde_json::json!({
            "bfrb_type": "dismissed"
        }));

        // Log as false positive to session log
        state.session_log.log_dismissed(bfrb_type);
    }

    Ok(())
}

/// Mark a missed event (false negative).
#[tauri::command]
pub fn mark_missed_event(state: State<'_, Arc<AppState>>) -> Result<(), NailbiteError> {
    info!("Missed event marked by user");

    // Log to session log for training data
    state.session_log.log_missed_event();

    Ok(())
}
