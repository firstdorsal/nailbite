//! Statistics and control commands.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use tauri::{Emitter, State};
use tracing::info;

use crate::actions::types::Action;
use crate::errors::NailbiteError;
use crate::state::AppState;
use crate::stats::event_history::EventTrigger;
use crate::stats::session_log::SessionStats;
use crate::tray::{apply_tray_state, TrayState};

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

    // When pausing while an alert is firing, kill the in-flight sound too —
    // the user's intent is "stop nagging me," and leaving a `repeat: true`
    // beep running after pause feels like a bug.
    if now_paused {
        if let Err(e) = state.sound_action.lock().stop() {
            tracing::warn!(error = %e, "Failed to stop sound on pause");
        }
        *state.sound_stop_time.lock() = None;
    }

    info!(paused = now_paused, "Detection pause toggled");

    // Update the tray icon to reflect the new state.
    let alert_active = state.alert_active.load(Ordering::Relaxed);
    let tray_state = if now_paused {
        TrayState::Paused
    } else if alert_active {
        TrayState::Detecting
    } else if state.detection_running.load(Ordering::Relaxed) {
        TrayState::Ready
    } else {
        TrayState::NotReady
    };
    let count_opt = if state.config.read().general.show_detection_count {
        Some(state.today_detection_count.load(Ordering::Relaxed))
    } else {
        None
    };
    apply_tray_state(&state.app_handle, tray_state, count_opt);

    // Emit tray state change for any UI listening to the global state event.
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
    // Close the desktop notification regardless of whether the alert is
    // still flagged active — the user's in-app verdict click might have
    // raced ahead of the alert-ending pipeline, and leaving the
    // notification's "Correct / False positive" toast on screen after
    // the user has already labeled the event in-app is just confusing.
    crate::commands::camera::close_active_notification(state.inner());
    if state.alert_active.load(Ordering::Relaxed) {
        info!("Alert dismissed by user");

        // Stop the alert sound. When the user invokes dismiss explicitly
        // (F9, the in-app verdict button, or a notification action), the
        // looping `repeat: true` sound action would otherwise keep beeping
        // until its sample finished — which never happens with
        // `repeat_infinite`. The detection-loop teardown paths in `camera.rs`
        // call `sound_action.stop()` too; this is the equivalent call for
        // user-initiated dismissals.
        if let Err(e) = state.sound_action.lock().stop() {
            tracing::warn!(error = %e, "Failed to stop sound on dismiss");
        }
        *state.sound_stop_time.lock() = None;

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

        // Trigger event history recording as false positive
        state.event_history.lock().trigger(
            EventTrigger::FalsePositive,
            bfrb_type,
            None,
        );
    }

    Ok(())
}

/// Get the current pause and mute state.
#[tauri::command]
pub fn get_runtime_state(state: State<'_, Arc<AppState>>) -> serde_json::Value {
    serde_json::json!({
        "paused": state.paused.load(Ordering::Relaxed),
        "muted": state.muted.load(Ordering::Relaxed),
        "today_detection_count": state.today_detection_count.load(Ordering::Relaxed),
    })
}

/// Get just the count of detections so far today (local date).
#[tauri::command]
pub fn get_today_detection_count(state: State<'_, Arc<AppState>>) -> u32 {
    state.today_detection_count.load(Ordering::Relaxed)
}

/// Toggle the runtime mute flag. When muted, the sound action is silenced
/// even though the rest of the alert pipeline keeps running (vignette,
/// webhook, history recording).
#[tauri::command]
pub fn toggle_mute(state: State<'_, Arc<AppState>>) -> Result<bool, NailbiteError> {
    let was_muted = state.muted.load(Ordering::Relaxed);
    let now_muted = !was_muted;
    state.muted.store(now_muted, Ordering::Relaxed);

    // If muting while a sound is currently playing, stop it immediately.
    if now_muted {
        if let Err(e) = state.sound_action.lock().stop() {
            tracing::warn!(error = %e, "Failed to stop sound on mute");
        }
        *state.sound_stop_time.lock() = None;
    }

    info!(muted = now_muted, "Sound mute toggled");
    Ok(now_muted)
}

/// Mark a missed event (false negative).
#[tauri::command]
pub fn mark_missed_event(state: State<'_, Arc<AppState>>) -> Result<(), NailbiteError> {
    info!("Missed event marked by user");

    // Log to session log for training data
    state.session_log.log_missed_event();

    // Trigger event history recording for missed event
    state.event_history.lock().trigger(
        EventTrigger::MissedEvent,
        None,
        None,
    );

    Ok(())
}
