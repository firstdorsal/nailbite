//! Application state shared across Tauri commands.

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::{Mutex, RwLock};
use tauri::AppHandle;

#[cfg(target_os = "linux")]
use crate::camera::CameraCapture;
use crate::actions::sound::SoundAction;
use crate::actions::webhook::WebhookAction;
use crate::config::NailbiteConfig;
use crate::stats::session_log::SessionLog;
use crate::detection::behaviors::BehaviorDetector;
use crate::detection::hand_tracker::HandTracker;
use crate::detection::tracker::DetectionTracker;
use crate::detection::types::{BfrbType, FaceDetection};
use crate::inference::session::ModelSessions;

/// Shared application state managed by Tauri.
pub struct AppState {
    /// ONNX model sessions (shared, read-only after init).
    pub sessions: Arc<ModelSessions>,

    /// Active behavior detectors (one per enabled BFRB type).
    /// Wrapped in RwLock to allow hot-reload of config (ARCH-3).
    pub detectors: RwLock<Vec<Box<dyn BehaviorDetector>>>,

    /// Temporal tracking state machine.
    /// Wrapped in RwLock to allow hot-reload of config (ARCH-3).
    pub tracker: RwLock<DetectionTracker>,

    /// Current configuration.
    pub config: RwLock<NailbiteConfig>,

    /// Whether detection is paused.
    pub paused: AtomicBool,

    /// Previous face detection for EMA smoothing (keyed by camera ID).
    pub prev_faces: Mutex<HashMap<String, FaceDetection>>,

    /// Face detection miss counts for grace period (keyed by camera ID).
    pub face_miss_counts: Mutex<HashMap<String, u8>>,

    /// Hand trackers for temporal consistency (keyed by camera ID).
    pub hand_trackers: Mutex<HashMap<String, HandTracker>>,

    /// Currently active BFRB type (if alerting).
    pub current_bfrb: Mutex<Option<BfrbType>>,

    /// Current detection confidence.
    pub current_confidence: Mutex<Option<f32>>,

    /// Whether alert sound is currently playing.
    pub alert_active: AtomicBool,

    /// Sound action for alert beeps.
    pub sound_action: Mutex<SoundAction>,

    /// Webhook action for remote notifications (None if disabled).
    pub webhook_action: Mutex<Option<WebhookAction>>,

    /// When to stop the sound (alert end time + 0.5s tail).
    pub sound_stop_time: Mutex<Option<Instant>>,

    /// Tauri app handle for emitting events and tray updates.
    pub app_handle: AppHandle,

    /// Camera capture handles keyed by camera ID (Linux only).
    #[cfg(target_os = "linux")]
    pub cameras: Mutex<HashMap<String, CameraCapture>>,

    /// Detection loop running flag.
    pub detection_running: AtomicBool,

    /// Session log for statistics and training data.
    pub session_log: SessionLog,
}

impl AppState {
    /// Create new application state.
    pub fn new(
        sessions: Arc<ModelSessions>,
        detectors: Vec<Box<dyn BehaviorDetector>>,
        tracker: DetectionTracker,
        config: NailbiteConfig,
        app_handle: AppHandle,
    ) -> Self {
        let sound_action = SoundAction::new(
            &config.actions.sound.file,
            config.actions.sound.volume,
            config.actions.sound.repeat,
        );

        // Create webhook action if enabled and URL is configured.
        // Create session log (with default path if config path fails).
        let session_log = SessionLog::new(std::path::Path::new(&config.general.stats_file))
            .unwrap_or_else(|e| {
                tracing::warn!(error = %e, "Failed to create session log at configured path, using default");
                SessionLog::new(std::path::Path::new("~/.local/share/nailbite/stats.jsonl"))
                    .expect("Failed to create session log at default path")
            });

        let webhook_action = if config.actions.webhook.enabled && !config.actions.webhook.url.is_empty() {
            match WebhookAction::new(
                &config.actions.webhook.url,
                config.actions.webhook.timeout_ms,
                config.actions.webhook.headers.clone(),
            ) {
                Ok(action) => Some(action),
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to create webhook action, disabling");
                    None
                }
            }
        } else {
            None
        };

        Self {
            sessions,
            detectors: RwLock::new(detectors),
            tracker: RwLock::new(tracker),
            config: RwLock::new(config),
            paused: AtomicBool::new(false),
            prev_faces: Mutex::new(HashMap::new()),
            face_miss_counts: Mutex::new(HashMap::new()),
            hand_trackers: Mutex::new(HashMap::new()),
            current_bfrb: Mutex::new(None),
            current_confidence: Mutex::new(None),
            alert_active: AtomicBool::new(false),
            sound_action: Mutex::new(sound_action),
            webhook_action: Mutex::new(webhook_action),
            sound_stop_time: Mutex::new(None),
            app_handle,
            #[cfg(target_os = "linux")]
            cameras: Mutex::new(HashMap::new()),
            detection_running: AtomicBool::new(false),
            session_log,
        }
    }

    /// Clear per-camera state when cameras are stopped.
    /// This prevents unbounded growth of HashMaps.
    pub fn clear_camera_state(&self) {
        self.prev_faces.lock().clear();
        self.face_miss_counts.lock().clear();
        self.hand_trackers.lock().clear();
    }

    /// Clear state for a specific camera.
    #[allow(dead_code)]
    pub fn clear_camera_state_for(&self, camera_id: &str) {
        self.prev_faces.lock().remove(camera_id);
        self.face_miss_counts.lock().remove(camera_id);
        self.hand_trackers.lock().remove(camera_id);
    }
}
