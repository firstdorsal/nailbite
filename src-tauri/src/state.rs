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
use crate::stats::event_history::EventHistoryRecorder;
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

    /// Runtime mute toggle for the alert sound. Persistent mute lives in
    /// `config.actions.sound.enabled`; this flag is the user-controllable
    /// "shut up for now" switch reachable from the global sidebar.
    pub muted: AtomicBool,

    /// Number of detection events confirmed today (local time). Seeded
    /// from the session log on startup, incremented in the detection
    /// loop, and reset when the local date rolls over.
    pub today_detection_count: std::sync::atomic::AtomicU32,

    /// Local-time date the count above belongs to. Used to detect
    /// midnight rollover and reset the counter.
    pub today_count_date: parking_lot::Mutex<chrono::NaiveDate>,

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

    /// XDG notification id of the desktop notification currently
    /// awaiting a verdict click, if any. Stored so the in-app
    /// AlertModal's verdict / dismiss buttons can also tear down the
    /// notification — leaving it hanging around once the user has
    /// already labeled the event would be confusing.
    ///
    /// **Not used for dedup.** `notify-rust`'s `wait_for_action` can
    /// hang indefinitely on some daemons (notably KDE / `swaync`
    /// configurations); when it does, the slot would otherwise stick
    /// and block every future notification. Dedup lives in
    /// `last_notification_at` below; this id only feeds
    /// `close_active_notification`.
    pub active_notification_id: Mutex<Option<u32>>,

    /// When the last desktop notification was submitted to the daemon.
    /// Used for time-based dedup so two BFRB events that fire back-to-
    /// back don't produce two overlapping toasts. Independent of
    /// `active_notification_id` so a hung `wait_for_action` thread
    /// can't lock out future notifications.
    pub last_notification_at: Mutex<Option<Instant>>,

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

    /// Event history recorder for debugging/review.
    pub event_history: Mutex<EventHistoryRecorder>,
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

        let event_history = EventHistoryRecorder::new(&config.history);
        event_history.prune_on_startup();
        // Seed the today-counter from the EVENT HISTORY directory rather
        // than the raw session log. The session log records every detection
        // trigger, but back-to-back triggers within `frames_after` are
        // dropped by the recorder — counting from the history dir ensures
        // the badge matches what the user actually sees in the list.
        let seeded_today_count = event_history.count_today_events();

        Self {
            sessions,
            detectors: RwLock::new(detectors),
            tracker: RwLock::new(tracker),
            config: RwLock::new(config),
            paused: AtomicBool::new(false),
            muted: AtomicBool::new(false),
            today_detection_count: std::sync::atomic::AtomicU32::new(seeded_today_count),
            today_count_date: parking_lot::Mutex::new(
                chrono::Local::now().date_naive(),
            ),
            prev_faces: Mutex::new(HashMap::new()),
            face_miss_counts: Mutex::new(HashMap::new()),
            hand_trackers: Mutex::new(HashMap::new()),
            current_bfrb: Mutex::new(None),
            current_confidence: Mutex::new(None),
            alert_active: AtomicBool::new(false),
            sound_action: Mutex::new(sound_action),
            active_notification_id: Mutex::new(None),
            last_notification_at: Mutex::new(None),
            webhook_action: Mutex::new(webhook_action),
            sound_stop_time: Mutex::new(None),
            app_handle,
            #[cfg(target_os = "linux")]
            cameras: Mutex::new(HashMap::new()),
            detection_running: AtomicBool::new(false),
            session_log,
            event_history: Mutex::new(event_history),
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

    /// Increment the today-count, resetting on local-date rollover.
    /// Returns the new count so the caller can emit it without a second
    /// load.
    pub fn bump_today_detection_count(&self) -> u32 {
        use std::sync::atomic::Ordering;
        let today = chrono::Local::now().date_naive();
        let mut held = self.today_count_date.lock();
        if *held != today {
            *held = today;
            self.today_detection_count.store(0, Ordering::Relaxed);
        }
        drop(held);
        let prev = self
            .today_detection_count
            .fetch_add(1, Ordering::Relaxed);
        prev.saturating_add(1)
    }
}

