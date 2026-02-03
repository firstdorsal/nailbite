//! Application lifecycle orchestrator.
//!
//! Coordinates camera pipelines, detection thread, action system,
//! exercise sessions, and UI components.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crossbeam_channel::{select, tick, Receiver, Sender};
use tracing::{debug, error, info, warn};

use crate::actions::notification::NotificationAction;
use crate::actions::popup::{PopupAction, PopupMessage};
use crate::actions::sound::SoundAction;
use crate::actions::types::Action;
use crate::actions::webhook::WebhookAction;
use crate::camera::backend::CameraBackend;
use crate::camera::pipeline::CameraPipeline;
use crate::camera::v4l_backend::V4lBackend;
use crate::config::NailbiteConfig;
use crate::detection::behaviors::nail_biting::NailBitingDetector;
use crate::detection::behaviors::nail_picking::NailPickingDetector;
use crate::detection::behaviors::BehaviorDetector;
use crate::detection::tracker::{BehaviorTracker, DetectionTracker};
use crate::detection::types::{BfrbType, FrameAnalysis};
use crate::errors::NailbiteError;
use crate::exercises::registry::ExerciseRegistry;
use crate::hotkeys::listener::{HotkeyAction, HotkeyListener};
use crate::inference::model_downloader;
use crate::inference::session::ModelSessions;
use crate::stats::session_log::SessionLog;
use crate::training::annotation::AnnotationType;
use crate::training::collector::TrainingCollector;
use crate::ui::preview::PreviewWindow;
use crate::ui::tray::{SystemTray, TrayCommand, TrayIconState};

/// Application state and coordination.
pub struct App {
    config: NailbiteConfig,
    paused: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
}

impl App {
    pub fn new(config: NailbiteConfig) -> Self {
        Self {
            config,
            paused: Arc::new(AtomicBool::new(false)),
            running: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Run the application.
    ///
    /// This is the main entry point that sets up all subsystems and runs
    /// the event loop. It returns when the user quits via tray menu or signal.
    pub fn run(&self, _show_preview: bool) -> Result<(), NailbiteError> {
        info!("Initializing nailbite application");

        // --- GTK init (required before tray-icon on Linux) ---
        gtk::init().map_err(|e| {
            NailbiteError::Ui(crate::errors::UiError::TrayIcon(format!(
                "GTK initialization failed: {e}"
            )))
        })?;
        info!("GTK initialized");

        // --- Preview window (created before cameras so it's ready to receive frames) ---
        let preview = PreviewWindow::new();
        info!("Preview window created (hidden)");

        // --- Download models if missing ---
        info!("Checking ONNX models");
        model_downloader::ensure_models(&self.config.models)?;

        // --- Load ONNX inference sessions ---
        info!("Loading ONNX model sessions");
        let sessions = ModelSessions::load(&self.config.models, &self.config.ort)?;

        // --- Channels ---
        let (analysis_tx, analysis_rx) = crossbeam_channel::bounded::<FrameAnalysis>(1);
        let (tray_cmd_tx, tray_cmd_rx) = crossbeam_channel::bounded::<TrayCommand>(16);
        let (hotkey_tx, hotkey_rx) = crossbeam_channel::bounded::<HotkeyAction>(16);
        let (popup_tx, _popup_rx) = crossbeam_channel::bounded::<PopupMessage>(16);

        // --- Spawn camera pipeline threads ---
        let mut camera_threads = Vec::new();
        for camera_config in &self.config.cameras {
            let pipeline = CameraPipeline::new(camera_config, &sessions);
            let tx = analysis_tx.clone();
            let running = Arc::clone(&self.running);
            let device = camera_config.device.clone();
            let width = camera_config.width;
            let height = camera_config.height;
            let fps = camera_config.fps;
            let camera_id = camera_config.id.clone();

            let camera_id_for_panic = camera_id.clone();
            let handle = thread::Builder::new()
                .name(format!("camera-{camera_id}"))
                .spawn(move || {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        // Retry loop: keeps trying to open the camera, handles hot-plug.
                        while running.load(std::sync::atomic::Ordering::Relaxed) {
                            match V4lBackend::open(&device, width, height, fps) {
                                Ok(mut backend) => {
                                    info!(camera = %camera_id, device = %device, "Camera opened");
                                    pipeline.run(&mut backend, &tx, &running);
                                    // run() returns when running=false or device is lost.
                                    if running.load(std::sync::atomic::Ordering::Relaxed) {
                                        warn!(camera = %camera_id, "Camera disconnected, will retry");
                                    }
                                }
                                Err(e) => {
                                    debug!(
                                        camera = %camera_id,
                                        device = %device,
                                        error = %e,
                                        "Camera not available, retrying in 2s"
                                    );
                                }
                            }
                            // Wait before retrying (unless shutting down).
                            if running.load(std::sync::atomic::Ordering::Relaxed) {
                                thread::sleep(Duration::from_secs(2));
                            }
                        }
                    }));

                    if let Err(panic_info) = result {
                        let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                            s.to_string()
                        } else if let Some(s) = panic_info.downcast_ref::<String>() {
                            s.clone()
                        } else {
                            "unknown panic".to_string()
                        };
                        error!(camera = %camera_id_for_panic, panic = %msg, "Camera thread panicked");
                    }
                })
                .map_err(NailbiteError::Io)?;

            camera_threads.push(handle);
        }
        // Drop the original sender so the channel closes when all pipeline threads stop.
        drop(analysis_tx);

        info!(
            count = camera_threads.len(),
            "Camera pipeline threads started"
        );

        // --- Detection subsystem ---
        let detectors = self.build_detectors();
        let tracker = self.build_tracker(&detectors);

        // --- Actions ---
        let actions = self.build_actions(&popup_tx);

        // --- Exercise registry ---
        let registry = ExerciseRegistry::new(
            self.config.exercises.selection_strategy,
            self.config.exercises.preferred_exercise.clone(),
        );

        // --- Session log ---
        let session_log = SessionLog::new(&self.config.general.stats_file)
            .map_err(|e| NailbiteError::Io(std::io::Error::other(e)))?;

        // --- Training collector ---
        let training_collector = TrainingCollector::new(&self.config.training)
            .map_err(|e| NailbiteError::Io(std::io::Error::other(e.to_string())))?;

        // --- System tray ---
        let mut tray = match SystemTray::new(tray_cmd_tx) {
            Ok(tray) => {
                info!("System tray initialized");
                Some(tray)
            }
            Err(e) => {
                warn!(error = %e, "Failed to initialize system tray, continuing without it");
                None
            }
        };

        // --- Global hotkeys ---
        let hotkeys = match HotkeyListener::new(&self.config.hotkeys, hotkey_tx) {
            Ok(listener) => {
                info!("Global hotkeys registered");
                Some(listener)
            }
            Err(e) => {
                warn!(error = %e, "Failed to register global hotkeys, continuing without them");
                None
            }
        };

        info!("Nailbite application started, entering main loop");

        // --- Main event loop ---
        self.event_loop(
            analysis_rx,
            tray_cmd_rx,
            hotkey_rx,
            detectors,
            tracker,
            actions,
            &registry,
            &session_log,
            &training_collector,
            &preview,
            tray.as_mut(),
            hotkeys.as_ref(),
        );

        // --- Shutdown: wait for camera threads ---
        info!("Nailbite application shutting down");
        for handle in camera_threads {
            let _ = handle.join();
        }

        Ok(())
    }

    /// Build behavior detectors from config.
    fn build_detectors(&self) -> Vec<Box<dyn BehaviorDetector>> {
        let mut detectors: Vec<Box<dyn BehaviorDetector>> = Vec::new();
        let fp = &self.config.detection.false_positive;

        if self.config.detection.behaviors.nail_biting.enabled {
            detectors.push(Box::new(NailBitingDetector::new(
                &self.config.detection.behaviors.nail_biting,
                fp.chin_rest_suppression,
                fp.typing_suppression,
            )));
        }

        if self.config.detection.behaviors.nail_picking.enabled {
            detectors.push(Box::new(NailPickingDetector::new(
                &self.config.detection.behaviors.nail_picking,
                fp.typing_suppression,
            )));
        }

        info!(count = detectors.len(), "Behavior detectors initialized");
        detectors
    }

    /// Build the temporal detection tracker from config and detectors.
    fn build_tracker(&self, detectors: &[Box<dyn BehaviorDetector>]) -> DetectionTracker {
        let window_duration =
            Duration::from_millis(self.config.detection.temporal.window_ms);
        let positive_ratio = self.config.detection.temporal.positive_ratio;
        let cooldown_duration =
            Duration::from_secs(self.config.general.cooldown_seconds);

        let behavior_trackers: Vec<BehaviorTracker> = detectors
            .iter()
            .map(|d| {
                BehaviorTracker::new(
                    d.bfrb_type(),
                    window_duration,
                    positive_ratio,
                    d.confidence_threshold(),
                    cooldown_duration,
                )
            })
            .collect();

        DetectionTracker::new(behavior_trackers)
    }

    /// Build action handlers from config.
    fn build_actions(&self, popup_tx: &Sender<PopupMessage>) -> Vec<Box<dyn Action>> {
        let mut actions: Vec<Box<dyn Action>> = Vec::new();

        if self.config.actions.sound.enabled {
            actions.push(Box::new(SoundAction::new(
                &self.config.actions.sound.file,
                self.config.actions.sound.volume,
                self.config.actions.sound.repeat,
            )));
        }

        if self.config.actions.notification.enabled {
            actions.push(Box::new(NotificationAction::new()));
        }

        if self.config.actions.webhook.enabled && !self.config.actions.webhook.url.is_empty() {
            match WebhookAction::new(
                &self.config.actions.webhook.url,
                self.config.actions.webhook.timeout_ms,
                self.config.actions.webhook.headers.clone(),
            ) {
                Ok(webhook) => actions.push(Box::new(webhook)),
                Err(e) => warn!(error = %e, "Failed to create webhook action, skipping"),
            }
        }

        if self.config.actions.popup.enabled {
            actions.push(Box::new(PopupAction::new(popup_tx.clone())));
        }

        info!(count = actions.len(), "Actions initialized");
        actions
    }

    /// Main event loop that coordinates all subsystems.
    #[allow(clippy::too_many_arguments)]
    fn event_loop(
        &self,
        analysis_rx: Receiver<FrameAnalysis>,
        tray_cmd_rx: Receiver<TrayCommand>,
        hotkey_rx: Receiver<HotkeyAction>,
        detectors: Vec<Box<dyn BehaviorDetector>>,
        mut tracker: DetectionTracker,
        mut actions: Vec<Box<dyn Action>>,
        _registry: &ExerciseRegistry,
        session_log: &SessionLog,
        training_collector: &TrainingCollector,
        preview: &PreviewWindow,
        mut tray: Option<&mut SystemTray>,
        hotkey_listener: Option<&HotkeyListener>,
    ) {
        let ticker = tick(Duration::from_millis(50));
        let mut current_bfrb: Option<BfrbType> = None;
        let mut current_confidence: Option<f32> = None;

        while self.running.load(Ordering::Relaxed) {
            select! {
                // Process incoming frame analyses.
                recv(analysis_rx) -> msg => {
                    if let Ok(mut analysis) = msg {
                        // Drain to latest frame: skip older frames that arrived
                        // while we were busy processing the previous one.
                        while let Ok(newer) = analysis_rx.try_recv() {
                            analysis = newer;
                        }

                        if self.paused.load(Ordering::Relaxed) {
                            continue;
                        }

                        // Run behavior detectors and collect confidence values.
                        // None = detector had insufficient data (e.g., no hand).
                        let results: Vec<(BfrbType, Option<f32>)> = detectors
                            .iter()
                            .map(|detector| {
                                let confidence = detector.analyze_frame(&analysis);
                                debug!(
                                    detector = %detector.bfrb_type(),
                                    confidence = ?confidence,
                                    hands = analysis.hands.len(),
                                    face = analysis.face.is_some(),
                                    "Frame detection result"
                                );
                                if let Some(c) = confidence {
                                    if c > 0.0 {
                                        info!(
                                            detector = %detector.bfrb_type(),
                                            confidence = c,
                                            "Behavior signal"
                                        );
                                    }
                                }
                                (detector.bfrb_type(), confidence)
                            })
                            .collect();

                        // Feed results to the temporal tracker.
                        let events = tracker.update(
                            &results,
                            analysis.timestamp,
                            &analysis.camera_id,
                        );

                        // Process any newly confirmed detection events.
                        for event in events {
                            info!(
                                bfrb_type = %event.bfrb_type,
                                confidence = event.confidence,
                                "BFRB detected, triggering actions"
                            );

                            session_log.log_detection(
                                event.bfrb_type,
                                event.confidence,
                                &event.camera_id,
                            );

                            current_bfrb = Some(event.bfrb_type);
                            current_confidence = Some(event.confidence);

                            // Update tray icon to alert state.
                            if let Some(ref mut t) = tray {
                                t.set_state(TrayIconState::Alert);
                            }

                            // Start all actions.
                            for action in &mut actions {
                                if let Err(e) = action.start(&event) {
                                    warn!(error = %e, "Failed to start action");
                                }
                            }
                        }

                        // Update preview window with latest frame + landmarks.
                        preview.update(&mut analysis);
                    }
                },

                // Process tray menu commands.
                recv(tray_cmd_rx) -> msg => {
                    if let Ok(cmd) = msg {
                        match cmd {
                            TrayCommand::TogglePreview => {
                                preview.toggle_visible();
                                debug!(visible = preview.is_visible(), "Toggle preview");
                            }
                            TrayCommand::PauseResume => {
                                let was_paused = self.paused.fetch_xor(true, Ordering::Relaxed);
                                if was_paused {
                                    info!("Detection resumed");
                                    session_log.log_resumed();
                                    if let Some(ref mut t) = tray {
                                        t.set_state(TrayIconState::Normal);
                                    }
                                } else {
                                    info!("Detection paused");
                                    session_log.log_paused();
                                    if let Some(ref mut t) = tray {
                                        t.set_state(TrayIconState::Paused);
                                    }
                                }
                            }
                            TrayCommand::Quit => {
                                info!("Quit requested via tray menu");
                                self.running.store(false, Ordering::Relaxed);
                            }
                        }
                    }
                },

                // Process global hotkey actions.
                recv(hotkey_rx) -> msg => {
                    if let Ok(action) = msg {
                        match action {
                            HotkeyAction::DismissFalsePositive => {
                                info!("Alert dismissed (false positive)");

                                // Stop all actions.
                                for a in &mut actions {
                                    if let Err(e) = a.stop() {
                                        warn!(error = %e, "Failed to stop action");
                                    }
                                }

                                // Put tracker in cooldown.
                                tracker.dismiss_all();

                                // Return tray icon to normal.
                                if let Some(ref mut t) = tray {
                                    t.set_state(TrayIconState::Normal);
                                }

                                // Record annotation.
                                session_log.log_dismissed(current_bfrb);
                                if let Err(e) = training_collector.record_annotation(
                                    AnnotationType::FalsePositive,
                                    current_bfrb,
                                    current_confidence,
                                ) {
                                    warn!(error = %e, "Failed to record false positive annotation");
                                }

                                current_bfrb = None;
                                current_confidence = None;
                            }
                            HotkeyAction::MarkMissedEvent => {
                                info!("Missed event flagged (false negative)");
                                session_log.log_missed_event();
                                if let Err(e) = training_collector.record_annotation(
                                    AnnotationType::FalseNegative,
                                    None,
                                    None,
                                ) {
                                    warn!(error = %e, "Failed to record false negative annotation");
                                }
                            }
                            HotkeyAction::PauseResume => {
                                let was_paused = self.paused.fetch_xor(true, Ordering::Relaxed);
                                if was_paused {
                                    info!("Detection resumed via hotkey");
                                    session_log.log_resumed();
                                    if let Some(ref mut t) = tray {
                                        t.set_state(TrayIconState::Normal);
                                    }
                                } else {
                                    info!("Detection paused via hotkey");
                                    session_log.log_paused();
                                    if let Some(ref mut t) = tray {
                                        t.set_state(TrayIconState::Paused);
                                    }
                                }
                            }
                        }
                    }
                },

                // Periodic tick for tray/hotkey polling and maintenance.
                recv(ticker) -> _ => {
                    // GTK event processing (required for tray-icon and global-hotkey).
                    while gtk::events_pending() {
                        gtk::main_iteration_do(false);
                    }

                    // Forward tray menu events to the command channel.
                    if let Some(ref t) = tray {
                        t.poll_events();
                    }

                    // Forward global hotkey events to the action channel.
                    if let Some(listener) = hotkey_listener {
                        listener.poll_events();
                    }
                }
            }
        }

        // Cleanup: stop all actions.
        for action in &mut actions {
            let _ = action.stop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_creates_from_config() {
        let config = NailbiteConfig::load("config.yaml").unwrap();
        let app = App::new(config);
        assert!(!app.paused.load(Ordering::Relaxed));
        assert!(app.running.load(Ordering::Relaxed));
    }

    #[test]
    fn build_detectors_respects_config() {
        let config = NailbiteConfig::load("config.yaml").unwrap();
        let app = App::new(config);
        let detectors = app.build_detectors();
        // Default config enables nail_biting and nail_picking.
        assert_eq!(detectors.len(), 2);
    }

    #[test]
    fn build_tracker_creates_per_behavior_trackers() {
        let config = NailbiteConfig::load("config.yaml").unwrap();
        let app = App::new(config);
        let detectors = app.build_detectors();
        let tracker = app.build_tracker(&detectors);
        // Should have a tracker for each detector.
        assert!(tracker.phase_of(BfrbType::NailBiting).is_some());
        assert!(tracker.phase_of(BfrbType::NailPicking).is_some());
        assert!(tracker.phase_of(BfrbType::HairPulling).is_none());
    }

    #[test]
    fn build_actions_from_config() {
        let config = NailbiteConfig::load("config.yaml").unwrap();
        let app = App::new(config);
        let (popup_tx, _popup_rx) = crossbeam_channel::unbounded();
        let actions = app.build_actions(&popup_tx);
        // Default config enables sound, notification, and popup (not webhook).
        assert!(actions.len() >= 2);
    }
}
