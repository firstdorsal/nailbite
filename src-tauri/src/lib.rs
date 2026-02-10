//! Nailbite Tauri application library.

use std::sync::Arc;

use tauri::menu::{MenuBuilder, MenuItemBuilder};
use tauri::tray::TrayIconBuilder;
use tauri::{Emitter, Manager};

use crate::tray::{generate_tray_icon, TrayState};
use tracing::info;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

pub mod actions;
#[cfg(target_os = "linux")]
pub mod camera;
pub mod commands;
pub mod config;
pub mod detection;
pub mod errors;
pub mod exercises;
pub mod frame;
pub mod inference;
pub mod paths;
pub mod pipeline;
pub mod state;
pub mod stats;
pub mod training;
pub mod tray;

use crate::config::NailbiteConfig;
use crate::detection::behaviors::{nail_biting::NailBitingDetector, nail_picking::NailPickingDetector, BehaviorDetector};
use crate::detection::tracker::{BehaviorTracker, DetectionTracker};
use crate::inference::session::ModelSessions;
use crate::state::AppState;

/// Run the Tauri application.
pub fn run() {
    // Initialize tracing
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();

    info!("Starting Nailbite Tauri application");

    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .setup(|app| {
            info!("Tauri setup starting");

            // Load configuration
            let config = NailbiteConfig::load("config.yaml").unwrap_or_else(|e| {
                tracing::warn!(error = %e, "Failed to load config, using defaults");
                NailbiteConfig::default()
            });

            info!(log_level = %config.general.log_level, "Configuration loaded");

            // Ensure models are downloaded
            if let Err(e) = crate::inference::model_downloader::ensure_models(&config.models) {
                tracing::error!(error = %e, "Failed to ensure models are downloaded");
                // Show error dialog and exit gracefully instead of panicking (TECH-4)
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.emit("fatal-error", serde_json::json!({
                        "title": "Model Download Failed",
                        "message": format!("Failed to download required ONNX models: {e}\n\nPlease check your internet connection and try again."),
                    }));
                }
                return Err(Box::new(e));
            }

            // Initialize ONNX sessions
            let sessions = match ModelSessions::load(&config.models, &config.ort) {
                Ok(s) => Arc::new(s),
                Err(e) => {
                    tracing::error!(error = %e, "Failed to initialize ONNX sessions");
                    // Show error dialog and exit gracefully instead of panicking (TECH-4)
                    if let Some(window) = app.get_webview_window("main") {
                        let _ = window.emit("fatal-error", serde_json::json!({
                            "title": "Model Initialization Failed",
                            "message": format!("Failed to load ONNX models: {e}\n\nPlease ensure the model files exist and are valid."),
                        }));
                    }
                    return Err(Box::new(e));
                }
            };

            info!("ONNX model sessions initialized");

            // Build detectors
            let mut detectors: Vec<Box<dyn BehaviorDetector>> = Vec::new();

            if config.detection.behaviors.nail_biting.enabled {
                detectors.push(Box::new(NailBitingDetector::new(
                    &config.detection.behaviors.nail_biting,
                    config.detection.false_positive.chin_rest_suppression,
                    config.detection.false_positive.typing_suppression,
                )));
                info!("Nail biting detector enabled");
            }

            if config.detection.behaviors.nail_picking.enabled {
                detectors.push(Box::new(NailPickingDetector::new(
                    &config.detection.behaviors.nail_picking,
                    config.detection.false_positive.typing_suppression,
                )));
                info!("Nail picking detector enabled");
            }

            // Build trackers - one per enabled behavior
            let mut trackers = Vec::new();
            if config.detection.behaviors.nail_biting.enabled {
                trackers.push(BehaviorTracker::new(
                    crate::detection::types::BfrbType::NailBiting,
                    std::time::Duration::from_millis(config.detection.temporal.window_ms),
                    config.detection.temporal.positive_ratio,
                    config.detection.behaviors.nail_biting.confidence_threshold,
                    std::time::Duration::from_secs(config.general.cooldown_seconds),
                ));
            }
            if config.detection.behaviors.nail_picking.enabled {
                trackers.push(BehaviorTracker::new(
                    crate::detection::types::BfrbType::NailPicking,
                    std::time::Duration::from_millis(config.detection.temporal.window_ms),
                    config.detection.temporal.positive_ratio,
                    config.detection.behaviors.nail_picking.confidence_threshold,
                    std::time::Duration::from_secs(config.general.cooldown_seconds),
                ));
            }
            let tracker = DetectionTracker::new(trackers);

            // Create app state
            let app_state = Arc::new(AppState::new(
                sessions,
                detectors,
                tracker,
                config,
                app.handle().clone(),
            ));

            // Register state with Tauri
            app.manage(app_state);

            // Create system tray icon with menu
            let quit_item = MenuItemBuilder::new("Quit").id("quit").build(app)?;
            let show_item = MenuItemBuilder::new("Show Window").id("show").build(app)?;
            let pause_item = MenuItemBuilder::new("Pause Detection").id("pause").build(app)?;

            let menu = MenuBuilder::new(app)
                .item(&show_item)
                .item(&pause_item)
                .separator()
                .item(&quit_item)
                .build()?;

            // Start with yellow (not ready) icon until camera is started
            let initial_icon = generate_tray_icon(TrayState::NotReady);
            let _tray = TrayIconBuilder::with_id("nailbite-tray")
                .icon(initial_icon)
                .tooltip("Nailbite - BFRB Detection (Camera not started)")
                .menu(&menu)
                .on_menu_event(|app, event| {
                    match event.id().as_ref() {
                        "quit" => {
                            info!("Quit requested from tray");
                            app.exit(0);
                        }
                        "show" => {
                            if let Some(window) = app.get_webview_window("main") {
                                let _ = window.show();
                                let _ = window.set_focus();
                            }
                        }
                        "pause" => {
                            info!("Pause/Resume requested from tray");
                            // Toggle pause state
                            if let Some(state) = app.try_state::<Arc<AppState>>() {
                                let paused = state.paused.load(std::sync::atomic::Ordering::Relaxed);
                                state.paused.store(!paused, std::sync::atomic::Ordering::Relaxed);
                                info!(paused = !paused, "Detection pause toggled");
                            }
                        }
                        _ => {}
                    }
                })
                .on_tray_icon_event(|tray, event| {
                    if let tauri::tray::TrayIconEvent::Click { button: tauri::tray::MouseButton::Left, .. } = event {
                        let app = tray.app_handle();
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                })
                .build(app)?;

            info!("System tray icon created");

            // Configure media permissions for WebKitGTK on Linux
            #[cfg(target_os = "linux")]
            {
                use webkit2gtk::{PermissionRequestExt, SettingsExt, WebViewExt};

                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.with_webview(|webview| {
                        let wv: webkit2gtk::WebView = webview.inner().clone();

                        // Enable media stream support in WebKit settings
                        if let Some(settings) = wv.settings() {
                            settings.set_enable_media_stream(true);
                            settings.set_enable_mediasource(true);
                            settings.set_media_playback_requires_user_gesture(false);
                            info!("WebKit media settings enabled");
                        }

                        // Connect permission handler (SECURITY-7: only allow camera)
                        wv.connect_permission_request(|_webview, request| {
                            use webkit2gtk::glib::prelude::Cast;
                            use webkit2gtk::UserMediaPermissionRequestExt;

                            // Check if this is a user media (camera/mic) request
                            if let Some(media_request) = request.downcast_ref::<webkit2gtk::UserMediaPermissionRequest>() {
                                // Only allow video (camera), deny audio-only requests
                                if media_request.is_for_video_device() {
                                    tracing::info!("WebKit camera permission request granted");
                                    request.allow();
                                } else {
                                    tracing::info!("WebKit audio-only permission request denied");
                                    request.deny();
                                }
                            } else {
                                // Deny all other permission types (geolocation, notifications, etc.)
                                tracing::info!("WebKit non-media permission request denied");
                                request.deny();
                            }
                            true
                        });

                        // Reload page to apply settings (camera request may have already failed)
                        wv.reload();
                    });
                    info!("WebKitGTK media permissions configured");
                }
            }

            info!("Nailbite Tauri setup complete");
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::detection::ensure_models,
            commands::config::get_config,
            commands::config::save_config,
            commands::exercises::get_exercise,
            commands::exercises::verify_exercise_frame,
            commands::stats::get_stats,
            commands::stats::toggle_pause,
            commands::stats::dismiss_alert,
            commands::stats::mark_missed_event,
            #[cfg(target_os = "linux")]
            commands::camera::start_camera,
            #[cfg(target_os = "linux")]
            commands::camera::stop_camera,
            #[cfg(target_os = "linux")]
            commands::camera::get_camera_status,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
