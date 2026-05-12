//! Camera control commands.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::{Emitter, State};
use tracing::{debug, info, warn};

use crate::actions::types::Action;
use crate::camera::CameraCapture;
use crate::config::CameraRole;
use crate::detection::types::{DetectionEvent, FaceDetection};
use crate::tray::{apply_tray_state, TrayState};
use crate::errors::NailbiteError;
use crate::inference::face_detection::FaceDetector;
use crate::inference::face_mesh::FaceMesher;
use crate::inference::hand_landmark::HandLandmarkBackend;
use crate::inference::palm_detection::PalmDetector;
use crate::inference::pose_landmark::PoseLandmarker;
use crate::pipeline::smooth_face_landmarks_with_grace;
use crate::state::AppState;
use crate::stats::event_history::{BufferedFrame, EventTrigger};

use super::detection::{DetectionEventResult, FaceResult, HandResult, LandmarkResult, PoseResult};

/// Frame update event payload sent to frontend.
#[derive(Debug, Clone, Serialize)]
pub struct FrameUpdateEvent {
    /// Camera ID this frame is from.
    pub camera_id: String,
    /// Camera role (primary or auxiliary).
    pub role: String,
    /// Base64-encoded JPEG frame for display.
    pub frame_base64: String,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Detected hands with landmarks.
    pub hands: Vec<HandResult>,
    /// Detected face with landmarks.
    pub face: Option<FaceResult>,
    /// Detected body pose with landmarks.
    pub pose: Option<PoseResult>,
    /// New detection events (alerts) - only from primary camera.
    pub detections: Vec<DetectionEventResult>,
    /// Per-detector signals computed for this frame, before any temporal
    /// confirmation. Lets the frontend render a live "why" panel.
    pub current_signals: Vec<crate::detection::types::DetectionExplanation>,
    /// Whether an alert is currently active.
    pub alert_active: bool,
    /// Whether detection is paused.
    pub paused: bool,
    /// Frame timestamp (ms since start).
    pub timestamp_ms: u64,
}

/// Individual camera status.
#[derive(Debug, Clone, Serialize)]
pub struct CameraSourceStatus {
    pub id: String,
    pub device: String,
    pub role: String,
    pub width: u32,
    pub height: u32,
    pub running: bool,
}

/// Camera status response for all cameras.
#[derive(Debug, Clone, Serialize)]
pub struct CameraStatus {
    pub cameras: Vec<CameraSourceStatus>,
    pub detection_running: bool,
    pub inference_fps: u32,
}

/// Start all configured cameras and the detection loop.
#[tauri::command]
pub fn start_camera(state: State<'_, Arc<AppState>>) -> Result<(), NailbiteError> {
    // Check if already running
    if state.detection_running.load(Ordering::Relaxed) {
        info!("Detection already running");
        return Ok(());
    }

    let config = state.config.read().clone();
    let inference_fps = config.camera.inference_fps.max(1);
    let preview_fps = config.camera.preview_fps.max(inference_fps);

    // Start all configured cameras
    let mut cameras_guard = state.cameras.lock();

    for source in &config.camera.sources {
        info!(
            id = %source.id,
            device = %source.device,
            width = source.resolution_width,
            height = source.resolution_height,
            role = ?source.role,
            "Starting camera"
        );

        let mut camera = CameraCapture::new(
            &source.device,
            source.resolution_width,
            source.resolution_height,
        );
        camera
            .start()
            .map_err(|e| NailbiteError::Camera(format!("{}: {}", source.id, e)))?;

        cameras_guard.insert(source.id.clone(), camera);
    }

    drop(cameras_guard);
    state.detection_running.store(true, Ordering::Relaxed);

    // Clone state for the detection thread.
    // We keep `preview_fps` as the *target* loop rate; inference runs on
    // every iteration so the displayed frame and overlay always match.
    // `inference_fps` is now treated as a lower bound: if the configured
    // preview rate would exceed inference capacity it just degrades.
    let state_clone = Arc::clone(state.inner());
    let target_fps = preview_fps.max(inference_fps);
    let frame_interval = Duration::from_millis(1000 / u64::from(target_fps));
    info!(
        inference_fps,
        preview_fps,
        target_fps,
        "Detection loop scheduling"
    );

    // Spawn detection loop thread
    thread::Builder::new()
        .name("detection-loop".into())
        .spawn(move || {
            run_detection_loop(state_clone, frame_interval);
        })
        .map_err(|e| NailbiteError::Camera(format!("Failed to spawn detection thread: {}", e)))?;

    info!("All cameras and detection loop started");
    Ok(())
}

/// Stop all cameras and the detection loop.
#[tauri::command]
pub fn stop_camera(state: State<'_, Arc<AppState>>) -> Result<(), NailbiteError> {
    info!("Stopping all cameras");

    // Signal detection loop to stop
    state.detection_running.store(false, Ordering::Relaxed);

    // Stop all cameras
    let mut cameras_guard = state.cameras.lock();
    for (id, mut camera) in cameras_guard.drain() {
        info!(id = %id, "Stopping camera");
        camera.stop();
    }
    drop(cameras_guard);

    // Clear per-camera state to prevent unbounded HashMap growth (TECH-2)
    state.clear_camera_state();

    info!("All cameras stopped");
    Ok(())
}

/// Get current camera status for all cameras.
#[tauri::command]
pub fn get_camera_status(state: State<'_, Arc<AppState>>) -> CameraStatus {
    let config = state.config.read();
    let cameras_guard = state.cameras.lock();
    let detection_running = state.detection_running.load(Ordering::Relaxed);

    let cameras = config
        .camera
        .sources
        .iter()
        .map(|source| {
            let running = cameras_guard.get(&source.id).is_some_and(|c| c.is_running());
            CameraSourceStatus {
                id: source.id.clone(),
                device: source.device.clone(),
                role: format!("{:?}", source.role).to_lowercase(),
                width: source.resolution_width,
                height: source.resolution_height,
                running,
            }
        })
        .collect();

    CameraStatus {
        cameras,
        detection_running,
        inference_fps: config.camera.inference_fps,
    }
}

/// Helper to update tray icon — delegates to the shared tray module so the
/// global pause/mute commands can update the tray identically. Today's
/// detection count is passed through to the tooltip text only; the icon
/// itself stays a plain colored disc.
fn update_tray_icon(state: &AppState, tray_state: TrayState) {
    let today_count = state
        .today_detection_count
        .load(std::sync::atomic::Ordering::Relaxed);
    let count_opt = if state.config.read().general.show_detection_count {
        Some(today_count)
    } else {
        None
    };
    apply_tray_state(&state.app_handle, tray_state, count_opt);
}

/// Main detection loop running in background thread.
///
/// Runs inference + frame emit on every iteration so the displayed
/// landmarks always match the displayed frame. The interval is set by
/// `frame_interval` (target preview rate); if inference is too slow the
/// effective FPS naturally degrades.
/// Recognizes "user in front of camera" sessions from per-frame face
/// visibility, with asymmetric hysteresis so a brief head-turn doesn't
/// close a session and a quick re-entry doesn't open one. Face is the
/// canonical signal — pose can hallucinate on an empty chair, and hands
/// drifting past camera are not "the user".
struct PresenceTracker {
    present: bool,
    /// Consecutive frames observed in the OPPOSITE state — debounce.
    counter: u32,
}

impl PresenceTracker {
    fn new() -> Self {
        // Start absent so the first arrival logs a session_start instead
        // of being silently assumed.
        Self { present: false, counter: 0 }
    }

    /// Update with a fresh per-frame observation. Returns:
    /// - `Some(true)` on absent → present transition
    /// - `Some(false)` on present → absent transition
    /// - `None` otherwise
    fn update(&mut self, face_visible: bool) -> Option<bool> {
        // ≈ 0.4 s at 8 fps to accept presence; ≈ 1.5 s to confirm
        // absence so a glance away doesn't end the session.
        const PRESENT_FRAMES: u32 = 3;
        const ABSENT_FRAMES: u32 = 12;

        if face_visible == self.present {
            self.counter = 0;
            return None;
        }
        self.counter += 1;
        let threshold = if face_visible { PRESENT_FRAMES } else { ABSENT_FRAMES };
        if self.counter < threshold {
            return None;
        }
        self.present = face_visible;
        self.counter = 0;
        Some(face_visible)
    }
}

fn run_detection_loop(state: Arc<AppState>, frame_interval: Duration) {
    info!("Detection loop started");
    let start_time = Instant::now();
    let mut last_frame_time = Instant::now();
    let mut frame_counter: u64 = 0;
    let mut presence = PresenceTracker::new();

    // Set tray to green (ready) when detection loop starts
    update_tray_icon(&state, TrayState::Ready);

    // Create inference components
    let palm_detector = PalmDetector::new(Arc::clone(&state.sessions.palm_detection));
    let hand_landmarker = HandLandmarkBackend::new(
        Arc::clone(&state.sessions.hand_landmark),
        state.sessions.hand_landmark_full.clone(),
    );
    info!(backend = hand_landmarker.name(), "Hand landmark backend selected");
    let face_detector = FaceDetector::new(Arc::clone(&state.sessions.face_detection));
    let face_mesher = FaceMesher::new(Arc::clone(&state.sessions.face_mesh));
    let pose_landmarker = PoseLandmarker::new(Arc::clone(&state.sessions.pose_landmark));

    // Get camera source configs for role lookup
    let camera_sources = state.config.read().camera.sources.clone();

    while state.detection_running.load(Ordering::Relaxed) {
        // Rate limiting
        let elapsed = last_frame_time.elapsed();
        if elapsed < frame_interval {
            thread::sleep(frame_interval - elapsed);
        }
        last_frame_time = Instant::now();

        // Check if scheduled sound stop time has elapsed
        {
            let mut stop_time = state.sound_stop_time.lock();
            if let Some(scheduled) = *stop_time {
                if Instant::now() >= scheduled {
                    debug!("Sound tail period elapsed, stopping sound");
                    if let Err(e) = state.sound_action.lock().stop() {
                        warn!(error = %e, "Failed to stop sound");
                    }
                    *stop_time = None;
                }
            }
        }

        // Check if paused
        let paused = state.paused.load(Ordering::Relaxed);

        // Process each camera
        let cameras_guard = state.cameras.lock();
        let camera_ids: Vec<String> = cameras_guard.keys().cloned().collect();
        drop(cameras_guard);

        for camera_id in camera_ids {
            // Get frame from this camera
            let captured_frame = {
                let cameras_guard = state.cameras.lock();
                cameras_guard
                    .get(&camera_id)
                    .and_then(|camera| camera.get_frame())
            };

            let captured = match captured_frame {
                Some(f) => f,
                None => {
                    debug!(camera_id = %camera_id, "No frame available");
                    continue;
                }
            };

            let frame = &captured.frame;
            let timestamp = captured.timestamp;

            // Find camera role
            let camera_role = camera_sources
                .iter()
                .find(|s| s.id == camera_id)
                .map(|s| s.role)
                .unwrap_or(CameraRole::Auxiliary);

            let role_str = format!("{:?}", camera_role).to_lowercase();
            let is_primary = camera_role == CameraRole::Primary;

            if paused {
                // Still emit frame for display, but skip detection
                if let Ok(frame_base64) = frame.to_base64_jpeg() {
                    let event = FrameUpdateEvent {
                        camera_id: camera_id.clone(),
                        role: role_str,
                        frame_base64,
                        width: frame.width,
                        height: frame.height,
                        hands: vec![],
                        face: None,
                        pose: None,
                        detections: vec![],
                        current_signals: vec![],
                        alert_active: state.alert_active.load(Ordering::Relaxed),
                        paused: true,
                        timestamp_ms: elapsed_ms(&start_time),
                    };
                    let _ = state.app_handle.emit("frame-update", &event);
                }
                continue;
            }

            // Always run full inference for the primary camera. Re-emitting
            // a fresh frame paired with *cached* landmarks (the previous
            // decoupled-preview path) made the overlay visibly lag behind
            // the user's hand. Sync over smoothness — if inference is too
            // slow for `preview_fps`, the rate naturally degrades.
            // Auxiliary cameras still skip inference and just stream raw.
            if !is_primary {
                if let Ok(frame_base64) = frame.to_base64_jpeg() {
                    let event = FrameUpdateEvent {
                        camera_id: camera_id.clone(),
                        role: role_str,
                        frame_base64,
                        width: frame.width,
                        height: frame.height,
                        hands: vec![],
                        face: None,
                        pose: None,
                        detections: vec![],
                        current_signals: vec![],
                        alert_active: state.alert_active.load(Ordering::Relaxed),
                        paused: false,
                        timestamp_ms: elapsed_ms(&start_time),
                    };
                    let _ = state.app_handle.emit("frame-update", &event);
                }
                continue;
            }

            // Run inference pipeline
            let mut face: Option<FaceDetection> = None;

            // Face pipeline
            if let Ok(face_result) = face_detector.detect(frame) {
                if let Some(roi) = face_result.face_rois.first() {
                    if let Ok(mesh) = face_mesher.estimate(frame, roi) {
                        face = mesh;
                    }
                }
            }

            // Smooth face landmarks (per-camera smoothing state with grace period)
            // Grace period of 4 frames (~500ms at 8 FPS) prevents flickering
            {
                let mut prev_faces = state.prev_faces.lock();
                let mut miss_counts = state.face_miss_counts.lock();
                let mut prev_face = prev_faces.get(&camera_id).cloned();
                let mut miss_count = miss_counts.get(&camera_id).copied().unwrap_or(0);

                smooth_face_landmarks_with_grace(&mut face, &mut prev_face, &mut miss_count, 0.5, 4);

                if let Some(ref f) = face {
                    prev_faces.insert(camera_id.clone(), f.clone());
                } else {
                    prev_faces.remove(&camera_id);
                }
                miss_counts.insert(camera_id.clone(), miss_count);
            }

            // Pose pipeline - estimate body pose for skeleton visualization
            // Use a centered full-frame ROI instead of pose detection for simplicity
            // This works well for single-person webcam scenarios
            let mut pose = None;
            let full_frame_roi = [0.0_f32, 0.0_f32, 1.0_f32, 1.0_f32];
            match pose_landmarker.estimate(frame, &full_frame_roi) {
                Ok(Some(pose_det)) => {
                    debug!(
                        confidence = pose_det.confidence,
                        landmarks = pose_det.landmarks.len(),
                        "Pose detected"
                    );
                    pose = Some(pose_det);
                }
                Ok(None) => {
                    debug!("No pose detected (below threshold)");
                }
                Err(e) => {
                    warn!(error = %e, "Pose estimation error");
                }
            }

            // Hand pipeline - use both palm detection AND pose wrists for robustness
            // The tracker will handle filtering duplicates
            let mut raw_hands = Vec::new();

            // First try palm detection
            match palm_detector.detect(frame) {
                Ok(palm_result) => {
                    debug!(
                        palm_rois = palm_result.hand_rois.len(),
                        detections = palm_result.detections.len(),
                        "Palm detection results"
                    );
                    for (i, (roi, det)) in palm_result.hand_rois.iter().zip(palm_result.detections.iter()).enumerate() {
                        debug!(
                            idx = i,
                            score = det.score,
                            cx = det.bbox[0],
                            cy = det.bbox[1],
                            "Palm ROI"
                        );
                        match hand_landmarker.estimate(frame, roi) {
                            Ok(Some(hand)) => {
                                debug!(
                                    idx = i,
                                    side = ?hand.side,
                                    confidence = hand.confidence,
                                    "Hand detected from palm"
                                );
                                raw_hands.push(hand);
                            }
                            Ok(None) => {
                                debug!(idx = i, "Hand landmarker returned None for palm ROI");
                            }
                            Err(e) => {
                                warn!(idx = i, error = %e, "Hand landmarker error");
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!(error = %e, "Palm detection error");
                }
            }

            // Use pose wrists to correct hand sides (pose is more reliable for L/R)
            if let Some(ref pose_det) = pose {
                use crate::detection::types::WRIST_INDEX;
                use crate::inference::pose_landmark::landmark_index::{LEFT_WRIST, RIGHT_WRIST};

                // Get pose wrist positions if visible
                let pose_left = pose_det.landmarks.get(LEFT_WRIST)
                    .filter(|w| w.visibility >= 0.3)
                    .map(|w| (w.landmark.x, w.landmark.y));
                let pose_right = pose_det.landmarks.get(RIGHT_WRIST)
                    .filter(|w| w.visibility >= 0.3)
                    .map(|w| (w.landmark.x, w.landmark.y));

                // Correct sides for palm-detected hands based on pose wrists
                for hand in &mut raw_hands {
                    let wrist = &hand.landmarks[WRIST_INDEX];
                    let mut best_side = None;
                    let mut best_dist = 0.20_f32; // Max distance to consider

                    if let Some((lx, ly)) = pose_left {
                        let dist = ((wrist.x - lx).powi(2) + (wrist.y - ly).powi(2)).sqrt();
                        if dist < best_dist {
                            best_dist = dist;
                            best_side = Some(crate::detection::types::HandSide::Left);
                        }
                    }
                    if let Some((rx, ry)) = pose_right {
                        let dist = ((wrist.x - rx).powi(2) + (wrist.y - ry).powi(2)).sqrt();
                        if dist < best_dist {
                            best_side = Some(crate::detection::types::HandSide::Right);
                        }
                    }

                    if let Some(side) = best_side {
                        if hand.side != Some(side) {
                            debug!(
                                old_side = ?hand.side,
                                new_side = ?side,
                                "Corrected hand side from pose"
                            );
                            hand.side = Some(side);
                        }
                    }
                }
            }

            // Try pose wrists for hands not found by palm detection
            if let Some(ref pose_det) = pose {
                use crate::detection::types::WRIST_INDEX;
                use crate::inference::pose_landmark::landmark_index::{
                    LEFT_ELBOW, LEFT_WRIST, RIGHT_ELBOW, RIGHT_WRIST,
                };

                for (wrist_idx, elbow_idx, side) in [
                    (LEFT_WRIST, LEFT_ELBOW, crate::detection::types::HandSide::Left),
                    (RIGHT_WRIST, RIGHT_ELBOW, crate::detection::types::HandSide::Right),
                ] {
                    if let Some(wrist_lm) = pose_det.landmarks.get(wrist_idx) {
                        // Only use if wrist is visible enough (raised from 0.2 to 0.4
                        // to reduce phantom hands from low-visibility pose wrists;
                        // combined with confirmation delay, this filters single-frame ghosts)
                        if wrist_lm.visibility < 0.4 {
                            continue;
                        }

                        // Check if any existing hand detection has a wrist near this pose wrist
                        let dominated = raw_hands.iter().any(|h| {
                            let hand_wrist = &h.landmarks[WRIST_INDEX];
                            let dx = hand_wrist.x - wrist_lm.landmark.x;
                            let dy = hand_wrist.y - wrist_lm.landmark.y;
                            let dist = (dx * dx + dy * dy).sqrt();
                            dist < 0.15 // If wrists are within 15% of frame, consider it the same hand
                        });

                        if dominated {
                            debug!(
                                wrist = ?side,
                                "Pose wrist already covered by palm detection"
                            );
                            continue;
                        }

                        // Shift the ROI center past the wrist along the
                        // elbow→wrist direction so the box covers the actual
                        // hand (palm + fingers), not half forearm + wrist.
                        // Without this, the landmark model gets a wrist-
                        // centered patch and tends to collapse the predicted
                        // landmarks around the wrist itself.
                        let elbow_lm = pose_det.landmarks.get(elbow_idx);
                        let forearm_visible = elbow_lm
                            .is_some_and(|e| e.visibility >= 0.3);

                        let (roi_cx, roi_cy, roi_size) = if forearm_visible {
                            #[allow(clippy::expect_used)]
                            let elbow = elbow_lm.expect("checked above");
                            let dx = wrist_lm.landmark.x - elbow.landmark.x;
                            let dy = wrist_lm.landmark.y - elbow.landmark.y;
                            let forearm_len = (dx * dx + dy * dy).sqrt().max(1e-3);
                            // A typical hand spans ~70% of the forearm length;
                            // shift the ROI center half a hand past the wrist.
                            let hand_span = (forearm_len * 0.7).clamp(0.10, 0.40);
                            let shift = hand_span * 0.5;
                            let cx = wrist_lm.landmark.x + (dx / forearm_len) * shift;
                            let cy = wrist_lm.landmark.y + (dy / forearm_len) * shift;
                            // Box big enough to also tolerate a flexed wrist.
                            let size = (hand_span * 1.6).clamp(0.25, 0.45);
                            (cx, cy, size)
                        } else {
                            // No elbow → fall back to a wrist-centered ROI.
                            (wrist_lm.landmark.x, wrist_lm.landmark.y, 0.35_f32)
                        };

                        let roi = [
                            (roi_cx - roi_size / 2.0).max(0.0),
                            (roi_cy - roi_size / 2.0).max(0.0),
                            (roi_cx + roi_size / 2.0).min(1.0),
                            (roi_cy + roi_size / 2.0).min(1.0),
                        ];

                        debug!(
                            wrist = ?side,
                            visibility = wrist_lm.visibility,
                            x = wrist_lm.landmark.x,
                            y = wrist_lm.landmark.y,
                            "Trying pose wrist detection"
                        );

                        if let Ok(Some(mut hand)) = hand_landmarker.estimate(frame, &roi) {
                            hand.side = Some(side);
                            debug!(
                                wrist = ?side,
                                confidence = hand.confidence,
                                "Pose wrist detected hand"
                            );
                            raw_hands.push(hand);
                        }
                    }
                }
            }

            let raw_hand_count = raw_hands.len();

            // Log raw hand positions before tracking
            for (i, h) in raw_hands.iter().enumerate() {
                let wrist = &h.landmarks[crate::detection::types::WRIST_INDEX];
                debug!(
                    idx = i,
                    side = ?h.side,
                    confidence = h.confidence,
                    wrist_x = wrist.x,
                    wrist_y = wrist.y,
                    "Raw hand detection"
                );
            }

            // Run through hand tracker for temporal consistency
            // This filters duplicate detections and maintains consistent hand identity
            let hands = {
                let mut trackers = state.hand_trackers.lock();
                let tracker = trackers.entry(camera_id.clone()).or_insert_with(|| {
                    let tuning = &state.config.read().detection.tracking;
                    crate::detection::hand_tracker::HandTracker::with_config(
                        crate::detection::hand_tracker::TrackingConfig {
                            smoothing_alpha: tuning.smoothing_alpha,
                            grace_frames: tuning.grace_frames,
                            confirmation_frames: tuning.confirmation_frames,
                            new_hand_confidence: tuning.new_hand_confidence,
                            existing_hand_confidence: tuning.existing_hand_confidence,
                        },
                    )
                });
                tracker.update(raw_hands)
            };

            debug!(
                camera_id = %camera_id,
                raw_hands = raw_hand_count,
                tracked_hands = hands.len(),
                hands_sides = ?hands.iter().map(|h| h.side).collect::<Vec<_>>(),
                "Hand tracking result"
            );

            // Structured per-frame telemetry for flickering analysis
            // Parse with scripts/analyze_flickering.py
            {
                let trackers = state.hand_trackers.lock();
                if let Some(tracker) = trackers.get(&camera_id) {
                    debug!(
                        message = "FRAME_TELEMETRY",
                        frame_ts = elapsed_ms(&start_time),
                        camera_id = %camera_id,
                        raw_hand_count = raw_hand_count,
                        tracked_hand_count = hands.len(),
                        face_detected = face.is_some(),
                        pose_detected = pose.is_some(),
                        left_miss_count = tracker.left_miss_count().unwrap_or(255),
                        right_miss_count = tracker.right_miss_count().unwrap_or(255),
                        left_consecutive = tracker.left_consecutive_detections().unwrap_or(0),
                        right_consecutive = tracker.right_consecutive_detections().unwrap_or(0),
                    );
                }
            }

            // Only run behavior detection on primary camera
            let mut detection_results = Vec::new();
            // Collect per-detector explanations for this frame so they can
            // be saved into event history alongside the frame and emitted on
            // the live `frame-update` event for the signals panel.
            let mut frame_explanations: Vec<
                crate::detection::types::DetectionExplanation,
            > = Vec::new();
            if is_primary {
                // Create frame analysis for detection
                // Uses tracked hands from primary person
                let analysis = crate::detection::types::FrameAnalysis {
                    timestamp,
                    camera_id: Arc::from(camera_id.as_str()),
                    hands: hands.clone(),
                    face: face.clone(),
                    raw_frame: None,
                };

                // Run behavior detection
                let mut detection_results_tuples: Vec<(
                    crate::detection::types::BfrbType,
                    Option<f32>,
                    Option<crate::detection::types::DetectionExplanation>,
                )> = Vec::new();
                for detector in state.detectors.read().iter() {
                    let (confidence, explanation) =
                        match detector.analyze_frame_explained(&analysis) {
                            Some((c, e)) => (Some(c), Some(e)),
                            None => (None, None),
                        };
                    if let Some(ref exp) = explanation {
                        frame_explanations.push(exp.clone());
                    }
                    detection_results_tuples
                        .push((detector.bfrb_type(), confidence, explanation));
                }

                // Drive per-loop presence from the face signal so we can
                // mark session boundaries and skip buffering empty-chair
                // pre-frames.
                match presence.update(face.is_some()) {
                    Some(true) => {
                        info!(camera = %camera_id, "User present — session start");
                        state.session_log.log_session_start();
                    }
                    Some(false) => {
                        info!(camera = %camera_id, "User absent — session end");
                        state.session_log.log_session_end();
                        // Drop pre-trigger frames so the NEXT detection
                        // doesn't pull in empty-chair frames from the
                        // away period.
                        state.event_history.lock().clear_ring_buffer();
                    }
                    None => {}
                }

                // Push this frame into the event history ring buffer with the
                // explanations we just produced. Done after detection so the
                // saved trigger frame is the one whose signals fired the alert.
                // Skip while absent — there is nothing meaningful to record.
                if presence.present {
                    let mut history = state.event_history.lock();
                    if history.is_enabled() {
                        history.push_frame(BufferedFrame {
                            frame: frame.clone(),
                            hands: hands.clone(),
                            face: face.clone(),
                            pose: pose.clone(),
                            explanations: frame_explanations.clone(),
                            timestamp_ms: elapsed_ms(&start_time),
                        });
                    }
                }

                // Update tracker
                let events;
                let was_alerting;
                let is_alerting;
                {
                    let mut tracker = state.tracker.write();
                    was_alerting = tracker.any_alerting();
                    events = tracker.update_with_explanations(
                        &detection_results_tuples,
                        timestamp,
                        &camera_id,
                    );
                    is_alerting = tracker.any_alerting();
                }

                // Handle new detection events
                for event in &events {
                    info!(bfrb = %event.bfrb_type, confidence = event.confidence, camera = %camera_id, "BFRB detected");

                    *state.current_bfrb.lock() = Some(event.bfrb_type);
                    *state.current_confidence.lock() = Some(event.confidence);
                    state.alert_active.store(true, Ordering::Relaxed);

                    // Try to start an event-history capture FIRST. The
                    // recorder rejects this if a previous capture is still
                    // collecting post-frames; in that case we also skip
                    // bumping the today-counter so the badge stays aligned
                    // with the events that will actually land on disk.
                    let event_saved = state.event_history.lock().trigger_with_explanation(
                        EventTrigger::Detection,
                        Some(event.bfrb_type),
                        Some(event.confidence),
                        event.explanation.clone(),
                    );

                    // Bump and broadcast the today counter so the UI badge
                    // / tray tooltip stay in sync without polling — only
                    // when the event actually saved.
                    let today_count = if event_saved {
                        let n = state.bump_today_detection_count();
                        let _ = state.app_handle.emit(
                            "detection-count",
                            serde_json::json!({ "count": n }),
                        );
                        n
                    } else {
                        state
                            .today_detection_count
                            .load(Ordering::Relaxed)
                    };
                    let show_count =
                        state.config.read().general.show_detection_count;
                    if show_count {
                        apply_tray_state(
                            &state.app_handle,
                            TrayState::Detecting,
                            Some(today_count),
                        );
                    }

                    // Desktop notification with TP/FP buttons. Spawned in a
                    // dedicated thread because notify-rust's wait-for-action
                    // blocks until the user clicks or the timeout expires.
                    //
                    // Skip if a previous notification is still on screen
                    // waiting for a verdict — back-to-back detections (the
                    // tracker briefly flickering in and out of alert) used
                    // to spawn a second toast on top of the first.
                    let notif_cfg = state.config.read().actions.notification.clone();
                    let notif_pending = state.active_notification_id.lock().is_some();
                    if notif_cfg.enabled && !notif_pending {
                        let state_for_notif = Arc::clone(&state);
                        let bfrb = event.bfrb_type;
                        // Build a zoomed-in crop of the action area on the
                        // current frame so the notification shows what the
                        // user was just doing. Best-effort: failures fall back
                        // to a notification with no image.
                        let image_path = build_alert_image(
                            frame,
                            &hands,
                            face.as_ref(),
                            event.explanation.as_ref(),
                        );
                        std::thread::spawn(move || {
                            spawn_alert_notification(
                                state_for_notif,
                                bfrb,
                                notif_cfg.timeout_ms,
                                image_path,
                            );
                        });
                    }

                    let det_event = DetectionEvent {
                        bfrb_type: event.bfrb_type,
                        confidence: event.confidence,
                        started_at: event.started_at,
                        duration: event.duration,
                        camera_id: camera_id.clone(),
                        explanation: event.explanation.clone(),
                    };

                    // Start sound alert (skipped while runtime mute is on)
                    if !state.muted.load(Ordering::Relaxed) {
                        if let Err(e) = state.sound_action.lock().start(&det_event) {
                            warn!(error = %e, "Failed to start sound alert");
                        }
                    }

                    // Send webhook notification if configured (ARCH-10)
                    if let Some(ref mut webhook) = *state.webhook_action.lock() {
                        if let Err(e) = webhook.start(&det_event) {
                            warn!(error = %e, "Failed to send webhook");
                        }
                    }

                    // (Event-history trigger was already issued above so the
                    // today-counter only increments when an event saves.)
                    let _ = event_saved;

                    // Clear any scheduled stop time since we're detecting again
                    *state.sound_stop_time.lock() = None;

                    // Update tray to red (detecting)
                    update_tray_icon(&state, TrayState::Detecting);

                    let _ = state.app_handle.emit(
                        "bfrb-detected",
                        serde_json::json!({
                            "bfrb_type": event.bfrb_type.as_str(),
                            "confidence": event.confidence,
                            "camera_id": camera_id,
                            "explanation": event.explanation,
                        }),
                    );

                    detection_results.push(DetectionEventResult {
                        bfrb_type: event.bfrb_type.as_str().to_string(),
                        confidence: event.confidence,
                        timestamp: chrono::Utc::now().to_rfc3339(),
                        explanation: event.explanation.clone(),
                        event_id: None,
                    });
                }

                // Handle alert auto-stop
                if was_alerting && !is_alerting {
                    info!("BFRB behavior stopped, scheduling sound stop in 0.5s");
                    let bfrb_type = state.current_bfrb.lock().take();
                    *state.current_confidence.lock() = None;
                    state.alert_active.store(false, Ordering::Relaxed);

                    // Schedule sound stop 0.5s from now (tail)
                    *state.sound_stop_time.lock() = Some(Instant::now() + Duration::from_millis(500));

                    // Update tray back to green (ready)
                    update_tray_icon(&state, TrayState::Ready);

                    let _ = state.app_handle.emit(
                        "alert-ended",
                        serde_json::json!({
                            "bfrb_type": bfrb_type.map(|b| b.as_str()).unwrap_or_default(),
                        }),
                    );
                }
            }

            // Encode frame to JPEG for frontend display
            let frame_base64 = match frame.to_base64_jpeg() {
                Ok(b64) => b64,
                Err(e) => {
                    warn!(error = %e, camera_id = %camera_id, "Failed to encode frame to JPEG");
                    continue;
                }
            };

            // Build the IPC-shaped detection results once so we can both
            // emit them and cache them for the in-between preview frames.
            let hands_result: Vec<HandResult> = hands
                .iter()
                .map(|h| HandResult {
                    landmarks: h.landmarks.iter().map(LandmarkResult::from).collect(),
                    handedness: h
                        .side
                        .map(|s| format!("{:?}", s).to_lowercase())
                        .unwrap_or_else(|| "unknown".to_string()),
                    confidence: h.confidence,
                })
                .collect();
            let face_result = face.as_ref().map(|f| FaceResult {
                landmarks: f.landmarks.iter().map(LandmarkResult::from).collect(),
                confidence: f.confidence,
            });
            let pose_result = pose.as_ref().map(PoseResult::from);

            // Emit frame update event
            let event = FrameUpdateEvent {
                camera_id: camera_id.clone(),
                role: role_str,
                frame_base64,
                width: frame.width,
                height: frame.height,
                hands: hands_result,
                face: face_result,
                pose: pose_result,
                detections: detection_results,
                current_signals: frame_explanations,
                alert_active: state.alert_active.load(Ordering::Relaxed),
                paused: false,
                timestamp_ms: elapsed_ms(&start_time),
            };

            if let Err(e) = state.app_handle.emit("frame-update", &event) {
                warn!(error = %e, camera_id = %camera_id, "Failed to emit frame-update event");
            }
        }
        frame_counter = frame_counter.wrapping_add(1);
    }

    // Cleanup when loop stops
    if let Err(e) = state.sound_action.lock().stop() {
        warn!(error = %e, "Failed to stop sound on loop exit");
    }
    *state.sound_stop_time.lock() = None;
    update_tray_icon(&state, TrayState::NotReady);

    info!("Detection loop stopped");
}

/// Get elapsed milliseconds as u64, saturating on overflow.
/// Crop a zoomed-in PNG of the contributing action area and write it to a
/// temp file. Returns the path on success.
///
/// The crop is centered on the contributing fingertip (and its partner —
/// either the mouth for nail biting or the partner fingertip for nail
/// picking), padded so the relevant skin/face region is visible, then
/// upscaled to ~256 px for a notification-sized thumbnail.
fn build_alert_image(
    frame: &crate::frame::Frame,
    hands: &[crate::detection::types::HandDetection],
    face: Option<&crate::detection::types::FaceDetection>,
    explanation: Option<&crate::detection::types::DetectionExplanation>,
) -> Option<std::path::PathBuf> {
    use crate::detection::types::{BfrbType, INNER_LIP_INDICES};
    use image::{ImageBuffer, Rgb};

    let exp = explanation?;
    // Collect the (x, y) points (in [0,1]) the crop must contain. Keep the
    // set tight — just the contributing fingertip + its partner (mouth or
    // other tip) — so the crop frames the action itself, not the whole
    // hand. A minimum-side guarantee below prevents extreme close-ups when
    // the contact is small.
    let mut points: Vec<(f32, f32)> = Vec::new();

    let strongest = exp
        .hands
        .iter()
        .max_by(|a, b| {
            a.confidence
                .partial_cmp(&b.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;

    let from_hand = hands.get(strongest.hand_index)?;
    let tip_idx = strongest.contributing_fingertip.unwrap_or(8);
    let tip = from_hand.landmarks.get(tip_idx)?;
    points.push((tip.x, tip.y));

    match exp.bfrb_type {
        BfrbType::NailBiting => {
            if let Some(face) = face {
                // Centre of the inner lip contour.
                let mut sum_x = 0.0_f32;
                let mut sum_y = 0.0_f32;
                let mut n = 0u32;
                for &i in &INNER_LIP_INDICES {
                    if let Some(lm) = face.landmarks.get(i) {
                        sum_x += lm.x;
                        sum_y += lm.y;
                        n += 1;
                    }
                }
                if n > 0 {
                    points.push((sum_x / n as f32, sum_y / n as f32));
                }
            }
        }
        BfrbType::NailPicking => {
            if let Some(partner_idx) = strongest.partner_fingertip {
                // For inter-hand picking, the partner tip is on the *other*
                // hand. For same-hand picking it's the same hand.
                let partner_hand_idx = exp
                    .hands
                    .iter()
                    .find(|other| {
                        !std::ptr::eq(*other, strongest)
                            && other.hand_index != strongest.hand_index
                            && (other.normalized_distance
                                - strongest.normalized_distance)
                                .abs()
                                < 1e-4
                    })
                    .map(|s| s.hand_index)
                    .unwrap_or(strongest.hand_index);
                if let Some(partner_hand) = hands.get(partner_hand_idx) {
                    if let Some(p) = partner_hand.landmarks.get(partner_idx) {
                        points.push((p.x, p.y));
                    }
                }
            }
        }
        _ => {}
    }

    if points.is_empty() {
        return None;
    }

    // Compute the bounding box in *pixel* space so the resulting crop is
    // square in pixels too. Doing this in normalized 0..1 coordinates and
    // then multiplying by (width, height) silently produces a 4:3 rectangle
    // on a 640×480 frame, which the 512×512 resize would squash.
    let fw = frame.width as f32;
    let fh = frame.height as f32;
    let smaller_dim = fw.min(fh);

    let mut min_px = f32::INFINITY;
    let mut min_py = f32::INFINITY;
    let mut max_px = f32::NEG_INFINITY;
    let mut max_py = f32::NEG_INFINITY;
    for &(nx, ny) in &points {
        let x = nx * fw;
        let y = ny * fh;
        if x < min_px { min_px = x; }
        if y < min_py { min_py = y; }
        if x > max_px { max_px = x; }
        if y > max_py { max_py = y; }
    }

    // Padding + minimum side, both expressed as a fraction of the frame's
    // smaller dimension so the framing feels consistent regardless of the
    // input resolution.
    let pad_px = 0.07_f32 * smaller_dim;
    let min_side_px = 0.40_f32 * smaller_dim;
    let w_px = (max_px - min_px).max(0.0);
    let h_px = (max_py - min_py).max(0.0);
    let mut side_px = w_px.max(h_px) + pad_px * 2.0;
    if side_px < min_side_px {
        side_px = min_side_px;
    }
    if side_px > smaller_dim {
        side_px = smaller_dim;
    }

    let cx_px = ((min_px + max_px) / 2.0).clamp(side_px / 2.0, fw - side_px / 2.0);
    let cy_px = ((min_py + max_py) / 2.0).clamp(side_px / 2.0, fh - side_px / 2.0);

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let px = (cx_px - side_px / 2.0).round().max(0.0) as u32;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let py = (cy_px - side_px / 2.0).round().max(0.0) as u32;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let mut pw = side_px.round().max(1.0) as u32;
    let mut ph = pw;
    // Final guard: clamp to the actual frame (rounding could push 1px out).
    if px + pw > frame.width { pw = frame.width.saturating_sub(px); }
    if py + ph > frame.height { ph = frame.height.saturating_sub(py); }
    let side_px_u = pw.min(ph);
    pw = side_px_u;
    ph = side_px_u;
    if pw == 0 || ph == 0 {
        return None;
    }

    let raw = frame.data.clone();
    let img: ImageBuffer<Rgb<u8>, _> =
        ImageBuffer::from_raw(frame.width, frame.height, raw)?;

    // Crop and upscale to a high-DPI size. KDE/Plasma scales the image
    // down to fit the notification, so we want to ship enough pixels that
    // the result still looks crisp at any reasonable display density.
    // 1024 keeps the file under 1 MB at PNG-RGB but gives a sharp image
    // even when the notification expands to a large preview.
    let cropped = image::imageops::crop_imm(&img, px, py, pw, ph).to_image();
    let target = 1024u32;
    let resized = image::imageops::resize(
        &cropped,
        target,
        target,
        image::imageops::FilterType::CatmullRom,
    );

    // Write to a unique temp file. Use a stable name so the previous
    // image is overwritten — avoids accumulation in /tmp.
    let path = std::env::temp_dir().join(format!(
        "nailbite-alert-{}.png",
        std::process::id()
    ));
    if resized.save(&path).is_err() {
        return None;
    }
    Some(path)
}

/// Show a desktop notification with built-in `Correct` / `False positive`
/// buttons. Blocks the calling thread until the user clicks one button,
/// dismisses, or the timeout expires. On either button click:
///  1. The alert sound is stopped immediately (regardless of which choice).
///  2. The corresponding verdict is persisted to the most recently saved
///     event directory matching this `bfrb_type`.
fn spawn_alert_notification(
    state: Arc<AppState>,
    bfrb: crate::detection::types::BfrbType,
    timeout_ms: u32,
    image_path: Option<std::path::PathBuf>,
) {
    use notify_rust::{Notification, Timeout};

    let title = format!("{bfrb} detected");
    let body = "Was this correct?";

    let mut notif = Notification::new();
    notif
        .summary(&title)
        .body(body)
        .appname("Nailbite")
        .icon("dialog-warning")
        .action("verdict-tp", "Correct")
        .action("verdict-fp", "False positive")
        .timeout(Timeout::Milliseconds(timeout_ms));

    if let Some(ref p) = image_path {
        if let Some(s) = p.to_str() {
            notif.image_path(s);
        }
    }

    let res = notif.show();

    let handle = match res {
        Ok(h) => h,
        Err(e) => {
            warn!(error = %e, "Failed to show desktop notification");
            return;
        }
    };

    // Stash the notification id so the in-app modal's verdict / dismiss
    // buttons can close the notification too. We still let
    // `wait_for_action` consume the handle (it has to — that's the
    // notify-rust API for blocking on action signals).
    *state.active_notification_id.lock() = Some(handle.id());

    handle.wait_for_action(|action| {
        use tauri::Emitter;

        // Whichever button the user picked, the sound stops now.
        if let Err(e) = state.sound_action.lock().stop() {
            warn!(error = %e, "Failed to stop sound from notification action");
        }
        *state.sound_stop_time.lock() = None;

        let verdict = match action {
            "verdict-tp" => Some(crate::commands::history::Verdict::TruePositive),
            "verdict-fp" => Some(crate::commands::history::Verdict::FalsePositive),
            _ => None,
        };

        if let Some(v) = verdict {
            let history_dir = {
                let cfg = state.config.read();
                crate::paths::expand_tilde(&cfg.history.dir)
            };
            let id = crate::commands::history::record_verdict_for_recent_detection(
                &history_dir,
                bfrb.as_str(),
                v,
            );
            info!(
                bfrb = %bfrb,
                verdict = v.as_str(),
                event_id = ?id,
                "Verdict recorded from desktop notification"
            );
        }

        // Either button (or any user interaction at all) closes the
        // matching in-app AlertModal — leaving it sitting open after
        // the user already decided is just noise. We mirror what the
        // `dismiss_alert` Tauri command does so the modal's `alertActive`
        // listener flips and its 12 s linger timer is cancelled by the
        // `alert-ended` event.
        state.current_bfrb.lock().take();
        *state.current_confidence.lock() = None;
        state.alert_active.store(false, Ordering::Relaxed);
        // The notification is being torn down naturally by the user's
        // own action click — clear the stored id so a later in-app
        // dismissal doesn't try to close an already-gone notification.
        *state.active_notification_id.lock() = None;
        let _ = state.app_handle.emit(
            "alert-ended",
            serde_json::json!({ "bfrb_type": "notification_action" }),
        );
    });
}

/// Close any desktop notification we previously fired for the active
/// alert. No-op when no notification is currently outstanding.
///
/// notify-rust's `NotificationHandle::close` consumes the handle, so we
/// can't share the original handle across threads. Instead we replicate
/// the underlying behavior: build a fresh `Notification` reusing the
/// same XDG id with a near-zero timeout, which causes the notification
/// daemon to dismiss the existing notification immediately.
pub fn close_active_notification(state: &AppState) {
    let id = match state.active_notification_id.lock().take() {
        Some(i) => i,
        None => return,
    };
    use notify_rust::{Notification, Timeout};
    // Replace-by-id with a 1 ms timeout. The notification daemon
    // updates the existing notification (no new toast appears) and then
    // immediately expires it.
    let res = Notification::new()
        .summary("")
        .body("")
        .id(id)
        .timeout(Timeout::Milliseconds(1))
        .show();
    if let Err(e) = res {
        warn!(error = %e, id = id, "Failed to close active desktop notification");
    }
}

fn elapsed_ms(start: &Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}
