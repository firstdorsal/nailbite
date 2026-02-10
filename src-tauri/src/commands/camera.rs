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
use crate::tray::{generate_tray_icon, TrayState};
use crate::errors::NailbiteError;
use crate::inference::face_detection::FaceDetector;
use crate::inference::face_mesh::FaceMesher;
use crate::inference::hand_landmark::HandLandmarker;
use crate::inference::palm_detection::PalmDetector;
use crate::inference::pose_landmark::PoseLandmarker;
use crate::pipeline::smooth_face_landmarks_with_grace;
use crate::state::AppState;

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
    let inference_fps = config.camera.inference_fps;

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

    // Clone state for the detection thread
    let state_clone = Arc::clone(state.inner());
    let frame_interval = Duration::from_millis(1000 / u64::from(inference_fps));

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

/// Helper to update tray icon.
fn update_tray_icon(state: &AppState, tray_state: TrayState) {
    if let Some(tray) = state.app_handle.tray_by_id("nailbite-tray") {
        let icon = generate_tray_icon(tray_state);
        if let Err(e) = tray.set_icon(Some(icon)) {
            warn!(error = %e, "Failed to update tray icon");
        }
        let tooltip = match tray_state {
            TrayState::Ready => "Nailbite - Monitoring",
            TrayState::Detecting => "Nailbite - BFRB Detected!",
            TrayState::NotReady => "Nailbite - Camera not started",
        };
        let _ = tray.set_tooltip(Some(tooltip));
    }
}

/// Main detection loop running in background thread.
fn run_detection_loop(state: Arc<AppState>, frame_interval: Duration) {
    info!("Detection loop started");
    let start_time = Instant::now();
    let mut last_frame_time = Instant::now();

    // Set tray to green (ready) when detection loop starts
    update_tray_icon(&state, TrayState::Ready);

    // Create inference components
    let palm_detector = PalmDetector::new(Arc::clone(&state.sessions.palm_detection));
    let hand_landmarker = HandLandmarker::new(Arc::clone(&state.sessions.hand_landmark));
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
                        alert_active: state.alert_active.load(Ordering::Relaxed),
                        paused: true,
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
                    info!(
                        palm_rois = palm_result.hand_rois.len(),
                        detections = palm_result.detections.len(),
                        "Palm detection results"
                    );
                    for (i, (roi, det)) in palm_result.hand_rois.iter().zip(palm_result.detections.iter()).enumerate() {
                        info!(
                            idx = i,
                            score = det.score,
                            cx = det.bbox[0],
                            cy = det.bbox[1],
                            "Palm ROI"
                        );
                        match hand_landmarker.estimate(frame, roi) {
                            Ok(Some(hand)) => {
                                info!(
                                    idx = i,
                                    side = ?hand.side,
                                    confidence = hand.confidence,
                                    "Hand detected from palm"
                                );
                                raw_hands.push(hand);
                            }
                            Ok(None) => {
                                info!(idx = i, "Hand landmarker returned None for palm ROI");
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
                use crate::inference::pose_landmark::landmark_index::{LEFT_WRIST, RIGHT_WRIST};

                for (wrist_idx, side) in [
                    (LEFT_WRIST, crate::detection::types::HandSide::Left),
                    (RIGHT_WRIST, crate::detection::types::HandSide::Right),
                ] {
                    if let Some(wrist_lm) = pose_det.landmarks.get(wrist_idx) {
                        // Only use if wrist is visible enough
                        if wrist_lm.visibility < 0.2 {
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

                        // Create ROI around pose wrist
                        let roi_size = 0.35;
                        let roi = [
                            (wrist_lm.landmark.x - roi_size / 2.0).max(0.0),
                            (wrist_lm.landmark.y - roi_size / 2.0).max(0.0),
                            (wrist_lm.landmark.x + roi_size / 2.0).min(1.0),
                            (wrist_lm.landmark.y + roi_size / 2.0).min(1.0),
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
                info!(
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
                let tracker = trackers.entry(camera_id.clone()).or_default();
                tracker.update(raw_hands)
            };

            info!(
                camera_id = %camera_id,
                raw_hands = raw_hand_count,
                tracked_hands = hands.len(),
                hands_sides = ?hands.iter().map(|h| h.side).collect::<Vec<_>>(),
                "Hand tracking result"
            );

            // Only run behavior detection on primary camera
            let mut detection_results = Vec::new();
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
                )> = Vec::new();
                for detector in state.detectors.read().iter() {
                    let confidence = detector.analyze_frame(&analysis);
                    detection_results_tuples.push((detector.bfrb_type(), confidence));
                }

                // Update tracker
                let events;
                let was_alerting;
                let is_alerting;
                {
                    let mut tracker = state.tracker.write();
                    was_alerting = tracker.any_alerting();
                    events = tracker.update(&detection_results_tuples, timestamp, &camera_id);
                    is_alerting = tracker.any_alerting();
                }

                // Handle new detection events
                for event in &events {
                    info!(bfrb = %event.bfrb_type, confidence = event.confidence, camera = %camera_id, "BFRB detected");

                    *state.current_bfrb.lock() = Some(event.bfrb_type);
                    *state.current_confidence.lock() = Some(event.confidence);
                    state.alert_active.store(true, Ordering::Relaxed);

                    let det_event = DetectionEvent {
                        bfrb_type: event.bfrb_type,
                        confidence: event.confidence,
                        started_at: event.started_at,
                        duration: event.duration,
                        camera_id: camera_id.clone(),
                    };

                    // Start sound alert
                    if let Err(e) = state.sound_action.lock().start(&det_event) {
                        warn!(error = %e, "Failed to start sound alert");
                    }

                    // Send webhook notification if configured (ARCH-10)
                    if let Some(ref mut webhook) = *state.webhook_action.lock() {
                        if let Err(e) = webhook.start(&det_event) {
                            warn!(error = %e, "Failed to send webhook");
                        }
                    }

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
                        }),
                    );

                    detection_results.push(DetectionEventResult {
                        bfrb_type: event.bfrb_type.as_str().to_string(),
                        confidence: event.confidence,
                        timestamp: chrono::Utc::now().to_rfc3339(),
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

            // Emit frame update event
            let event = FrameUpdateEvent {
                camera_id: camera_id.clone(),
                role: role_str,
                frame_base64,
                width: frame.width,
                height: frame.height,
                hands: hands
                    .iter()
                    .map(|h| HandResult {
                        landmarks: h.landmarks.iter().map(LandmarkResult::from).collect(),
                        handedness: h
                            .side
                            .map(|s| format!("{:?}", s).to_lowercase())
                            .unwrap_or_else(|| "unknown".to_string()),
                        confidence: h.confidence,
                    })
                    .collect(),
                face: face.as_ref().map(|f| FaceResult {
                    landmarks: f.landmarks.iter().map(LandmarkResult::from).collect(),
                    confidence: f.confidence,
                }),
                pose: pose.as_ref().map(PoseResult::from),
                detections: detection_results,
                alert_active: state.alert_active.load(Ordering::Relaxed),
                paused: false,
                timestamp_ms: elapsed_ms(&start_time),
            };

            if let Err(e) = state.app_handle.emit("frame-update", &event) {
                warn!(error = %e, camera_id = %camera_id, "Failed to emit frame-update event");
            }
        }
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
fn elapsed_ms(start: &Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}
