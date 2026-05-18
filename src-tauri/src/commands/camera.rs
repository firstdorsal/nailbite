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
            config.camera.controls.clone(),
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
/// Recognizes "user in front of camera" sessions from per-frame face +
/// pose signals, with asymmetric hysteresis so a brief head-turn doesn't
/// close a session and a quick re-entry doesn't open one. Two corroborating
/// signals are required — both the face detector and the pose model can
/// hallucinate on cluttered furniture in isolation, so we demand the face
/// be high-confidence AND that the torso pose landmarks (nose + shoulders)
/// also vote yes.
struct PresenceTracker {
    present: bool,
    /// Consecutive frames observed in the OPPOSITE state — debounce.
    counter: u32,
}

/// Minimum face-mesh confidence for the face vote. The mesh's own
/// internal threshold is permissive (0.5) to keep landmark display
/// stable; we want a stricter bar for "the user is here".
/// Face-mesh confidence required to vote "person here". The face mesh
/// itself only emits a result when its internal confidence is ≥ 0.5,
/// so this threshold is the *additional* margin the presence gate
/// demands. Set just above the mesh's own floor: we still want to reject
/// the rare 0.5–0.55 hallucinations on shirts/posters, but real faces
/// almost always sit at 0.7+ except during hard occlusions (e.g. hands
/// over the mouth during a bite) — exactly when we DON'T want to drop
/// presence.
const PRESENCE_FACE_CONFIDENCE: f32 = 0.55;

/// Face-confidence floor above which the face alone counts as presence,
/// no pose corroboration required. This rescues the "user is at the
/// desk but shoulders cropped" case — pose sees only the head, so a
/// shoulder-based vote permanently fails and the gate would stick
/// absent. A high-confidence face is unambiguously a person, not chair
/// hallucination noise, so we don't need a second signal.
const PRESENCE_FACE_STRONG_CONFIDENCE: f32 = 0.70;

/// Minimum pose-landmark visibility for an upper-body landmark to count
/// toward presence.
const PRESENCE_POSE_VISIBILITY: f32 = 0.5;

/// Minimum number of visible upper-body landmarks (out of nose, eyes,
/// shoulders) for the pose model to vote "person here". Pulling eyes
/// into the candidate set means a user sitting close to the desk —
/// whose shoulders are below the frame — can still get a pose vote
/// from the head region alone, instead of failing forever.
const PRESENCE_POSE_MIN_LANDMARKS: usize = 2;

/// Pose landmark indices (BlazePose / MediaPipe convention) used as
/// upper-body anchors for presence.
const POSE_NOSE: usize = 0;
const POSE_LEFT_EYE: usize = 2;
const POSE_RIGHT_EYE: usize = 5;
const POSE_LEFT_SHOULDER: usize = 11;
const POSE_RIGHT_SHOULDER: usize = 12;

/// Sentinel value stored in `AppState::active_notification_id` while a
/// notification is being built and not yet handed off to notify-rust.
/// `0` is never a real XDG notification id, so reusing the same field
/// for both states works without widening the type. The
/// claim+publish-real-id sequence happens under `Mutex`, so the
/// sentinel is observable only between the spawn-thread call and
/// `notif.show()`'s return — exactly the window that used to race.
const NOTIFICATION_CLAIM_SENTINEL: u32 = 0;

fn face_votes_present(face: &Option<FaceDetection>) -> bool {
    face.as_ref()
        .is_some_and(|f| f.confidence >= PRESENCE_FACE_CONFIDENCE)
}

fn face_votes_present_strong(face: &Option<FaceDetection>) -> bool {
    face.as_ref()
        .is_some_and(|f| f.confidence >= PRESENCE_FACE_STRONG_CONFIDENCE)
}

fn pose_votes_present(pose: &Option<crate::detection::types::PoseDetection>) -> bool {
    let Some(p) = pose else { return false; };
    let visible = [
        POSE_NOSE,
        POSE_LEFT_EYE,
        POSE_RIGHT_EYE,
        POSE_LEFT_SHOULDER,
        POSE_RIGHT_SHOULDER,
    ]
    .iter()
    .filter(|&&i| {
        p.landmarks
            .get(i)
            .is_some_and(|lm| lm.visibility >= PRESENCE_POSE_VISIBILITY)
    })
    .count();
    visible >= PRESENCE_POSE_MIN_LANDMARKS
}

impl PresenceTracker {
    fn new() -> Self {
        // Start absent so the first arrival logs a session_start instead
        // of being silently assumed.
        Self { present: false, counter: 0 }
    }

    /// True when presence is currently `true` but the per-frame vote
    /// has flipped to "no person" for at least one frame — i.e. we're
    /// mid-departure and the 6 s absent debounce hasn't expired yet.
    ///
    /// Used to gate detections off during that window: when the user is
    /// walking out of frame, the face/pose models briefly hallucinate
    /// on whatever they leave behind (chair, jacket, lamp) and we'd
    /// fire spurious BFRB events on static objects. Detection
    /// **disables** as soon as a single absent vote lands; presence
    /// itself only drops after the full debounce.
    fn is_transitioning_to_absent(&self) -> bool {
        self.present && self.counter > 0
    }

    /// Update with a fresh per-frame observation. Returns:
    /// - `Some(true)` on absent → present transition
    /// - `Some(false)` on present → absent transition
    /// - `None` otherwise
    fn update(&mut self, person_visible: bool) -> Option<bool> {
        // ≈ 1.0 s at 8 fps to accept presence (gives the face mesh and
        // pose model time to agree on a real user, rather than a frame
        // of correlated noise); ≈ 6 s to confirm absence so brief
        // occlusions (hands over face during a bite, glance away,
        // bending out of frame to reach a drink) don't end the session.
        // The cost of a slow absent transition is just a few extra
        // seconds of `Ready` tray colour — far less annoying than the
        // gate snapping off during legitimate use.
        const PRESENT_FRAMES: u32 = 8;
        const ABSENT_FRAMES: u32 = 48;

        if person_visible == self.present {
            self.counter = 0;
            return None;
        }
        self.counter += 1;
        let threshold = if person_visible { PRESENT_FRAMES } else { ABSENT_FRAMES };
        if self.counter < threshold {
            return None;
        }
        self.present = person_visible;
        self.counter = 0;
        Some(person_visible)
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

            // Snapshot the RAW face (pre-smoothing) for the presence vote.
            // The grace-period smoothing keeps a stale face alive for ~4
            // frames after the model returns None, which is exactly what
            // we *don't* want when deciding whether the user is in frame
            // — a single hallucinated face would stick around long enough
            // to keep presence "true" through the next absent debounce.
            // Smoothing still applies to `face` itself so the landmark
            // overlay stays stable.
            let raw_face = face.clone();

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

            // Pose-validated hand filter and side correction.
            //
            // Palm detection sometimes hallucinates "hands" on chair
            // edges, sleeves, lamps, mugs, etc. — anything roughly
            // hand-sized and skin-toned. The pose model gives us a
            // strong cross-check: if pose says the wrist isn't visible
            // (e.g. hand under the desk, off-screen), there shouldn't
            // be a hand here. We drop palm detections that have no
            // nearby pose wrist, and then re-use the same wrists to
            // correct L/R sides on the survivors (pose handedness is
            // more reliable than palm-detector handedness).
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

                // Hand-suppression filter: when pose is confident the
                // user's wrists are NOT visible, drop palm detections
                // entirely — they're hallucinations. When at least one
                // wrist is visible, each palm hand must sit near SOME
                // pose wrist to survive; the 0.20-of-frame radius is
                // the same threshold used below for L/R assignment.
                if pose_left.is_none() && pose_right.is_none() {
                    if !raw_hands.is_empty() {
                        debug!(
                            dropped = raw_hands.len(),
                            "Pose reports no visible wrists; dropping palm-detected hands as hallucinations"
                        );
                        raw_hands.clear();
                    }
                } else {
                    let before = raw_hands.len();
                    raw_hands.retain(|hand| {
                        let wrist = &hand.landmarks[WRIST_INDEX];
                        let near_pose = [pose_left, pose_right]
                            .iter()
                            .filter_map(|p| *p)
                            .any(|(px, py)| {
                                let d = ((wrist.x - px).powi(2)
                                    + (wrist.y - py).powi(2))
                                .sqrt();
                                d < 0.20
                            });
                        near_pose
                    });
                    let after = raw_hands.len();
                    if after < before {
                        debug!(
                            dropped = before - after,
                            kept = after,
                            "Dropped palm-detected hands with no nearby pose wrist"
                        );
                    }
                }

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

            // Wrist-proximity dedupe: two raw hands whose wrists sit within
            // ~12% of the frame from each other are almost certainly the
            // same physical hand (palm detection firing twice, or palm +
            // pose-wrist fallback both landing on the same hand). Keep the
            // higher-confidence one. This is a final guard after NMS and
            // the pose-domination check; it covers cases neither catches.
            {
                use crate::detection::types::WRIST_INDEX;
                const DUPLICATE_WRIST_DIST: f32 = 0.12;
                raw_hands.sort_by(|a, b| {
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut keep = vec![true; raw_hands.len()];
                for i in 0..raw_hands.len() {
                    if !keep.get(i).copied().unwrap_or(false) {
                        continue;
                    }
                    let Some(hand_i) = raw_hands.get(i) else { continue };
                    let Some(wrist_i) = hand_i.landmarks.get(WRIST_INDEX) else {
                        continue;
                    };
                    let wrist_i_x = wrist_i.x;
                    let wrist_i_y = wrist_i.y;
                    for j in (i + 1)..raw_hands.len() {
                        if !keep.get(j).copied().unwrap_or(false) {
                            continue;
                        }
                        let Some(hand_j) = raw_hands.get(j) else { continue };
                        let Some(wrist_j) = hand_j.landmarks.get(WRIST_INDEX) else {
                            continue;
                        };
                        let dx = wrist_i_x - wrist_j.x;
                        let dy = wrist_i_y - wrist_j.y;
                        if (dx * dx + dy * dy).sqrt() < DUPLICATE_WRIST_DIST {
                            if let Some(slot) = keep.get_mut(j) {
                                *slot = false;
                            }
                        }
                    }
                }
                let mut idx = 0;
                raw_hands.retain(|_| {
                    let k = keep.get(idx).copied().unwrap_or(true);
                    idx += 1;
                    k
                });
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

                // Per-loop presence vote. The standard path requires BOTH
                // the face mesh AND the pose model to agree — either
                // alone hallucinates on furniture and clothing. But a
                // strongly-confident face by itself (≥0.70) is enough
                // because that floor sits comfortably above the
                // hallucination band; this rescues the "user sits close
                // to the desk with shoulders cropped" case where pose
                // can't see any torso landmarks and the AND-gate would
                // otherwise stick at absent forever.
                let person_visible = face_votes_present_strong(&raw_face)
                    || (face_votes_present(&raw_face) && pose_votes_present(&pose));
                match presence.update(person_visible) {
                    Some(true) => {
                        info!(camera = %camera_id, "User present — session start");
                        state.session_log.log_session_start();
                        // Switch tray + UI back to "Monitoring" unless an
                        // alert is currently active.
                        if !state.alert_active.load(Ordering::Relaxed)
                            && !state.paused.load(Ordering::Relaxed)
                        {
                            update_tray_icon(&state, TrayState::Ready);
                        }
                        let _ = state.app_handle.emit(
                            "presence-changed",
                            serde_json::json!({ "present": true }),
                        );
                    }
                    Some(false) => {
                        info!(camera = %camera_id, "User absent — session end");
                        state.session_log.log_session_end();
                        // Drop pre-trigger frames so the NEXT detection
                        // doesn't pull in empty-chair frames from the
                        // away period.
                        state.event_history.lock().clear_ring_buffer();
                        // Drop any partial tracker state so a phantom
                        // detection mid-departure can't bleed into the
                        // next session.
                        state.tracker.write().reset_all();
                        // Flip tray + UI to the muted dark-gray "absent"
                        // state so the user can tell at a glance that
                        // detection is gated off. Skip while an alert is
                        // still active or paused — those states own the
                        // icon.
                        if !state.alert_active.load(Ordering::Relaxed)
                            && !state.paused.load(Ordering::Relaxed)
                        {
                            update_tray_icon(&state, TrayState::Absent);
                        }
                        let _ = state.app_handle.emit(
                            "presence-changed",
                            serde_json::json!({ "present": false }),
                        );
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

                // Update tracker — but only when the user is present.
                // Skipping the tracker while absent guarantees we never
                // fire an alert, send a notification, or start a capture
                // when there is no one in front of the camera; phantom
                // hand/face detections on background objects can't escalate.
                let events;
                let was_alerting;
                let is_alerting;
                {
                    let mut tracker = state.tracker.write();
                    was_alerting = tracker.any_alerting();
                    // Suppress tracker updates not only when presence is
                    // confirmed absent but ALSO while we're transitioning
                    // toward absent. The window between the user's last
                    // good frame and the 6 s absent debounce expiring is
                    // exactly when the face/pose models hallucinate on
                    // the now-empty chair, jacket, lamp etc; firing
                    // events from that window means alerts on furniture.
                    events = if presence.present
                        && !presence.is_transitioning_to_absent()
                    {
                        tracker.update_with_explanations(
                            &detection_results_tuples,
                            timestamp,
                            &camera_id,
                        )
                    } else {
                        Vec::new()
                    };
                    is_alerting = tracker.any_alerting();
                }

                // When multiple behaviors trip in the same frame, only the
                // strongest event drives notification + sound — otherwise
                // the user gets two simultaneous toasts and overlapping
                // alert sounds, even though both events refer to the same
                // physical moment.
                let alert_event = events
                    .iter()
                    .max_by(|a, b| {
                        a.confidence
                            .partial_cmp(&b.confidence)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });

                // Encode the trigger frame once for any alerts firing this
                // frame so the in-app modal can render it immediately
                // (event-history's annotated frame still arrives later and
                // replaces this raw one, but the modal must never open
                // without an image — the previous flow waited on the event
                // dir to be finalised and showed an empty box in between).
                let trigger_frame_b64: Option<String> = if !events.is_empty() {
                    frame.to_base64_jpeg().ok()
                } else {
                    None
                };

                // Only the strongest event drives user-visible state for this
                // frame: history capture, today-counter, tray, alert event.
                // When two detectors trip together (e.g. biting + picking)
                // they describe the same physical moment from different
                // angles — saving both produces two history rows at the
                // same timestamp with the second one's trigger frame
                // missing (the recorder rejects overlapping captures).
                //
                // Webhooks still fire per-event so upstream consumers can
                // see each detector's verdict — they are not user-facing.
                for event in &events {
                    info!(bfrb = %event.bfrb_type, confidence = event.confidence, camera = %camera_id, "BFRB detected");
                    let det_event = DetectionEvent {
                        bfrb_type: event.bfrb_type,
                        confidence: event.confidence,
                        started_at: event.started_at,
                        duration: event.duration,
                        camera_id: camera_id.clone(),
                        explanation: event.explanation.clone(),
                    };
                    if let Some(ref mut webhook) = *state.webhook_action.lock() {
                        if let Err(e) = webhook.start(&det_event) {
                            warn!(error = %e, "Failed to send webhook");
                        }
                    }
                    detection_results.push(DetectionEventResult {
                        bfrb_type: event.bfrb_type.as_str().to_string(),
                        confidence: event.confidence,
                        timestamp: chrono::Utc::now().to_rfc3339(),
                        explanation: event.explanation.clone(),
                        event_id: None,
                    });
                }

                if let Some(event) = alert_event {
                    *state.current_bfrb.lock() = Some(event.bfrb_type);
                    *state.current_confidence.lock() = Some(event.confidence);
                    state.alert_active.store(true, Ordering::Relaxed);

                    let event_saved = state.event_history.lock().trigger_with_explanation(
                        EventTrigger::Detection,
                        Some(event.bfrb_type),
                        Some(event.confidence),
                        event.explanation.clone(),
                    );

                    let today_count = if event_saved {
                        let n = state.bump_today_detection_count();
                        let _ = state.app_handle.emit(
                            "detection-count",
                            serde_json::json!({ "count": n }),
                        );
                        n
                    } else {
                        state.today_detection_count.load(Ordering::Relaxed)
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

                    *state.sound_stop_time.lock() = None;
                    update_tray_icon(&state, TrayState::Detecting);

                    let _ = state.app_handle.emit(
                        "bfrb-detected",
                        serde_json::json!({
                            "bfrb_type": event.bfrb_type.as_str(),
                            "confidence": event.confidence,
                            "camera_id": camera_id,
                            "explanation": event.explanation,
                            "trigger_frame_b64": trigger_frame_b64,
                        }),
                    );
                }

                // Single user-facing alert per frame (the strongest event).
                // We pulled this out of the per-event loop so two detectors
                // tripping in the same frame don't fire two desktop toasts
                // and two overlapping alert sounds.
                if let Some(event) = alert_event {
                    let notif_cfg =
                        state.config.read().actions.notification.clone();
                    // Cross-frame dedup: claim the notification slot
                    // atomically. Two detectors tripping in consecutive
                    // frames otherwise raced past the check-then-spawn
                    // window and each fired their own toast — and the
                    // second one rendered without the inline image
                    // because `build_alert_image` had already overwritten
                    // the predictable temp path. We read+write the slot
                    // under one lock so only one event-loop iteration
                    // can win the spawn; the sentinel id `0` marks
                    // "claim made, real id pending" so other threads
                    // see the slot as occupied even before notify-rust
                    // returns the daemon's handle.
                    let claimed = if notif_cfg.enabled {
                        let mut guard = state.active_notification_id.lock();
                        if guard.is_some() {
                            false
                        } else {
                            *guard = Some(NOTIFICATION_CLAIM_SENTINEL);
                            true
                        }
                    } else {
                        false
                    };
                    if claimed {
                        let image_path = build_alert_image(
                            frame,
                            &hands,
                            face.as_ref(),
                            event.explanation.as_ref(),
                        );
                        // Skip the notification entirely if we couldn't
                        // produce an image — an imageless toast was a
                        // recurring UX bug ("popup without picture").
                        // Release the claim so the next event can take
                        // the slot.
                        if let Some(path) = image_path {
                            let state_for_notif = Arc::clone(&state);
                            let bfrb = event.bfrb_type;
                            std::thread::spawn(move || {
                                spawn_alert_notification(
                                    state_for_notif,
                                    bfrb,
                                    notif_cfg.timeout_ms,
                                    Some(path),
                                );
                            });
                        } else {
                            *state.active_notification_id.lock() = None;
                            warn!(
                                bfrb = %event.bfrb_type,
                                "Could not produce alert image; skipping desktop notification"
                            );
                        }
                    }

                    if !state.muted.load(Ordering::Relaxed) {
                        let det_event = DetectionEvent {
                            bfrb_type: event.bfrb_type,
                            confidence: event.confidence,
                            started_at: event.started_at,
                            duration: event.duration,
                            camera_id: camera_id.clone(),
                            explanation: event.explanation.clone(),
                        };
                        if let Err(e) =
                            state.sound_action.lock().start(&det_event)
                        {
                            warn!(error = %e, "Failed to start sound alert");
                        }
                    }
                }

                // Handle alert auto-stop
                if was_alerting && !is_alerting {
                    info!("BFRB behavior stopped, scheduling sound stop in 0.5s");
                    let bfrb_type = state.current_bfrb.lock().take();
                    *state.current_confidence.lock() = None;
                    state.alert_active.store(false, Ordering::Relaxed);

                    // Schedule sound stop 0.5s from now (tail)
                    *state.sound_stop_time.lock() = Some(Instant::now() + Duration::from_millis(500));

                    // Deliberately do NOT close the desktop notification
                    // here — the toast is the user's window to label the
                    // event, and the behavior typically ends well before
                    // they have a chance to click. The notification's own
                    // `timeout_ms` (default 20s) governs when it goes away.

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

/// Render and save the notification-image companion for an alert.
///
/// The desktop notification daemon expects an on-disk PNG referenced
/// by path (notify-rust's `image_path`). This function builds it.
///
/// Contract: returns `Some(path)` whenever *any* image was written.
/// Internally it tries to produce a zoomed-in crop centered on the
/// contributing landmarks (so the toast frames the actual action),
/// and falls back to the raw frame on any failure (no contributing
/// points, OOB indices, image-encode failure, write failure, …).
/// The "never fire a notification without an image" contract upstream
/// depends on this fallback being honored.
///
/// Path security: writes land in [`alert_image_directory`] with file
/// mode `0o600` and a nanosecond-tagged filename, so concurrent alerts
/// do not collide and the file is not world-readable.
fn build_alert_image(
    frame: &crate::frame::Frame,
    hands: &[crate::detection::types::HandDetection],
    face: Option<&crate::detection::types::FaceDetection>,
    explanation: Option<&crate::detection::types::DetectionExplanation>,
) -> Option<std::path::PathBuf> {
    use image::{ImageBuffer, Rgb};
    use std::borrow::Cow;

    // Construct the source image *once*. Both the crop path and the
    // fallback path need it; building it twice is a ~900 KB memcpy we
    // can skip.
    let source: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(frame.width, frame.height, frame.data.clone())?;

    // Try the detailed crop first; it shares the source buffer with
    // the fallback so we don't pay a second clone if it falls through.
    let to_write: Cow<'_, ImageBuffer<Rgb<u8>, Vec<u8>>> =
        match crop_for_alert(frame, hands, face, explanation, &source) {
            Some(cropped) => Cow::Owned(cropped),
            None => Cow::Borrowed(&source),
        };

    let destination = alert_image_destination()?;
    write_png_user_only(to_write.as_ref(), &destination).ok()?;
    prune_stale_alert_images(&destination);
    Some(destination)
}

/// Render the cropped notification-thumbnail variant of the frame.
///
/// Returns `None` for any failure mode that should fall back to the
/// full-frame image: missing explanation, no contributing landmarks,
/// OOB indices, or degenerate crop geometry. Returns the cropped +
/// upscaled image otherwise; the caller is responsible for writing it
/// to disk.
fn crop_for_alert(
    frame: &crate::frame::Frame,
    hands: &[crate::detection::types::HandDetection],
    face: Option<&crate::detection::types::FaceDetection>,
    explanation: Option<&crate::detection::types::DetectionExplanation>,
    source: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
) -> Option<image::ImageBuffer<image::Rgb<u8>, Vec<u8>>> {
    use crate::detection::types::{BfrbType, INNER_LIP_INDICES};

    let exp = explanation?;
    // Collect the (x, y) points (in [0,1]) the crop must contain. Keep
    // the set tight — just the contributing fingertip + its partner
    // (mouth or other tip) — so the crop frames the action itself,
    // not the whole hand. A minimum-side guarantee below prevents
    // extreme close-ups when the contact is small.
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
    let tip_index = strongest.contributing_fingertip.unwrap_or(8);
    let tip = from_hand.landmarks.get(tip_index)?;
    points.push((tip.x, tip.y));

    match exp.bfrb_type {
        BfrbType::NailBiting => {
            if let Some(face) = face {
                // Centre of the inner lip contour.
                let mut sum_x = 0.0_f32;
                let mut sum_y = 0.0_f32;
                let mut count = 0u32;
                for &i in &INNER_LIP_INDICES {
                    if let Some(landmark) = face.landmarks.get(i) {
                        sum_x += landmark.x;
                        sum_y += landmark.y;
                        count += 1;
                    }
                }
                if count > 0 {
                    points.push((sum_x / count as f32, sum_y / count as f32));
                }
            }
        }
        BfrbType::NailPicking => {
            if let Some(partner_index) = strongest.partner_fingertip {
                // For inter-hand picking, the partner tip is on the
                // *other* hand. For same-hand picking it's the same
                // hand.
                let partner_hand_index = exp
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
                if let Some(partner_hand) = hands.get(partner_hand_index) {
                    if let Some(partner_landmark) =
                        partner_hand.landmarks.get(partner_index)
                    {
                        points.push((partner_landmark.x, partner_landmark.y));
                    }
                }
            }
        }
        _ => {}
    }

    if points.is_empty() {
        return None;
    }

    // Compute the bounding box in *pixel* space so the resulting crop
    // is square in pixels too. Doing this in normalized 0..1
    // coordinates and then multiplying by (width, height) silently
    // produces a 4:3 rectangle on a 640×480 frame, which the
    // 1024×1024 resize would squash.
    let frame_width = frame.width as f32;
    let frame_height = frame.height as f32;
    let smaller_dim = frame_width.min(frame_height);

    let mut min_x_pixels = f32::INFINITY;
    let mut min_y_pixels = f32::INFINITY;
    let mut max_x_pixels = f32::NEG_INFINITY;
    let mut max_y_pixels = f32::NEG_INFINITY;
    for &(normalized_x, normalized_y) in &points {
        let x = normalized_x * frame_width;
        let y = normalized_y * frame_height;
        if x < min_x_pixels { min_x_pixels = x; }
        if y < min_y_pixels { min_y_pixels = y; }
        if x > max_x_pixels { max_x_pixels = x; }
        if y > max_y_pixels { max_y_pixels = y; }
    }

    // Padding + minimum side, both expressed as a fraction of the
    // frame's smaller dimension so the framing feels consistent
    // regardless of the input resolution.
    let padding_pixels = 0.07_f32 * smaller_dim;
    let min_side_pixels = 0.40_f32 * smaller_dim;
    let width_pixels = (max_x_pixels - min_x_pixels).max(0.0);
    let height_pixels = (max_y_pixels - min_y_pixels).max(0.0);
    let mut side_pixels = width_pixels.max(height_pixels) + padding_pixels * 2.0;
    if side_pixels < min_side_pixels {
        side_pixels = min_side_pixels;
    }
    if side_pixels > smaller_dim {
        side_pixels = smaller_dim;
    }

    let center_x_pixels = ((min_x_pixels + max_x_pixels) / 2.0)
        .clamp(side_pixels / 2.0, frame_width - side_pixels / 2.0);
    let center_y_pixels = ((min_y_pixels + max_y_pixels) / 2.0)
        .clamp(side_pixels / 2.0, frame_height - side_pixels / 2.0);

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let pixel_x = (center_x_pixels - side_pixels / 2.0).round().max(0.0) as u32;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let pixel_y = (center_y_pixels - side_pixels / 2.0).round().max(0.0) as u32;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let mut pixel_width = side_pixels.round().max(1.0) as u32;
    let mut pixel_height = pixel_width;
    // Final guard: clamp to the actual frame (rounding could push 1px
    // out).
    if pixel_x + pixel_width > frame.width {
        pixel_width = frame.width.saturating_sub(pixel_x);
    }
    if pixel_y + pixel_height > frame.height {
        pixel_height = frame.height.saturating_sub(pixel_y);
    }
    let square_side = pixel_width.min(pixel_height);
    pixel_width = square_side;
    pixel_height = square_side;
    if pixel_width == 0 || pixel_height == 0 {
        return None;
    }

    // Crop and upscale to a high-DPI size. KDE/Plasma scales the
    // image down to fit the notification, so we want to ship enough
    // pixels that the result still looks crisp at any reasonable
    // display density. 1024 keeps the file under 1 MB at PNG-RGB but
    // gives a sharp image even when the notification expands to a
    // large preview.
    let cropped = image::imageops::crop_imm(
        source,
        pixel_x,
        pixel_y,
        pixel_width,
        pixel_height,
    )
    .to_image();
    let target = 1024u32;
    Some(image::imageops::resize(
        &cropped,
        target,
        target,
        image::imageops::FilterType::CatmullRom,
    ))
}

/// Build the absolute path the next alert image should be written to.
///
/// Layout: `$XDG_RUNTIME_DIR/nailbite/alert-{ns}.png` on Linux when
/// the runtime dir is set, otherwise `<temp>/nailbite/alert-{ns}.png`.
/// Each call returns a fresh filename (monotonic nanos + PID) so two
/// near-simultaneous alerts never race over the same path, and so
/// PID reuse across daemon restarts cannot point us at a stale file
/// a previous user might have created.
fn alert_image_destination() -> Option<std::path::PathBuf> {
    use std::time::{SystemTime, UNIX_EPOCH};

    let base = alert_image_directory()?;
    // Best-effort dir create; if it already exists this is a no-op.
    if let Err(e) = std::fs::create_dir_all(&base) {
        warn!(error = %e, dir = %base.display(), "Could not create alert-image dir");
        return None;
    }
    // Restrict the directory itself to user-only access. AlreadyExists
    // is fine — we only own the dir we created, but tightening the
    // mode on a returned existing dir is cheap and idempotent.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = std::fs::metadata(&base) {
            let mut perms = meta.permissions();
            if perms.mode() & 0o077 != 0 {
                perms.set_mode(0o700);
                let _ = std::fs::set_permissions(&base, perms);
            }
        }
    }
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    Some(base.join(format!("alert-{pid}-{nanos}.png")))
}

/// Return the directory alert images should live in. Honors
/// `XDG_RUNTIME_DIR` when set (mode-0700 per spec), otherwise falls
/// back to a `nailbite` subdir of the system temp dir.
fn alert_image_directory() -> Option<std::path::PathBuf> {
    if let Ok(runtime_dir) = std::env::var("XDG_RUNTIME_DIR") {
        if !runtime_dir.is_empty() {
            return Some(std::path::PathBuf::from(runtime_dir).join("nailbite"));
        }
    }
    Some(std::env::temp_dir().join("nailbite"))
}

/// Write `img` as a PNG with user-only file permissions.
///
/// Uses `create_new` so we never silently overwrite a file pre-staged
/// by another process (the predictable-PID symlink-attack vector from
/// the old `/tmp/nailbite-alert-{pid}.png` layout). The PNG encoder
/// streams into the open `File` handle.
fn write_png_user_only(
    img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    path: &std::path::Path,
) -> std::io::Result<()> {
    use std::fs::OpenOptions;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options.open(path)?;
    let buffered = std::io::BufWriter::new(file);
    let encoder = image::codecs::png::PngEncoder::new(buffered);
    img.write_with_encoder(encoder)
        .map_err(|e| std::io::Error::other(e.to_string()))
}

/// Delete alert PNGs in the same directory that are older than the
/// longest plausible notification lifetime. Keeps the dir bounded
/// without breaking notification daemons that may still hold the
/// just-written file open.
fn prune_stale_alert_images(just_written: &std::path::Path) {
    use std::time::{Duration, SystemTime};

    let Some(dir) = just_written.parent() else { return };
    let max_age = Duration::from_secs(120);
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        if entry.path() == just_written {
            continue;
        }
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else { continue };
        if !name_str.starts_with("alert-") || !name_str.ends_with(".png") {
            continue;
        }
        let Ok(metadata) = entry.metadata() else { continue };
        let Ok(modified) = metadata.modified() else { continue };
        if SystemTime::now()
            .duration_since(modified)
            .map(|age| age > max_age)
            .unwrap_or(false)
        {
            let _ = std::fs::remove_file(entry.path());
        }
    }
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
            // Release the claim taken by the detection loop so the
            // next event can try again. Without this the sentinel
            // would stick and block every future notification.
            *state.active_notification_id.lock() = None;
            return;
        }
    };

    // Publish the real XDG id, overwriting the claim sentinel. The
    // in-app AlertModal's verdict / dismiss buttons read this slot
    // to close the notification too. We still let `wait_for_action`
    // consume the handle (it has to — that's the notify-rust API
    // for blocking on action signals).
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
    // The sentinel marks a claim made before notify-rust has returned
    // a real handle id; "closing" id 0 would spawn a stray empty
    // toast on some daemons, so just drop the claim and return.
    if id == NOTIFICATION_CLAIM_SENTINEL {
        return;
    }
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

#[cfg(test)]
#[allow(clippy::indexing_slicing, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
mod tests {
    //! Tests for the presence-detection helpers and `PresenceTracker`
    //! hysteresis. The detection loop itself isn't exercised here — it
    //! needs a live camera + ONNX sessions — but the gating logic that
    //! decides whether the user is in frame is pure and worth covering.

    use super::{
        face_votes_present, pose_votes_present, PresenceTracker, POSE_LEFT_SHOULDER, POSE_NOSE,
        POSE_RIGHT_SHOULDER, PRESENCE_FACE_CONFIDENCE, PRESENCE_POSE_VISIBILITY,
    };
    use crate::detection::types::{FaceDetection, Landmark, PoseDetection, PoseLandmark};

    fn dummy_landmark() -> Landmark {
        Landmark { x: 0.5, y: 0.5, z: 0.0 }
    }

    fn face_with_confidence(conf: f32) -> FaceDetection {
        FaceDetection {
            landmarks: Vec::new(),
            confidence: conf,
        }
    }

    /// Build a 33-landmark pose with given visibilities at specific
    /// indices; everything else is invisible. Anything not in `set`
    /// gets `visibility = 0.0`.
    fn pose_with(set: &[(usize, f32)]) -> PoseDetection {
        let mut lm = vec![
            PoseLandmark {
                landmark: dummy_landmark(),
                visibility: 0.0,
                presence: 1.0,
            };
            33
        ];
        for &(i, v) in set {
            lm[i].visibility = v;
        }
        PoseDetection {
            landmarks: lm,
            confidence: 0.9,
        }
    }

    // --- face_votes_present ---

    #[test]
    fn face_vote_rejects_none() {
        assert!(!face_votes_present(&None));
    }

    #[test]
    fn face_vote_rejects_below_threshold() {
        let f = Some(face_with_confidence(PRESENCE_FACE_CONFIDENCE - 0.01));
        assert!(!face_votes_present(&f));
    }

    #[test]
    fn face_vote_accepts_at_threshold() {
        let f = Some(face_with_confidence(PRESENCE_FACE_CONFIDENCE));
        assert!(face_votes_present(&f));
    }

    #[test]
    fn face_vote_accepts_high_confidence() {
        let f = Some(face_with_confidence(0.99));
        assert!(face_votes_present(&f));
    }

    // --- pose_votes_present ---

    #[test]
    fn pose_vote_rejects_none() {
        assert!(!pose_votes_present(&None));
    }

    #[test]
    fn pose_vote_rejects_partial_visibility() {
        // Only one of three torso landmarks visible; we require at least
        // two — a single floating nose isn't enough.
        let p = Some(pose_with(&[(POSE_NOSE, PRESENCE_POSE_VISIBILITY)]));
        assert!(!pose_votes_present(&p));
    }

    #[test]
    fn pose_vote_accepts_two_of_three_landmarks() {
        // Two of three visible is enough — a real user leaning back or
        // with one shoulder cropped should still pass.
        let p = Some(pose_with(&[
            (POSE_NOSE, 0.9),
            (POSE_LEFT_SHOULDER, 0.9),
        ]));
        assert!(pose_votes_present(&p));
    }

    #[test]
    fn pose_vote_accepts_all_three_above_threshold() {
        let p = Some(pose_with(&[
            (POSE_NOSE, 0.9),
            (POSE_LEFT_SHOULDER, 0.9),
            (POSE_RIGHT_SHOULDER, 0.9),
        ]));
        assert!(pose_votes_present(&p));
    }

    #[test]
    fn pose_vote_rejects_low_visibility_landmarks() {
        // All three "present" but below the visibility floor — the
        // landmark is there but the model thinks it's occluded or
        // hallucinated.
        let below = PRESENCE_POSE_VISIBILITY - 0.05;
        let p = Some(pose_with(&[
            (POSE_NOSE, below),
            (POSE_LEFT_SHOULDER, below),
            (POSE_RIGHT_SHOULDER, below),
        ]));
        assert!(!pose_votes_present(&p));
    }

    // --- PresenceTracker hysteresis ---

    #[test]
    fn tracker_starts_absent() {
        let mut t = PresenceTracker::new();
        // First true observation doesn't immediately flip — must clear
        // the present-debounce.
        assert!(t.update(true).is_none());
    }

    #[test]
    fn tracker_promotes_after_present_debounce() {
        let mut t = PresenceTracker::new();
        // PRESENT_FRAMES = 8 in source. Feed 7 true → still None, then
        // 8th true → Some(true).
        for _ in 0..7 {
            assert!(t.update(true).is_none());
        }
        assert_eq!(t.update(true), Some(true));
    }

    #[test]
    fn tracker_demotes_after_absent_debounce() {
        let mut t = PresenceTracker::new();
        // Get into the present state first.
        for _ in 0..8 {
            t.update(true);
        }
        // ABSENT_FRAMES = 48. Feed 47 false → still None, 48th → Some(false).
        for _ in 0..47 {
            assert!(t.update(false).is_none());
        }
        assert_eq!(t.update(false), Some(false));
    }

    #[test]
    fn tracker_ignores_brief_flicker_to_present() {
        // A single hallucinated face frame inside an absent run shouldn't
        // start a session — that's the whole point of the present-debounce.
        let mut t = PresenceTracker::new();
        for _ in 0..3 {
            assert!(t.update(true).is_none());
        }
        // Flicker back to absent; counter resets.
        assert!(t.update(false).is_none());
        // Now ramp up properly: takes another full PRESENT_FRAMES.
        for _ in 0..7 {
            assert!(t.update(true).is_none());
        }
        assert_eq!(t.update(true), Some(true));
    }

    #[test]
    fn tracker_ignores_brief_glance_away() {
        // User looks sideways for a few frames; the session shouldn't end.
        let mut t = PresenceTracker::new();
        for _ in 0..8 {
            t.update(true);
        }
        // 5 false frames — fewer than ABSENT_FRAMES (48). Stays present.
        for _ in 0..5 {
            assert!(t.update(false).is_none());
        }
        // One true frame resets the counter; remains present.
        assert!(t.update(true).is_none());
    }
}
