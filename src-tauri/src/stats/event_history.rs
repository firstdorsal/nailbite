//! Event history recording for detection events.
//!
//! Captures frame sequences around detection events for debugging and review.
//! Each event is saved as a directory with JPEG frames (raw + annotated) and
//! a JSON metadata file.

use std::collections::VecDeque;
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use image::{ImageBuffer, Rgb};
use serde::Serialize;
use tracing::{debug, info, warn};

use crate::commands::detection::{FaceResult, HandResult, PoseResult};
use crate::config::HistoryConfig;
use crate::detection::types::{
    BfrbType, DetectionExplanation, FaceDetection, HandDetection, PoseDetection,
};
use crate::frame::Frame;
use crate::paths::expand_tilde;
use crate::stats::annotate::annotate_frame;

/// A single buffered frame with associated detection state.
#[derive(Clone)]
pub struct BufferedFrame {
    /// The raw camera frame.
    pub frame: Frame,
    /// Hand detections for this frame.
    pub hands: Vec<HandDetection>,
    /// Face detection for this frame.
    pub face: Option<FaceDetection>,
    /// Pose detection for this frame.
    pub pose: Option<PoseDetection>,
    /// Per-detector explanation produced for this frame. May be empty when
    /// the frame predates detection (e.g. paused) or when detectors were
    /// not run (auxiliary cameras).
    pub explanations: Vec<DetectionExplanation>,
    /// Frame timestamp (ms since detection start).
    pub timestamp_ms: u64,
}

/// Trigger type for event recording.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EventTrigger {
    /// BFRB behavior detected (confirmed by temporal tracker).
    Detection,
    /// User reported a missed event.
    MissedEvent,
    /// User dismissed a false positive.
    FalsePositive,
}

/// Vector overlay data for a single frame — landmarks stored as JSON so
/// they can be re-rendered on a canvas at display time. Saving them this
/// way (instead of baking pixels into a JPEG) keeps the recorded frame
/// pristine for re-analysis at any zoom and any future visualization.
#[derive(Debug, Serialize)]
struct FrameOverlay {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    hands: Vec<HandResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    face: Option<FaceResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pose: Option<PoseResult>,
}

/// Per-frame metadata in the event JSON.
#[derive(Debug, Serialize)]
struct FrameMetadata {
    /// Offset from trigger frame (negative = before, positive = after).
    offset: i32,
    /// Frame timestamp (ms since detection start).
    timestamp_ms: u64,
    /// Number of tracked hands.
    hand_count: usize,
    /// Hand sides detected.
    hand_sides: Vec<String>,
    /// Hand confidences.
    hand_confidences: Vec<f32>,
    /// Whether face was detected.
    face_detected: bool,
    /// Whether pose was detected.
    pose_detected: bool,
    /// Vector overlay (hand/face/pose landmarks). Drawn on top of the raw
    /// frame at display time so the stored pixels stay untouched.
    #[serde(skip_serializing_if = "Option::is_none")]
    overlay: Option<FrameOverlay>,
    /// Per-detector explanation for this frame (one entry per active detector).
    /// Empty when this frame had no detection run against it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    explanations: Vec<DetectionExplanation>,
}

/// Event metadata written to event.json.
#[derive(Debug, Serialize)]
struct EventMetadata {
    /// When the event was recorded.
    timestamp: String,
    /// What triggered the recording.
    trigger: EventTrigger,
    /// BFRB type (if detection trigger).
    bfrb_type: Option<String>,
    /// Detection confidence (if detection trigger).
    confidence: Option<f32>,
    /// Trigger-frame explanation: contributing signals at the moment the
    /// alert fired. Only present for `Detection` triggers.
    #[serde(skip_serializing_if = "Option::is_none")]
    explanation: Option<DetectionExplanation>,
    /// Per-frame detection state.
    frames: Vec<FrameMetadata>,
}

/// State for ongoing post-trigger frame capture.
struct PostCapture {
    /// Pre-trigger frames (already captured).
    pre_frames: Vec<BufferedFrame>,
    /// Post-trigger frames being collected.
    post_frames: Vec<BufferedFrame>,
    /// How many post-trigger frames to collect.
    remaining: usize,
    /// Event trigger type.
    trigger: EventTrigger,
    /// BFRB type.
    bfrb_type: Option<BfrbType>,
    /// Detection confidence.
    confidence: Option<f32>,
    /// Trigger-frame explanation for `Detection` triggers.
    explanation: Option<DetectionExplanation>,
}

/// Records event history with frame ring buffer.
pub struct EventHistoryRecorder {
    /// Configuration.
    config: HistoryConfig,
    /// Resolved history directory path (tilde expanded).
    history_dir: PathBuf,
    /// Ring buffer of recent frames.
    ring_buffer: VecDeque<BufferedFrame>,
    /// Whether annotation is enabled.
    annotate: bool,
    /// Active post-capture state (if collecting post-trigger frames).
    post_capture: Option<PostCapture>,
}

impl EventHistoryRecorder {
    /// Create a new event history recorder.
    pub fn new(config: &HistoryConfig) -> Self {
        let history_dir = expand_tilde(&config.dir);
        let buffer_size = config.frames_before + 1; // +1 for trigger frame
        Self {
            config: config.clone(),
            history_dir,
            ring_buffer: VecDeque::with_capacity(buffer_size),
            annotate: config.annotate_landmarks,
            post_capture: None,
        }
    }

    /// Check if recording is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Drop the pre-trigger ring buffer. Called when presence is lost so
    /// a later event doesn't include empty-chair frames in its
    /// `frames_before`. No effect on an in-progress post-capture (we
    /// still want the post frames from the event that already fired).
    pub fn clear_ring_buffer(&mut self) {
        self.ring_buffer.clear();
    }

    /// Push a new frame into the ring buffer.
    /// If a post-capture is in progress, also collects post-trigger frames.
    pub fn push_frame(&mut self, frame: BufferedFrame) {
        // Check if we're in post-capture mode
        if let Some(ref mut pc) = self.post_capture {
            pc.post_frames.push(frame.clone());
            pc.remaining = pc.remaining.saturating_sub(1);
            if pc.remaining == 0 {
                // Post-capture complete — save the event
                let pc = self.post_capture.take().unwrap();
                self.save_event(pc);
            }
        }

        // Maintain ring buffer at configured size
        let max_size = self.config.frames_before + 1;
        while self.ring_buffer.len() >= max_size {
            self.ring_buffer.pop_front();
        }
        self.ring_buffer.push_back(frame);
    }

    /// Trigger event recording. Captures current ring buffer as pre-frames
    /// and starts collecting post-trigger frames. Returns `true` if a new
    /// capture was started (or saved immediately), `false` if it was
    /// dropped because the recorder was busy or disabled.
    pub fn trigger(
        &mut self,
        trigger: EventTrigger,
        bfrb_type: Option<BfrbType>,
        confidence: Option<f32>,
    ) -> bool {
        self.trigger_with_explanation(trigger, bfrb_type, confidence, None)
    }

    /// Like `trigger`, but additionally records the trigger-frame explanation
    /// so the saved event can be rendered with full signal context. Returns
    /// `true` when the call actually started (or completed) a capture —
    /// callers use this to keep the today-counter in sync with the events
    /// that will actually land on disk.
    pub fn trigger_with_explanation(
        &mut self,
        trigger: EventTrigger,
        bfrb_type: Option<BfrbType>,
        confidence: Option<f32>,
        explanation: Option<DetectionExplanation>,
    ) -> bool {
        if !self.config.enabled {
            return false;
        }

        // Don't start a new capture if one is already in progress
        if self.post_capture.is_some() {
            debug!("Event recording already in progress, skipping trigger");
            return false;
        }

        let pre_frames: Vec<BufferedFrame> = self.ring_buffer.iter().cloned().collect();

        if self.config.frames_after == 0 {
            // No post-capture needed, save immediately
            let pc = PostCapture {
                pre_frames,
                post_frames: Vec::new(),
                remaining: 0,
                trigger,
                bfrb_type,
                confidence,
                explanation,
            };
            self.save_event(pc);
        } else {
            self.post_capture = Some(PostCapture {
                pre_frames,
                post_frames: Vec::new(),
                remaining: self.config.frames_after,
                trigger,
                bfrb_type,
                confidence,
                explanation,
            });
            debug!(
                frames_after = self.config.frames_after,
                "Started post-capture for event history"
            );
        }
        true
    }

    /// Save an event to disk.
    fn save_event(&self, capture: PostCapture) {
        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
        let bfrb_str = capture
            .bfrb_type
            .map(|b| b.as_str().to_string())
            .unwrap_or_else(|| format!("{:?}", capture.trigger).to_lowercase());
        let event_dir = self.history_dir.join(format!("{timestamp}_{bfrb_str}"));

        // Create directories
        let frames_dir = event_dir.join("frames");
        if let Err(e) = fs::create_dir_all(&frames_dir) {
            warn!(error = %e, path = %frames_dir.display(), "Failed to create event history directory");
            return;
        }

        let pre_count = capture.pre_frames.len();
        let mut all_frames = capture.pre_frames;
        all_frames.extend(capture.post_frames);

        let mut frame_metadata = Vec::new();

        for (i, buffered) in all_frames.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let offset = i as i32 - pre_count as i32;
            let filename = format!("frame_{offset:+03}.jpg");

            // Save raw frame
            if let Err(e) = save_frame_jpeg(&buffered.frame, &frames_dir.join(&filename)) {
                warn!(error = %e, filename = %filename, "Failed to save frame");
                continue;
            }

            // Save annotated frame
            if self.annotate {
                let annotated = annotate_frame(
                    &buffered.frame,
                    &buffered.hands,
                    buffered.face.as_ref(),
                    buffered.pose.as_ref(),
                );
                let ann_filename = filename.replace(".jpg", "_annotated.jpg");
                if let Err(e) = save_frame_jpeg(&annotated, &frames_dir.join(&ann_filename)) {
                    warn!(error = %e, filename = %ann_filename, "Failed to save annotated frame");
                }
            }

            // Build the vector overlay so the UI can re-render landmarks
            // on top of the raw frame without baking them into pixels.
            let overlay = build_overlay(buffered);

            frame_metadata.push(FrameMetadata {
                offset,
                timestamp_ms: buffered.timestamp_ms,
                hand_count: buffered.hands.len(),
                hand_sides: buffered
                    .hands
                    .iter()
                    .map(|h| {
                        h.side
                            .map(|s| format!("{s:?}").to_lowercase())
                            .unwrap_or_else(|| "unknown".to_string())
                    })
                    .collect(),
                hand_confidences: buffered.hands.iter().map(|h| h.confidence).collect(),
                face_detected: buffered.face.is_some(),
                pose_detected: buffered.pose.is_some(),
                overlay,
                explanations: buffered.explanations.clone(),
            });
        }

        // Write event metadata
        let metadata = EventMetadata {
            timestamp: chrono::Utc::now().to_rfc3339(),
            trigger: capture.trigger,
            bfrb_type: capture.bfrb_type.map(|b| b.as_str().to_string()),
            confidence: capture.confidence,
            explanation: capture.explanation,
            frames: frame_metadata,
        };

        let event_json = event_dir.join("event.json");
        match serde_json::to_string_pretty(&metadata) {
            Ok(json) => {
                if let Err(e) = fs::write(&event_json, json) {
                    warn!(error = %e, "Failed to write event.json");
                }
            }
            Err(e) => {
                warn!(error = %e, "Failed to serialize event metadata");
            }
        }

        info!(
            path = %event_dir.display(),
            trigger = ?capture.trigger,
            frame_count = all_frames.len(),
            "Saved event history"
        );

        // Auto-prune
        self.prune_old_events();
    }

    /// Remove oldest event directories when max_events is exceeded.
    fn prune_old_events(&self) {
        let Ok(entries) = fs::read_dir(&self.history_dir) else {
            return;
        };

        let mut dirs: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .map(|e| e.path())
            .collect();

        if dirs.len() <= self.config.max_events {
            return;
        }

        // Sort by name (which starts with timestamp, so oldest first)
        dirs.sort();

        let to_remove = dirs.len() - self.config.max_events;
        for dir in dirs.iter().take(to_remove) {
            info!(path = %dir.display(), "Pruning old event history");
            if let Err(e) = fs::remove_dir_all(dir) {
                warn!(error = %e, path = %dir.display(), "Failed to prune event directory");
            }
        }
    }

    /// Count detection events that landed on disk for `today` (local time).
    /// Directory names are `YYYYMMDD_HHMMSS_<bfrb>` so we just match the
    /// date prefix; the badge must mirror what the user sees in the event
    /// history list, not the raw trigger count from the session log.
    pub fn count_today_events(&self) -> u32 {
        let today = chrono::Local::now().date_naive();
        let prefix = today.format("%Y%m%d_").to_string();
        let Ok(entries) = fs::read_dir(&self.history_dir) else {
            return 0;
        };
        u32::try_from(
            entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .filter_map(|e| e.file_name().to_str().map(String::from))
                .filter(|name| name.starts_with(&prefix))
                .count(),
        )
        .unwrap_or(u32::MAX)
    }

    /// Prune on startup (in case max_events was reduced).
    pub fn prune_on_startup(&self) {
        if !self.config.enabled {
            return;
        }
        if let Err(e) = fs::create_dir_all(&self.history_dir) {
            warn!(error = %e, "Failed to create history directory");
            return;
        }
        self.prune_old_events();
    }
}

/// Build the vector overlay payload from a buffered frame so it can be
/// serialized into event.json. Returns `None` when nothing was detected.
fn build_overlay(buffered: &BufferedFrame) -> Option<FrameOverlay> {
    if buffered.hands.is_empty() && buffered.face.is_none() && buffered.pose.is_none() {
        return None;
    }
    Some(FrameOverlay {
        hands: buffered.hands.iter().map(HandResult::from).collect(),
        face: buffered.face.as_ref().map(FaceResult::from),
        pose: buffered.pose.as_ref().map(PoseResult::from),
    })
}

/// Save a frame as JPEG to a file path.
fn save_frame_jpeg(frame: &Frame, path: &Path) -> Result<(), image::ImageError> {
    let img: ImageBuffer<Rgb<u8>, _> =
        ImageBuffer::from_raw(frame.width, frame.height, frame.data.clone()).ok_or_else(|| {
            image::ImageError::Parameter(image::error::ParameterError::from_kind(
                image::error::ParameterErrorKind::DimensionMismatch,
            ))
        })?;

    let mut buffer = Cursor::new(Vec::new());
    img.write_to(&mut buffer, image::ImageFormat::Jpeg)?;
    fs::write(path, buffer.into_inner())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn test_config(dir: &Path) -> HistoryConfig {
        HistoryConfig {
            enabled: true,
            dir: dir.to_path_buf(),
            frames_before: 2,
            frames_after: 2,
            annotate_landmarks: false,
            max_events: 3,
        }
    }

    fn make_buffered_frame(ts: u64) -> BufferedFrame {
        BufferedFrame {
            frame: Frame::new(vec![100u8; 10 * 10 * 3], 10, 10),
            hands: vec![],
            face: None,
            pose: None,
            explanations: vec![],
            timestamp_ms: ts,
        }
    }

    #[test]
    fn ring_buffer_maintains_size() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());
        let mut recorder = EventHistoryRecorder::new(&config);

        // Push more frames than buffer size (frames_before + 1 = 3)
        for i in 0..10 {
            recorder.push_frame(make_buffered_frame(i * 100));
        }

        assert_eq!(recorder.ring_buffer.len(), 3);
    }

    #[test]
    fn trigger_saves_event_immediately_when_no_post_frames() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = test_config(dir.path());
        config.frames_after = 0;
        let mut recorder = EventHistoryRecorder::new(&config);

        // Push some frames
        for i in 0..5 {
            recorder.push_frame(make_buffered_frame(i * 100));
        }

        // Trigger
        recorder.trigger(
            EventTrigger::Detection,
            Some(BfrbType::NailBiting),
            Some(0.85),
        );

        // Event directory should exist
        let entries: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(entries.len(), 1);

        // Check event.json exists
        let event_dir = &entries[0].path();
        assert!(event_dir.join("event.json").exists());
        assert!(event_dir.join("frames").exists());
    }

    #[test]
    fn trigger_with_post_capture_collects_frames() {
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path()); // frames_after = 2
        let mut recorder = EventHistoryRecorder::new(&config);

        // Push pre-frames
        for i in 0..5 {
            recorder.push_frame(make_buffered_frame(i * 100));
        }

        // Trigger
        recorder.trigger(
            EventTrigger::Detection,
            Some(BfrbType::NailBiting),
            Some(0.85),
        );

        // Should be in post-capture mode
        assert!(recorder.post_capture.is_some());

        // Push post-frames
        recorder.push_frame(make_buffered_frame(600));
        assert!(recorder.post_capture.is_some()); // still collecting

        recorder.push_frame(make_buffered_frame(700));
        assert!(recorder.post_capture.is_none()); // done

        // Event should be saved
        let entries: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn prune_removes_oldest_events() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = test_config(dir.path());
        config.max_events = 2;
        config.frames_after = 0;
        let mut recorder = EventHistoryRecorder::new(&config);

        // Create 3 events (max_events = 2)
        for i in 0..3 {
            for j in 0..3 {
                recorder.push_frame(make_buffered_frame(i * 1000 + j * 100));
            }
            recorder.trigger(
                EventTrigger::Detection,
                Some(BfrbType::NailBiting),
                Some(0.8),
            );
            // Small delay to ensure unique timestamps
            std::thread::sleep(std::time::Duration::from_millis(1100));
        }

        // Should have pruned to max_events
        let entries: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .collect();
        assert!(
            entries.len() <= 2,
            "Expected <= 2 events after pruning, got {}",
            entries.len()
        );
    }

    #[test]
    fn event_json_round_trips_explanation() {
        use crate::detection::types::{HandSignal, SuppressionReason};

        let dir = tempfile::tempdir().unwrap();
        let mut config = test_config(dir.path());
        config.frames_after = 0;
        let mut recorder = EventHistoryRecorder::new(&config);

        let exp = DetectionExplanation {
            bfrb_type: BfrbType::NailBiting,
            hands: vec![HandSignal {
                hand_index: 0,
                side: None,
                normalized_distance: 0.12,
                distance_threshold: 0.35,
                contributing_fingertip: Some(8),
                partner_fingertip: None,
                curl: Some(0.6),
                bonus: 0.0,
                confidence: 0.9,
            }],
            suppressions: vec![SuppressionReason::ChinRest],
            frame_confidence: 0.9,
        };

        // Fill the ring buffer (size = frames_before + 1) so the last pushed
        // frame becomes offset 0 (the trigger frame). The latest frame is the
        // one we care about — give it the explanation.
        for i in 0..config.frames_before {
            let mut earlier = make_buffered_frame((i as u64) * 100);
            earlier.explanations = vec![]; // pre-trigger has no explanation
            recorder.push_frame(earlier);
        }
        let mut trigger_buf = make_buffered_frame((config.frames_before as u64) * 100);
        trigger_buf.explanations = vec![exp.clone()];
        recorder.push_frame(trigger_buf);

        recorder.trigger_with_explanation(
            EventTrigger::Detection,
            Some(BfrbType::NailBiting),
            Some(0.9),
            Some(exp.clone()),
        );

        // Find the saved event.json.
        let entry = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .next()
            .unwrap();
        let json_path = entry.path().join("event.json");
        let body = fs::read_to_string(&json_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();

        // Top-level explanation present.
        let top_exp = parsed
            .get("explanation")
            .and_then(|v| serde_json::from_value::<DetectionExplanation>(v.clone()).ok())
            .expect("trigger explanation persisted");
        assert!((top_exp.frame_confidence - 0.9).abs() < 1e-6);
        assert_eq!(top_exp.hands[0].contributing_fingertip, Some(8));
        assert_eq!(
            top_exp.suppressions,
            vec![SuppressionReason::ChinRest]
        );

        // Per-frame explanations array persisted on the latest pre-trigger
        // frame (offset = -1 when frames_after == 0).
        let frames = parsed.get("frames").and_then(|f| f.as_array()).unwrap();
        let last_pre = frames
            .iter()
            .max_by_key(|f| f.get("offset").and_then(|o| o.as_i64()).unwrap_or(i64::MIN))
            .expect("at least one frame saved");
        let frame_exps = last_pre
            .get("explanations")
            .and_then(|v| serde_json::from_value::<Vec<DetectionExplanation>>(v.clone()).ok())
            .expect("per-frame explanations persisted");
        assert_eq!(frame_exps.len(), 1);
        assert_eq!(frame_exps[0].bfrb_type, BfrbType::NailBiting);
    }

    #[test]
    fn disabled_recorder_does_not_save() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = test_config(dir.path());
        config.enabled = false;
        let mut recorder = EventHistoryRecorder::new(&config);

        recorder.push_frame(make_buffered_frame(100));
        recorder.trigger(EventTrigger::Detection, Some(BfrbType::NailBiting), Some(0.9));

        let entries: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(entries.len(), 0);
    }
}
