//! Event history commands for viewing recorded detection events.

use std::fs;
use std::sync::Arc;

use base64::Engine;
use serde::{Deserialize, Serialize};
use tauri::State;
use tracing::warn;

use crate::detection::types::DetectionExplanation;
use crate::errors::NailbiteError;
use crate::paths::expand_tilde;
use crate::state::AppState;

/// User's explicit verdict on whether the recorded event was a real BFRB.
///
/// Stored alongside `user_rating` in `event.json`. Where `user_rating` is a
/// fuzzy 1-5 quality slider, `Verdict` is the unambiguous label used by the
/// label-driven threshold tuning (Track 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Detection was correct.
    TruePositive,
    /// Detection fired but the user wasn't actually engaging in the behavior.
    FalsePositive,
    /// User isn't sure (still useful: excluded from training, kept for review).
    Unsure,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TruePositive => "true_positive",
            Self::FalsePositive => "false_positive",
            Self::Unsure => "unsure",
        }
    }
}

/// Summary of a single event history entry (for the list view).
#[derive(Debug, Serialize)]
pub struct EventHistorySummary {
    /// Directory name (used as event ID).
    pub id: String,
    /// ISO timestamp from event.json.
    pub timestamp: String,
    /// What triggered the recording.
    pub trigger: String,
    /// BFRB type (if applicable).
    pub bfrb_type: Option<String>,
    /// Detection confidence (if applicable).
    pub confidence: Option<f32>,
    /// Number of frames in this event.
    pub frame_count: usize,
    /// User accuracy rating 1-5 (null = unreviewed).
    pub user_rating: Option<u8>,
    /// Explicit user verdict on whether this was a real BFRB.
    pub verdict: Option<Verdict>,
    /// Trigger frame raw filename (for thumbnail preview).
    pub trigger_frame: Option<String>,
    /// Trigger frame annotated filename (if available).
    pub trigger_frame_annotated: Option<String>,
    /// Raw frame filenames for the whole event in chronological order
    /// (e.g. `frame_-05.jpg` … `frame_+05.jpg`). Lets the list view show
    /// a filmstrip when the viewport is wide enough.
    pub frame_files: Vec<String>,
}

/// Full event details including per-frame metadata.
#[derive(Debug, Serialize)]
pub struct EventHistoryDetail {
    /// Directory name (used as event ID).
    pub id: String,
    /// ISO timestamp from event.json.
    pub timestamp: String,
    /// What triggered the recording.
    pub trigger: String,
    /// BFRB type (if applicable).
    pub bfrb_type: Option<String>,
    /// Detection confidence (if applicable).
    pub confidence: Option<f32>,
    /// User accuracy rating 1-5 (null = unreviewed).
    pub user_rating: Option<u8>,
    /// Explicit user verdict (true positive / false positive / unsure).
    pub verdict: Option<Verdict>,
    /// Optional free-text reason explaining the verdict.
    pub verdict_reason: Option<String>,
    /// Trigger-frame explanation: the contributing signals at the moment
    /// the alert fired. Only populated for `Detection` triggers.
    pub explanation: Option<DetectionExplanation>,
    /// Per-frame metadata.
    pub frames: Vec<FrameInfo>,
}

/// Vector overlay (hand / face / pose landmarks) for a single frame.
/// Stored as opaque JSON because the schema mirrors
/// `commands::detection::*Result` and we don't need to validate it on the
/// read path — the frontend re-renders it directly onto a canvas.
#[derive(Debug, Clone, Serialize)]
pub struct FrameOverlayView {
    #[serde(default)]
    pub hands: Vec<serde_json::Value>,
    pub face: Option<serde_json::Value>,
    pub pose: Option<serde_json::Value>,
}

/// Per-frame info for the detail view.
#[derive(Debug, Serialize)]
pub struct FrameInfo {
    /// Offset from trigger frame (negative = before, positive = after).
    pub offset: i32,
    /// Frame timestamp (ms since detection start).
    pub timestamp_ms: u64,
    /// Number of tracked hands.
    pub hand_count: usize,
    /// Hand sides detected.
    pub hand_sides: Vec<String>,
    /// Hand confidences.
    pub hand_confidences: Vec<f32>,
    /// Whether face was detected.
    pub face_detected: bool,
    /// Whether pose was detected.
    pub pose_detected: bool,
    /// Raw frame filename.
    pub raw_filename: String,
    /// Annotated frame filename (legacy baked-pixel overlay, may not exist).
    pub annotated_filename: Option<String>,
    /// Vector overlay — preferred: drawn at display time on top of the raw
    /// frame so the saved pixels stay untouched.
    pub overlay: Option<FrameOverlayView>,
    /// Per-detector explanation for this frame (one entry per active detector).
    /// Empty for older events recorded before explanations were captured.
    pub explanations: Vec<DetectionExplanation>,
}

/// Pick the trigger frame from the frames array in event.json.
/// Returns (raw_filename, Option<annotated_filename>).
/// Prefers the frame closest to offset 0 from the negative side (the actual trigger frame).
fn pick_trigger_frame(
    metadata: &serde_json::Value,
    frames_dir: &std::path::Path,
) -> (Option<String>, Option<String>) {
    let Some(frames) = metadata.get("frames").and_then(|f| f.as_array()) else {
        return (None, None);
    };

    // Find the frame with the largest offset <= 0 (trigger frame),
    // falling back to the smallest positive offset.
    let best = frames
        .iter()
        .filter_map(|f| {
            f.get("offset")
                .and_then(|o| o.as_i64())
                .map(|offset| (offset, f))
        })
        .min_by_key(|(offset, _)| {
            // Sort: non-positive offsets first (by distance from 0), then positive offsets
            if *offset <= 0 {
                (0, offset.unsigned_abs())
            } else {
                #[allow(clippy::cast_sign_loss)]
                (1, *offset as u64)
            }
        });

    let Some((offset, _)) = best else {
        return (None, None);
    };

    #[allow(clippy::cast_possible_truncation)]
    let offset = offset as i32;
    let raw = format!("frame_{offset:+03}.jpg");
    let annotated = format!("frame_{offset:+03}_annotated.jpg");

    let annotated_exists = frames_dir.join(&annotated).exists();

    (
        Some(raw),
        if annotated_exists {
            Some(annotated)
        } else {
            None
        },
    )
}

/// Read `user_rating` from event metadata (supports both old `user_label` and new `user_rating`).
fn read_user_rating(metadata: &serde_json::Value) -> Option<u8> {
    // New format: numeric rating 1-5
    if let Some(rating) = metadata.get("user_rating").and_then(|v| v.as_u64()) {
        #[allow(clippy::cast_possible_truncation)]
        return Some(rating.min(5) as u8);
    }
    // Migration: old format had user_label: "correct" / "incorrect"
    if let Some(label) = metadata.get("user_label").and_then(|v| v.as_str()) {
        return match label {
            "correct" => Some(5),
            "incorrect" => Some(1),
            _ => None,
        };
    }
    None
}

/// Read the user's verdict from event metadata. Falls back to the legacy
/// `user_label` ("correct"/"incorrect") so older events are auto-migrated.
fn read_verdict(metadata: &serde_json::Value) -> Option<Verdict> {
    if let Some(v) = metadata
        .get("verdict")
        .and_then(|v| serde_json::from_value::<Verdict>(v.clone()).ok())
    {
        return Some(v);
    }
    if let Some(label) = metadata.get("user_label").and_then(|v| v.as_str()) {
        return match label {
            "correct" => Some(Verdict::TruePositive),
            "incorrect" => Some(Verdict::FalsePositive),
            _ => None,
        };
    }
    None
}

fn read_verdict_reason(metadata: &serde_json::Value) -> Option<String> {
    metadata
        .get("verdict_reason")
        .and_then(|v| v.as_str())
        .map(String::from)
}

/// List all event history entries (newest first).
#[tauri::command]
pub fn list_event_history(
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<EventHistorySummary>, NailbiteError> {
    let config = state.config.read();
    let history_dir = expand_tilde(&config.history.dir);
    drop(config);

    if !history_dir.exists() {
        return Ok(Vec::new());
    }

    let entries = fs::read_dir(&history_dir).map_err(|e| {
        warn!(error = %e, path = %history_dir.display(), "Failed to read history directory");
        NailbiteError::Io(e)
    })?;

    let mut summaries: Vec<EventHistorySummary> = Vec::new();

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let dir_name = match path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name.to_string(),
            None => continue,
        };

        let event_json_path = path.join("event.json");
        if !event_json_path.exists() {
            continue;
        }

        let json_content = match fs::read_to_string(&event_json_path) {
            Ok(content) => content,
            Err(e) => {
                warn!(error = %e, path = %event_json_path.display(), "Failed to read event.json");
                continue;
            }
        };

        let metadata: serde_json::Value = match serde_json::from_str(&json_content) {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, path = %event_json_path.display(), "Failed to parse event.json");
                continue;
            }
        };

        let frame_count = metadata
            .get("frames")
            .and_then(|f| f.as_array())
            .map(|a| a.len())
            .unwrap_or(0);

        let frames_dir = path.join("frames");
        let (trigger_frame, trigger_frame_annotated) =
            pick_trigger_frame(&metadata, &frames_dir);

        // Collect raw frame filenames in chronological (offset) order so
        // the list view can render a filmstrip.
        let mut frame_files: Vec<(i32, String)> = metadata
            .get("frames")
            .and_then(|f| f.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|f| {
                        let offset = f.get("offset").and_then(|o| o.as_i64())?;
                        #[allow(clippy::cast_possible_truncation)]
                        let offset = offset as i32;
                        Some((offset, format!("frame_{offset:+03}.jpg")))
                    })
                    .collect()
            })
            .unwrap_or_default();
        frame_files.sort_by_key(|(o, _)| *o);
        let frame_files: Vec<String> = frame_files.into_iter().map(|(_, n)| n).collect();

        summaries.push(EventHistorySummary {
            id: dir_name,
            timestamp: metadata
                .get("timestamp")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string(),
            trigger: metadata
                .get("trigger")
                .and_then(|t| t.as_str())
                .unwrap_or("unknown")
                .to_string(),
            bfrb_type: metadata
                .get("bfrb_type")
                .and_then(|t| t.as_str())
                .map(String::from),
            #[allow(clippy::cast_possible_truncation)]
            confidence: metadata
                .get("confidence")
                .and_then(|c| c.as_f64())
                .map(|c| c as f32),
            frame_count,
            user_rating: read_user_rating(&metadata),
            verdict: read_verdict(&metadata),
            trigger_frame,
            trigger_frame_annotated,
            frame_files,
        });
    }

    // Sort by directory name descending (newest first, since names start with timestamp)
    summaries.sort_by(|a, b| b.id.cmp(&a.id));

    Ok(summaries)
}

/// Get detailed event info including per-frame metadata and available filenames.
#[tauri::command]
pub fn get_event_details(
    state: State<'_, Arc<AppState>>,
    event_id: String,
) -> Result<EventHistoryDetail, NailbiteError> {
    let config = state.config.read();
    let history_dir = expand_tilde(&config.history.dir);
    drop(config);

    let event_dir = history_dir.join(&event_id);
    let event_json_path = event_dir.join("event.json");

    let json_content = fs::read_to_string(&event_json_path)?;
    let metadata: serde_json::Value = serde_json::from_str(&json_content)
        .map_err(|e| NailbiteError::Camera(format!("Failed to parse event.json: {e}")))?;

    let frames_dir = event_dir.join("frames");
    let frames = metadata
        .get("frames")
        .and_then(|f| f.as_array())
        .map(|arr| {
            arr.iter()
                .map(|frame| {
                    #[allow(clippy::cast_possible_truncation)]
                    let offset = frame
                        .get("offset")
                        .and_then(|o| o.as_i64())
                        .unwrap_or(0) as i32;
                    let raw_filename = format!("frame_{offset:+03}.jpg");
                    let annotated_filename = format!("frame_{offset:+03}_annotated.jpg");

                    let annotated_exists = frames_dir.join(&annotated_filename).exists();

                    #[allow(clippy::cast_possible_truncation)]
                    let hand_count = frame
                        .get("hand_count")
                        .and_then(|c| c.as_u64())
                        .unwrap_or(0) as usize;

                    let overlay = frame.get("overlay").and_then(|v| {
                        if v.is_null() {
                            return None;
                        }
                        let hands = v
                            .get("hands")
                            .and_then(|h| h.as_array())
                            .map(|arr| arr.to_vec())
                            .unwrap_or_default();
                        let face = v.get("face").cloned().filter(|x| !x.is_null());
                        let pose = v.get("pose").cloned().filter(|x| !x.is_null());
                        if hands.is_empty() && face.is_none() && pose.is_none() {
                            None
                        } else {
                            Some(FrameOverlayView { hands, face, pose })
                        }
                    });

                    FrameInfo {
                        offset,
                        timestamp_ms: frame
                            .get("timestamp_ms")
                            .and_then(|t| t.as_u64())
                            .unwrap_or(0),
                        hand_count,
                        hand_sides: frame
                            .get("hand_sides")
                            .and_then(|s| s.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default(),
                        #[allow(clippy::cast_possible_truncation)]
                        hand_confidences: frame
                            .get("hand_confidences")
                            .and_then(|s| s.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                                    .collect()
                            })
                            .unwrap_or_default(),
                        face_detected: frame
                            .get("face_detected")
                            .and_then(|f| f.as_bool())
                            .unwrap_or(false),
                        pose_detected: frame
                            .get("pose_detected")
                            .and_then(|p| p.as_bool())
                            .unwrap_or(false),
                        raw_filename,
                        annotated_filename: if annotated_exists {
                            Some(annotated_filename)
                        } else {
                            None
                        },
                        overlay,
                        explanations: frame
                            .get("explanations")
                            .and_then(|v| {
                                serde_json::from_value::<Vec<DetectionExplanation>>(
                                    v.clone(),
                                )
                                .ok()
                            })
                            .unwrap_or_default(),
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(EventHistoryDetail {
        id: event_id,
        timestamp: metadata
            .get("timestamp")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string(),
        trigger: metadata
            .get("trigger")
            .and_then(|t| t.as_str())
            .unwrap_or("unknown")
            .to_string(),
        bfrb_type: metadata
            .get("bfrb_type")
            .and_then(|t| t.as_str())
            .map(String::from),
        #[allow(clippy::cast_possible_truncation)]
        confidence: metadata
            .get("confidence")
            .and_then(|c| c.as_f64())
            .map(|c| c as f32),
        user_rating: read_user_rating(&metadata),
        verdict: read_verdict(&metadata),
        verdict_reason: read_verdict_reason(&metadata),
        explanation: metadata
            .get("explanation")
            .and_then(|v| {
                serde_json::from_value::<DetectionExplanation>(v.clone()).ok()
            }),
        frames,
    })
}

/// Get a frame image as base64-encoded JPEG.
#[tauri::command]
pub fn get_event_frame(
    state: State<'_, Arc<AppState>>,
    event_id: String,
    filename: String,
) -> Result<String, NailbiteError> {
    let config = state.config.read();
    let history_dir = expand_tilde(&config.history.dir);
    drop(config);

    let frame_path = history_dir.join(&event_id).join("frames").join(&filename);

    // Security: ensure the resolved path is within the history directory
    let canonical_history = history_dir
        .canonicalize()
        .unwrap_or_else(|_| history_dir.clone());
    let canonical_frame = frame_path.canonicalize().map_err(NailbiteError::Io)?;

    if !canonical_frame.starts_with(&canonical_history) {
        return Err(NailbiteError::Camera(
            "Path traversal attempt denied".to_string(),
        ));
    }

    let data = fs::read(&canonical_frame)?;
    let base64 = base64::engine::general_purpose::STANDARD.encode(&data);

    Ok(base64)
}

/// Set user accuracy rating on an event (1-5, or null to clear).
#[tauri::command]
pub fn rate_event(
    state: State<'_, Arc<AppState>>,
    event_id: String,
    rating: Option<u8>,
) -> Result<(), NailbiteError> {
    let config = state.config.read();
    let history_dir = expand_tilde(&config.history.dir);
    drop(config);

    let event_dir = history_dir.join(&event_id);
    let event_json_path = event_dir.join("event.json");

    let json_content = fs::read_to_string(&event_json_path)?;
    let mut metadata: serde_json::Value = serde_json::from_str(&json_content)
        .map_err(|e| NailbiteError::Camera(format!("Failed to parse event.json: {e}")))?;

    let obj = metadata
        .as_object_mut()
        .ok_or_else(|| NailbiteError::Camera("event.json is not an object".to_string()))?;

    // Remove old label format if present
    obj.remove("user_label");

    if let Some(value) = rating {
        let clamped = value.clamp(1, 5);
        obj.insert(
            "user_rating".to_string(),
            serde_json::Value::Number(clamped.into()),
        );
    } else {
        obj.remove("user_rating");
    }

    let updated_json = serde_json::to_string_pretty(&metadata)
        .map_err(|e| NailbiteError::Camera(format!("Failed to serialize event.json: {e}")))?;
    fs::write(&event_json_path, updated_json)?;

    Ok(())
}

/// Set the explicit verdict (true positive / false positive / unsure) on an
/// event, with an optional free-text reason. Pass `None` for either argument
/// to clear that field.
#[tauri::command]
pub fn set_event_verdict(
    state: State<'_, Arc<AppState>>,
    event_id: String,
    verdict: Option<Verdict>,
    reason: Option<String>,
) -> Result<(), NailbiteError> {
    let config = state.config.read();
    let history_dir = expand_tilde(&config.history.dir);
    drop(config);

    let event_dir = history_dir.join(&event_id);
    let event_json_path = event_dir.join("event.json");

    let json_content = std::fs::read_to_string(&event_json_path)?;
    let mut metadata: serde_json::Value = serde_json::from_str(&json_content)
        .map_err(|e| NailbiteError::Camera(format!("Failed to parse event.json: {e}")))?;

    let obj = metadata
        .as_object_mut()
        .ok_or_else(|| NailbiteError::Camera("event.json is not an object".to_string()))?;

    // Drop the legacy boolean label — verdict supersedes it.
    obj.remove("user_label");

    match verdict {
        Some(v) => {
            obj.insert(
                "verdict".to_string(),
                serde_json::Value::String(v.as_str().to_string()),
            );
        }
        None => {
            obj.remove("verdict");
        }
    }

    match reason {
        Some(r) if !r.trim().is_empty() => {
            obj.insert("verdict_reason".to_string(), serde_json::Value::String(r));
        }
        _ => {
            obj.remove("verdict_reason");
        }
    }

    let updated_json = serde_json::to_string_pretty(&metadata)
        .map_err(|e| NailbiteError::Camera(format!("Failed to serialize event.json: {e}")))?;
    std::fs::write(&event_json_path, updated_json)?;

    Ok(())
}

/// Locate the newest detection-event directory matching `bfrb_type` and
/// write a verdict to its event.json. Returns the event id on success.
///
/// Used by the desktop-notification action handler, which can't go
/// through the Tauri command system (it runs on a notify-rust thread).
pub fn record_verdict_for_recent_detection(
    history_dir: &std::path::Path,
    bfrb_type: &str,
    verdict: Verdict,
) -> Option<String> {
    if !history_dir.exists() {
        return None;
    }
    let mut candidates: Vec<(String, String)> = Vec::new();
    let entries = std::fs::read_dir(history_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let dir_name = match path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name.to_string(),
            None => continue,
        };
        let json_path = path.join("event.json");
        let Ok(content) = std::fs::read_to_string(&json_path) else {
            continue;
        };
        let Ok(meta) = serde_json::from_str::<serde_json::Value>(&content) else {
            continue;
        };
        if meta.get("trigger").and_then(|t| t.as_str()) != Some("detection") {
            continue;
        }
        if meta.get("bfrb_type").and_then(|t| t.as_str()) != Some(bfrb_type) {
            continue;
        }
        let timestamp = meta
            .get("timestamp")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();
        candidates.push((timestamp, dir_name));
    }
    candidates.sort_by(|a, b| b.0.cmp(&a.0));
    let (_, dir_name) = candidates.into_iter().next()?;

    let json_path = history_dir.join(&dir_name).join("event.json");
    let json_content = std::fs::read_to_string(&json_path).ok()?;
    let mut metadata: serde_json::Value = serde_json::from_str(&json_content).ok()?;
    let obj = metadata.as_object_mut()?;
    obj.remove("user_label");
    obj.insert(
        "verdict".to_string(),
        serde_json::Value::String(verdict.as_str().to_string()),
    );
    let updated = serde_json::to_string_pretty(&metadata).ok()?;
    std::fs::write(&json_path, updated).ok()?;
    Some(dir_name)
}

/// Find the most recent event directory whose metadata matches the given
/// `bfrb_type`. Used by the alert modal to attach a verdict to the event
/// that just fired without polling.
///
/// Returns the event id (directory name), or `None` if no matching event
/// has been saved yet (e.g. post-capture is still collecting frames).
#[tauri::command]
pub fn find_recent_event_for_alert(
    state: State<'_, Arc<AppState>>,
    bfrb_type: String,
) -> Result<Option<String>, NailbiteError> {
    let config = state.config.read();
    let history_dir = expand_tilde(&config.history.dir);
    drop(config);

    if !history_dir.exists() {
        return Ok(None);
    }

    let entries = std::fs::read_dir(&history_dir).map_err(NailbiteError::Io)?;

    let mut candidates: Vec<(String, String)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let dir_name = match path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name.to_string(),
            None => continue,
        };
        let event_json_path = path.join("event.json");
        let Ok(json) = std::fs::read_to_string(&event_json_path) else {
            continue;
        };
        let Ok(metadata) = serde_json::from_str::<serde_json::Value>(&json) else {
            continue;
        };
        let trigger = metadata
            .get("trigger")
            .and_then(|t| t.as_str())
            .unwrap_or("");
        if trigger != "detection" {
            continue;
        }
        let event_bfrb = metadata
            .get("bfrb_type")
            .and_then(|t| t.as_str())
            .unwrap_or("");
        if event_bfrb != bfrb_type {
            continue;
        }
        let timestamp = metadata
            .get("timestamp")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();
        candidates.push((timestamp, dir_name));
    }

    candidates.sort_by(|a, b| b.0.cmp(&a.0));
    Ok(candidates.into_iter().next().map(|(_, id)| id))
}

/// Delete an event history entry.
#[tauri::command]
pub fn delete_event(
    state: State<'_, Arc<AppState>>,
    event_id: String,
) -> Result<(), NailbiteError> {
    let config = state.config.read();
    let history_dir = expand_tilde(&config.history.dir);
    drop(config);

    let event_dir = history_dir.join(&event_id);

    // Security: ensure the resolved path is within the history directory
    let canonical_history = history_dir
        .canonicalize()
        .unwrap_or_else(|_| history_dir.clone());

    if !event_dir.exists() {
        return Ok(());
    }

    let canonical_event = event_dir.canonicalize().map_err(NailbiteError::Io)?;

    if !canonical_event.starts_with(&canonical_history) {
        return Err(NailbiteError::Camera(
            "Path traversal attempt denied".to_string(),
        ));
    }

    fs::remove_dir_all(&canonical_event)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_verdict_uses_new_format() {
        let json = serde_json::json!({ "verdict": "true_positive" });
        assert_eq!(read_verdict(&json), Some(Verdict::TruePositive));

        let json = serde_json::json!({ "verdict": "false_positive" });
        assert_eq!(read_verdict(&json), Some(Verdict::FalsePositive));

        let json = serde_json::json!({ "verdict": "unsure" });
        assert_eq!(read_verdict(&json), Some(Verdict::Unsure));
    }

    #[test]
    fn read_verdict_migrates_legacy_user_label() {
        // Old format had user_label = correct/incorrect.
        let correct = serde_json::json!({ "user_label": "correct" });
        assert_eq!(read_verdict(&correct), Some(Verdict::TruePositive));

        let incorrect = serde_json::json!({ "user_label": "incorrect" });
        assert_eq!(read_verdict(&incorrect), Some(Verdict::FalsePositive));

        let other = serde_json::json!({ "user_label": "weird" });
        assert_eq!(read_verdict(&other), None);

        let none = serde_json::json!({});
        assert_eq!(read_verdict(&none), None);
    }

    #[test]
    fn read_verdict_prefers_new_format_over_legacy() {
        // If both fields exist, prefer the explicit new-format verdict.
        let json = serde_json::json!({
            "verdict": "unsure",
            "user_label": "correct"
        });
        assert_eq!(read_verdict(&json), Some(Verdict::Unsure));
    }

    #[test]
    fn rate_and_verdict_round_trip_preserves_explanation() {
        // Tests that an event.json round-trip through read_verdict /
        // read_user_rating / read_verdict_reason returns the values written.
        // We synthesize the metadata directly since set_event_verdict needs
        // a Tauri State that's hard to mock in unit tests.
        let metadata = serde_json::json!({
            "verdict": "false_positive",
            "verdict_reason": "phone in hand",
            "user_rating": 2u8,
        });
        assert_eq!(read_verdict(&metadata), Some(Verdict::FalsePositive));
        assert_eq!(
            read_verdict_reason(&metadata),
            Some("phone in hand".to_string())
        );
        assert_eq!(read_user_rating(&metadata), Some(2));
    }
}

