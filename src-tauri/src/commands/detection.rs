//! Detection commands for model management.
//!
//! Detection processing is handled by the V4L2 camera pipeline which runs
//! in a dedicated thread and emits frame-update events to the frontend.

use std::sync::Arc;

use serde::Serialize;
use tauri::State;

use crate::detection::types::{FaceDetection, HandDetection, Landmark, PoseDetection, PoseLandmark};
use crate::errors::NailbiteError;
use crate::state::AppState;

/// Simplified hand detection for IPC.
#[derive(Debug, Clone, Serialize)]
pub struct HandResult {
    pub landmarks: Vec<LandmarkResult>,
    pub handedness: String,
    pub confidence: f32,
}

/// Simplified face detection for IPC.
#[derive(Debug, Clone, Serialize)]
pub struct FaceResult {
    pub landmarks: Vec<LandmarkResult>,
    pub confidence: f32,
}

/// Simplified pose detection for IPC.
#[derive(Debug, Clone, Serialize)]
pub struct PoseResult {
    pub landmarks: Vec<PoseLandmarkResult>,
    pub confidence: f32,
}

/// Simplified pose landmark for IPC.
#[derive(Debug, Clone, Serialize)]
pub struct PoseLandmarkResult {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub visibility: f32,
    pub presence: f32,
}

/// Simplified landmark for IPC.
#[derive(Debug, Clone, Serialize)]
pub struct LandmarkResult {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Simplified detection event for IPC.
#[derive(Debug, Clone, Serialize)]
pub struct DetectionEventResult {
    pub bfrb_type: String,
    pub confidence: f32,
    pub timestamp: String,
    /// Contributing signals at the moment of detection. `None` for legacy
    /// callers / when explanation was not produced.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explanation: Option<crate::detection::types::DetectionExplanation>,
    /// Stable directory id of the saved event in event history (matches
    /// `EventHistorySummary::id`). Lets the frontend label this exact event.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_id: Option<String>,
}

impl From<&Landmark> for LandmarkResult {
    fn from(l: &Landmark) -> Self {
        Self {
            x: l.x,
            y: l.y,
            z: l.z,
        }
    }
}

impl From<&HandDetection> for HandResult {
    fn from(h: &HandDetection) -> Self {
        Self {
            landmarks: h.landmarks.iter().map(LandmarkResult::from).collect(),
            handedness: h.side.map(|s| format!("{:?}", s).to_lowercase()).unwrap_or_else(|| "unknown".to_string()),
            confidence: h.confidence,
        }
    }
}

impl From<&FaceDetection> for FaceResult {
    fn from(f: &FaceDetection) -> Self {
        Self {
            landmarks: f.landmarks.iter().map(LandmarkResult::from).collect(),
            confidence: f.confidence,
        }
    }
}

impl From<&PoseLandmark> for PoseLandmarkResult {
    fn from(pl: &PoseLandmark) -> Self {
        Self {
            x: pl.landmark.x,
            y: pl.landmark.y,
            z: pl.landmark.z,
            visibility: pl.visibility,
            presence: pl.presence,
        }
    }
}

impl From<&PoseDetection> for PoseResult {
    fn from(p: &PoseDetection) -> Self {
        Self {
            landmarks: p.landmarks.iter().map(PoseLandmarkResult::from).collect(),
            confidence: p.confidence,
        }
    }
}


/// Ensure all ONNX models are downloaded.
#[tauri::command]
pub fn ensure_models(state: State<'_, Arc<AppState>>) -> Result<(), NailbiteError> {
    let config = state.config.read();
    crate::inference::model_downloader::ensure_models(&config.models)?;
    Ok(())
}
