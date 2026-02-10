//! Exercise commands.

use std::sync::Arc;

use serde::Serialize;
use tauri::State;

use crate::detection::types::BfrbType;
use crate::errors::NailbiteError;
use crate::exercises::types::{Exercise, ExerciseCategory};
use crate::frame::Frame;
use crate::state::AppState;

/// Exercise info for the frontend.
#[derive(Debug, Clone, Serialize)]
pub struct ExerciseInfo {
    pub id: String,
    pub name: String,
    pub instructions: String,
    pub category: String,
    pub hold_duration_secs: u64,
    pub target_reps: u32,
}

impl From<&dyn Exercise> for ExerciseInfo {
    fn from(e: &dyn Exercise) -> Self {
        Self {
            id: e.id().to_string(),
            name: e.name().to_string(),
            instructions: e.instructions().to_string(),
            category: match e.category() {
                ExerciseCategory::TimedHold => "timed_hold".to_string(),
                ExerciseCategory::Repetitions => "repetitions".to_string(),
            },
            hold_duration_secs: e.hold_duration().as_secs(),
            target_reps: e.target_reps(),
        }
    }
}

/// Verification result for the frontend.
#[derive(Debug, Clone, Serialize)]
pub struct VerificationResultInfo {
    pub pose_correct: bool,
    pub feedback: String,
    pub progress: f32,
}

/// Get an exercise for the given BFRB type.
#[tauri::command]
pub fn get_exercise(
    bfrb_type: String,
    state: State<'_, Arc<AppState>>,
) -> Result<ExerciseInfo, NailbiteError> {
    let bfrb = match bfrb_type.as_str() {
        "nail_biting" => BfrbType::NailBiting,
        "nail_picking" => BfrbType::NailPicking,
        "hair_pulling" => BfrbType::HairPulling,
        "skin_picking" => BfrbType::SkinPicking,
        "lip_biting" => BfrbType::LipBiting,
        _ => {
            return Err(NailbiteError::Config(crate::errors::ConfigError::Validation(
                format!("Unknown BFRB type: {}", bfrb_type),
            )));
        }
    };

    let config = state.config.read();

    // Create a simple exercise registry and get an exercise
    let registry = crate::exercises::registry::ExerciseRegistry::new(
        config.exercises.selection_strategy,
        config.exercises.preferred_exercise.clone(),
    );
    let exercise = registry.select(bfrb).ok_or_else(|| {
        NailbiteError::Config(crate::errors::ConfigError::Validation(
            format!("No exercise available for BFRB type: {}", bfrb_type),
        ))
    })?;

    Ok(ExerciseInfo::from(exercise))
}

/// Verify an exercise frame.
#[tauri::command]
pub fn verify_exercise_frame(
    data: Vec<u8>,
    exercise_id: String,
    state: State<'_, Arc<AppState>>,
) -> Result<VerificationResultInfo, NailbiteError> {
    // Decode frame
    let frame = Frame::from_jpeg(&data)?;

    // Run hand detection for verification
    let palm_detector =
        crate::inference::palm_detection::PalmDetector::new(Arc::clone(&state.sessions.palm_detection));
    let hand_landmarker =
        crate::inference::hand_landmark::HandLandmarker::new(Arc::clone(&state.sessions.hand_landmark));
    let face_detector =
        crate::inference::face_detection::FaceDetector::new(Arc::clone(&state.sessions.face_detection));
    let face_mesher =
        crate::inference::face_mesh::FaceMesher::new(Arc::clone(&state.sessions.face_mesh));

    let mut hands = Vec::new();
    let mut face = None;

    // Face pipeline
    if let Ok(face_result) = face_detector.detect(&frame) {
        if let Some(roi) = face_result.face_rois.first() {
            if let Ok(mesh) = face_mesher.estimate(&frame, roi) {
                face = mesh;
            }
        }
    }

    // Hand pipeline
    if let Ok(palm_result) = palm_detector.detect(&frame) {
        for roi in &palm_result.hand_rois {
            if let Ok(Some(hand)) = hand_landmarker.estimate(&frame, roi) {
                hands.push(hand);
            }
        }
    }

    // Get exercise and verify
    let config = state.config.read();
    let registry = crate::exercises::registry::ExerciseRegistry::new(
        config.exercises.selection_strategy,
        config.exercises.preferred_exercise.clone(),
    );

    let exercise = registry
        .all()
        .iter()
        .find(|e| e.id() == exercise_id)
        .ok_or_else(|| {
            NailbiteError::Config(crate::errors::ConfigError::Validation(
                format!("Exercise not found: {}", exercise_id),
            ))
        })?;

    let result = exercise.verify(&hands, &face);

    Ok(VerificationResultInfo {
        pose_correct: result.pose_correct,
        feedback: result.feedback,
        progress: result.confidence,  // Using confidence as progress indicator
    })
}
