/// ONNX Runtime session management.
///
/// Handles loading models, configuring thread counts, and optimization levels.
use std::path::Path;
use std::sync::{Arc, Mutex};

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use tracing::info;

use crate::config::OrtConfig;
use crate::errors::InferenceError;

/// Wrapper around an ort::Session with shared configuration.
///
/// The session is behind a `Mutex` because `Session::run()` requires `&mut self`.
pub struct ModelSession {
    session: Mutex<Session>,
}

impl ModelSession {
    /// Load an ONNX model from disk and create an inference session.
    pub fn new(model_path: &Path, ort_config: &OrtConfig) -> Result<Self, InferenceError> {
        if !model_path.exists() {
            return Err(InferenceError::ModelNotFound {
                path: model_path.display().to_string(),
            });
        }

        info!(model = %model_path.display(), "Loading ONNX model");

        let session = Session::builder()
            .map_err(|e| InferenceError::Ort(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| InferenceError::Ort(e.to_string()))?
            .with_intra_threads(ort_config.intra_op_num_threads as usize)
            .map_err(|e| InferenceError::Ort(e.to_string()))?
            .with_inter_threads(ort_config.inter_op_num_threads as usize)
            .map_err(|e| InferenceError::Ort(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| InferenceError::Ort(e.to_string()))?;

        Ok(Self {
            session: Mutex::new(session),
        })
    }

    /// Lock the session and run inference. Extracts outputs via a closure
    /// to ensure the lock is released after data is extracted.
    pub fn run_inference<F, T>(&self, f: F) -> Result<T, InferenceError>
    where
        F: FnOnce(&mut Session) -> Result<T, InferenceError>,
    {
        let mut guard = self
            .session
            .lock()
            .map_err(|e| InferenceError::Ort(format!("Session lock poisoned: {e}")))?;
        f(&mut guard)
    }
}

/// Load all four model sessions from the configured paths.
pub struct ModelSessions {
    pub palm_detection: Arc<ModelSession>,
    pub hand_landmark: Arc<ModelSession>,
    pub face_detection: Arc<ModelSession>,
    pub face_mesh: Arc<ModelSession>,
}

impl ModelSessions {
    pub fn load(
        models: &crate::config::ModelsConfig,
        ort_config: &OrtConfig,
    ) -> Result<Self, InferenceError> {
        info!("Loading all ONNX model sessions");

        let palm_detection = Arc::new(ModelSession::new(&models.palm_detection, ort_config)?);
        let hand_landmark = Arc::new(ModelSession::new(&models.hand_landmark, ort_config)?);
        let face_detection = Arc::new(ModelSession::new(&models.face_detection, ort_config)?);
        let face_mesh = Arc::new(ModelSession::new(&models.face_mesh, ort_config)?);

        info!("All model sessions loaded successfully");

        Ok(Self {
            palm_detection,
            hand_landmark,
            face_detection,
            face_mesh,
        })
    }
}
