//! ONNX Runtime session management.
//!
//! Handles loading models, configuring thread counts, optimization levels,
//! and GPU execution providers.

use std::path::Path;
use std::sync::{Arc, Mutex};

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use tracing::info;

use crate::config::OrtConfig;
use crate::errors::InferenceError;

use super::execution_provider::{build_execution_providers, ActiveProvider};

/// Wrapper around an ort::Session with shared configuration.
///
/// The session is behind a `Mutex` because `Session::run()` requires `&mut self`.
pub struct ModelSession {
    session: Mutex<Session>,
    active_provider: ActiveProvider,
}

impl ModelSession {
    /// Load an ONNX model from disk and create an inference session.
    ///
    /// Configures GPU execution providers based on the ORT configuration,
    /// with automatic fallback to CPU if GPU is not available.
    pub fn new(model_path: &Path, ort_config: &OrtConfig) -> Result<Self, InferenceError> {
        if !model_path.exists() {
            return Err(InferenceError::ModelNotFound {
                path: model_path.display().to_string(),
            });
        }

        info!(model = %model_path.display(), "Loading ONNX model");

        // Build execution providers based on GPU config
        let (providers, expected_provider) = build_execution_providers(&ort_config.gpu)?;

        let mut builder = Session::builder()
            .map_err(|e| InferenceError::Ort(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| InferenceError::Ort(e.to_string()))?
            .with_intra_threads(ort_config.intra_op_num_threads as usize)
            .map_err(|e| InferenceError::Ort(e.to_string()))?
            .with_inter_threads(ort_config.inter_op_num_threads as usize)
            .map_err(|e| InferenceError::Ort(e.to_string()))?;

        // Register execution providers if any GPU providers are available
        if !providers.is_empty() {
            builder = builder
                .with_execution_providers(providers)
                .map_err(|e| InferenceError::Ort(e.to_string()))?;
            info!(provider = %expected_provider, "Registered GPU execution providers");
        }

        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| InferenceError::Ort(e.to_string()))?;

        // Determine actual provider used
        let active_provider = if expected_provider == ActiveProvider::Cpu {
            ActiveProvider::Cpu
        } else {
            expected_provider
        };

        info!(
            model = %model_path.display(),
            provider = %active_provider,
            "Model session created"
        );

        Ok(Self {
            session: Mutex::new(session),
            active_provider,
        })
    }

    /// Get the active execution provider for this session.
    pub fn active_provider(&self) -> ActiveProvider {
        self.active_provider
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

/// Load all model sessions from the configured paths.
pub struct ModelSessions {
    pub palm_detection: Arc<ModelSession>,
    pub hand_landmark: Arc<ModelSession>,
    pub face_detection: Arc<ModelSession>,
    pub face_mesh: Arc<ModelSession>,
    pub pose_detection: Arc<ModelSession>,
    pub pose_landmark: Arc<ModelSession>,
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
        let pose_detection = Arc::new(ModelSession::new(&models.pose_detection, ort_config)?);
        let pose_landmark = Arc::new(ModelSession::new(&models.pose_landmark, ort_config)?);

        // Log active provider (all sessions should use the same provider)
        info!(
            provider = %palm_detection.active_provider(),
            "All model sessions loaded successfully"
        );

        Ok(Self {
            palm_detection,
            hand_landmark,
            face_detection,
            face_mesh,
            pose_detection,
            pose_landmark,
        })
    }

    /// Get the active execution provider for the sessions.
    ///
    /// Returns the provider from the first session (palm_detection),
    /// as all sessions are configured with the same provider.
    pub fn active_provider(&self) -> ActiveProvider {
        self.palm_detection.active_provider()
    }
}
