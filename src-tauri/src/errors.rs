//! Error types for the Nailbite Tauri app.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum NailbiteError {
    #[error("Inference error: {0}")]
    Inference(#[from] InferenceError),

    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("Action error: {0}")]
    Action(#[from] ActionError),

    #[error("Training error: {0}")]
    Training(#[from] TrainingError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),

    #[error("Tauri error: {0}")]
    Tauri(String),

    #[error("Camera error: {0}")]
    Camera(String),
}

impl From<tauri::Error> for NailbiteError {
    fn from(e: tauri::Error) -> Self {
        NailbiteError::Tauri(e.to_string())
    }
}

impl serde::Serialize for NailbiteError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("ONNX Runtime error: {0}")]
    Ort(String),

    #[error("Model file not found: {path}")]
    ModelNotFound { path: String },

    #[error("Model download failed for {model_name}: {reason}")]
    DownloadFailed {
        model_name: String,
        reason: String,
    },

    #[error("Failed to create model directory {path}: {source}")]
    DirectoryCreation {
        path: String,
        source: std::io::Error,
    },

    #[error("Failed to write model file {path}: {source}")]
    WriteFailed {
        path: String,
        source: std::io::Error,
    },

    #[error("Invalid model output shape: expected {expected}, got {actual}")]
    InvalidOutputShape { expected: String, actual: String },

    #[error("Preprocessing error: {0}")]
    Preprocessing(String),

    #[error("GPU required but not available: {reason}")]
    GpuRequired { reason: String },

    #[error("GPU initialization failed: {reason}")]
    GpuInitFailed { reason: String },
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Failed to read config file {path}: {source}")]
    ReadFailed {
        path: String,
        source: std::io::Error,
    },

    #[error("Failed to write config file {path}: {source}")]
    WriteFailed {
        path: String,
        source: std::io::Error,
    },

    #[error("Failed to parse config: {0}")]
    ParseFailed(String),

    #[error("Invalid configuration: {0}")]
    Validation(String),
}

#[derive(Debug, Error)]
pub enum ActionError {
    #[error("Sound playback error: {0}")]
    Sound(String),

    #[error("Webhook error: {0}")]
    Webhook(String),
}

#[derive(Debug, Error)]
pub enum TrainingError {
    #[error("Failed to write annotation: {0}")]
    WriteFailed(String),

    #[error("Failed to save frame: {0}")]
    FrameSaveFailed(String),
}
