use thiserror::Error;

#[derive(Debug, Error)]
pub enum NailbiteError {
    #[error("Camera error: {0}")]
    Camera(#[from] CameraError),

    #[error("Inference error: {0}")]
    Inference(#[from] InferenceError),

    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("Action error: {0}")]
    Action(#[from] ActionError),

    #[error("UI error: {0}")]
    Ui(#[from] UiError),

    #[error("Training error: {0}")]
    Training(#[from] TrainingError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Error)]
pub enum CameraError {
    #[error("Failed to open camera device {device}: {source}")]
    OpenFailed {
        device: String,
        source: std::io::Error,
    },

    #[error("Failed to capture frame from {device}: {source}")]
    CaptureFailed {
        device: String,
        source: std::io::Error,
    },

    #[error("Unsupported pixel format: {0}")]
    UnsupportedFormat(String),

    #[error("Camera device not found: {0}")]
    DeviceNotFound(String),

    #[error("Camera configuration error: {0}")]
    Configuration(String),
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
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Failed to read config file {path}: {source}")]
    ReadFailed {
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

    #[error("Notification error: {0}")]
    Notification(String),

    #[error("Webhook error: {0}")]
    Webhook(String),
}

#[derive(Debug, Error)]
pub enum UiError {
    #[error("Tray icon error: {0}")]
    TrayIcon(String),

    #[error("Window creation error: {0}")]
    Window(String),
}

#[derive(Debug, Error)]
pub enum TrainingError {
    #[error("Failed to write annotation: {0}")]
    WriteFailed(String),

    #[error("Failed to save frame: {0}")]
    FrameSaveFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_error_display() {
        let err = CameraError::DeviceNotFound("/dev/video0".to_string());
        assert_eq!(err.to_string(), "Camera device not found: /dev/video0");
    }

    #[test]
    fn config_error_display() {
        let err = ConfigError::Validation("inference_fps must be <= fps".to_string());
        assert_eq!(
            err.to_string(),
            "Invalid configuration: inference_fps must be <= fps"
        );
    }

    #[test]
    fn inference_error_display() {
        let err = InferenceError::ModelNotFound {
            path: "models/hand.onnx".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Model file not found: models/hand.onnx"
        );
    }

    #[test]
    fn action_error_display() {
        let err = ActionError::Sound("device not found".to_string());
        assert_eq!(err.to_string(), "Sound playback error: device not found");
    }

    #[test]
    fn nailbite_error_from_config() {
        let config_err = ConfigError::Validation("bad value".to_string());
        let err: NailbiteError = config_err.into();
        assert!(matches!(err, NailbiteError::Config(_)));
    }

    #[test]
    fn nailbite_error_from_camera() {
        let cam_err = CameraError::DeviceNotFound("/dev/video0".to_string());
        let err: NailbiteError = cam_err.into();
        assert!(matches!(err, NailbiteError::Camera(_)));
    }

    #[test]
    fn nailbite_error_from_inference() {
        let inf_err = InferenceError::ModelNotFound {
            path: "test.onnx".to_string(),
        };
        let err: NailbiteError = inf_err.into();
        assert!(matches!(err, NailbiteError::Inference(_)));
    }

    #[test]
    fn training_error_display() {
        let err = TrainingError::WriteFailed("disk full".to_string());
        assert_eq!(err.to_string(), "Failed to write annotation: disk full");
    }
}
