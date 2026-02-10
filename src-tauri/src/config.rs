//! Configuration for the Nailbite Tauri app.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::errors::ConfigError;

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct NailbiteConfig {
    #[serde(default)]
    pub general: GeneralConfig,
    #[serde(default)]
    pub models: ModelsConfig,
    #[serde(default)]
    pub camera: CameraConfig,
    #[serde(default)]
    pub ort: OrtConfig,
    #[serde(default)]
    pub detection: DetectionConfig,
    #[serde(default)]
    pub fusion: FusionConfig,
    #[serde(default)]
    pub actions: ActionsConfig,
    #[serde(default)]
    pub exercises: ExercisesConfig,
    #[serde(default)]
    pub hotkeys: HotkeysConfig,
    #[serde(default)]
    pub training: TrainingConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GeneralConfig {
    #[serde(default = "default_log_level")]
    pub log_level: String,
    #[serde(default)]
    pub show_preview: bool,
    #[serde(default = "default_cooldown_seconds")]
    pub cooldown_seconds: u64,
    #[serde(default = "default_stats_file")]
    pub stats_file: PathBuf,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            log_level: default_log_level(),
            show_preview: false,
            cooldown_seconds: default_cooldown_seconds(),
            stats_file: default_stats_file(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelsConfig {
    #[serde(default = "default_palm_detection_path")]
    pub palm_detection: PathBuf,
    #[serde(default = "default_hand_landmark_path")]
    pub hand_landmark: PathBuf,
    #[serde(default = "default_face_detection_path")]
    pub face_detection: PathBuf,
    #[serde(default = "default_face_mesh_path")]
    pub face_mesh: PathBuf,
    #[serde(default = "default_pose_detection_path")]
    pub pose_detection: PathBuf,
    #[serde(default = "default_pose_landmark_path")]
    pub pose_landmark: PathBuf,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            palm_detection: default_palm_detection_path(),
            hand_landmark: default_hand_landmark_path(),
            face_detection: default_face_detection_path(),
            face_mesh: default_face_mesh_path(),
            pose_detection: default_pose_detection_path(),
            pose_landmark: default_pose_landmark_path(),
        }
    }
}

/// Camera config for V4L2 capture.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CameraConfig {
    /// List of camera sources.
    #[serde(default = "default_cameras")]
    pub sources: Vec<CameraSource>,
    /// Inference FPS (shared across all cameras).
    #[serde(default = "default_inference_fps")]
    pub inference_fps: u32,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            sources: default_cameras(),
            inference_fps: default_inference_fps(),
        }
    }
}

/// Individual camera source configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CameraSource {
    /// Unique identifier for this camera.
    #[serde(default = "default_camera_id")]
    pub id: String,
    /// V4L2 device path (Linux).
    #[serde(default = "default_camera_device")]
    pub device: String,
    /// Camera role: "primary" for detection, "auxiliary" for additional views.
    #[serde(default = "default_camera_role")]
    pub role: CameraRole,
    /// Resolution width.
    #[serde(default = "default_camera_width")]
    pub resolution_width: u32,
    /// Resolution height.
    #[serde(default = "default_camera_height")]
    pub resolution_height: u32,
}

impl Default for CameraSource {
    fn default() -> Self {
        Self {
            id: default_camera_id(),
            device: default_camera_device(),
            role: CameraRole::Primary,
            resolution_width: default_camera_width(),
            resolution_height: default_camera_height(),
        }
    }
}

/// Camera role in the detection system.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CameraRole {
    /// Primary camera used for BFRB detection.
    #[default]
    Primary,
    /// Auxiliary camera for additional viewing angles.
    Auxiliary,
}

/// GPU preference for inference acceleration.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GpuPreference {
    /// Try GPU, fall back to CPU if unavailable.
    #[default]
    Auto,
    /// CPU only, never use GPU.
    Disabled,
    /// Require GPU, fail if unavailable.
    Required,
}

/// GPU backend selection.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GpuBackend {
    /// Try TensorRT > CUDA > MIGraphX > ROCm.
    #[default]
    Auto,
    /// NVIDIA CUDA only.
    Cuda,
    /// NVIDIA TensorRT (falls back to CUDA).
    TensorRt,
    /// AMD MIGraphX only.
    MiGraphX,
}

/// GPU-specific configuration options.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GpuConfig {
    /// Whether to use GPU acceleration.
    #[serde(default)]
    pub preference: GpuPreference,
    /// Which GPU backend to use.
    #[serde(default)]
    pub backend: GpuBackend,
    /// GPU device index (0-7).
    #[serde(default)]
    pub device_id: u32,
    /// Enable FP16 inference for supported backends.
    #[serde(default = "default_true")]
    pub fp16_enable: bool,
    /// Optional memory limit in MB (None = no limit).
    #[serde(default)]
    pub memory_limit_mb: Option<u32>,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            preference: GpuPreference::Auto,
            backend: GpuBackend::Auto,
            device_id: 0,
            fp16_enable: true,
            memory_limit_mb: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OrtConfig {
    #[serde(default = "default_intra_op_threads")]
    pub intra_op_num_threads: u32,
    #[serde(default = "default_inter_op_threads")]
    pub inter_op_num_threads: u32,
    #[serde(default)]
    pub gpu: GpuConfig,
}

impl Default for OrtConfig {
    fn default() -> Self {
        Self {
            intra_op_num_threads: default_intra_op_threads(),
            inter_op_num_threads: default_inter_op_threads(),
            gpu: GpuConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct DetectionConfig {
    #[serde(default)]
    pub behaviors: BehaviorsConfig,
    #[serde(default)]
    pub temporal: TemporalConfig,
    #[serde(default)]
    pub false_positive: FalsePositiveConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BehaviorsConfig {
    #[serde(default = "default_nail_biting_config")]
    pub nail_biting: BehaviorConfig,
    #[serde(default = "default_nail_picking_config")]
    pub nail_picking: BehaviorConfig,
    #[serde(default)]
    pub hair_pulling: BehaviorConfig,
    #[serde(default)]
    pub skin_picking: BehaviorConfig,
    #[serde(default)]
    pub lip_biting: BehaviorConfig,
}

impl Default for BehaviorsConfig {
    fn default() -> Self {
        Self {
            nail_biting: default_nail_biting_config(),
            nail_picking: default_nail_picking_config(),
            hair_pulling: BehaviorConfig::default(),
            skin_picking: BehaviorConfig::default(),
            lip_biting: BehaviorConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BehaviorConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_proximity_threshold")]
    pub proximity_threshold: f32,
    #[serde(default = "default_min_sustained_ms")]
    pub min_sustained_ms: u64,
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f32,
}

impl Default for BehaviorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            proximity_threshold: default_proximity_threshold(),
            min_sustained_ms: default_min_sustained_ms(),
            confidence_threshold: default_confidence_threshold(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TemporalConfig {
    #[serde(default = "default_window_ms")]
    pub window_ms: u64,
    #[serde(default = "default_positive_ratio")]
    pub positive_ratio: f32,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            window_ms: default_window_ms(),
            positive_ratio: default_positive_ratio(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FalsePositiveConfig {
    #[serde(default = "default_true")]
    pub typing_suppression: bool,
    #[serde(default = "default_true")]
    pub chin_rest_suppression: bool,
    #[serde(default = "default_true")]
    pub eating_suppression: bool,
}

impl Default for FalsePositiveConfig {
    fn default() -> Self {
        Self {
            typing_suppression: true,
            chin_rest_suppression: true,
            eating_suppression: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FusionConfig {
    #[serde(default)]
    pub strategy: FusionStrategy,
    #[serde(default = "default_merge_tolerance_ms")]
    pub merge_tolerance_ms: u64,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            strategy: FusionStrategy::default(),
            merge_tolerance_ms: default_merge_tolerance_ms(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FusionStrategy {
    #[default]
    Any,
    Merge,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ActionsConfig {
    #[serde(default)]
    pub sound: SoundConfig,
    #[serde(default)]
    pub webhook: WebhookConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SoundConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_sound_file")]
    pub file: String,
    #[serde(default = "default_volume")]
    pub volume: f32,
    #[serde(default = "default_true")]
    pub repeat: bool,
}

impl Default for SoundConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            file: default_sound_file(),
            volume: default_volume(),
            repeat: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WebhookConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub url: String,
    #[serde(default = "default_webhook_timeout_ms")]
    pub timeout_ms: u64,
    #[serde(default)]
    pub headers: std::collections::HashMap<String, String>,
}

impl Default for WebhookConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            url: String::new(),
            timeout_ms: default_webhook_timeout_ms(),
            headers: std::collections::HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExercisesConfig {
    #[serde(default)]
    pub selection_strategy: SelectionStrategy,
    #[serde(default)]
    pub preferred_exercise: Option<String>,
    #[serde(default)]
    pub hold_duration_override: Option<u64>,
    #[serde(default)]
    pub reps_override: Option<u32>,
    #[serde(default = "default_exercise_timeout")]
    pub timeout_seconds: u64,
    #[serde(default = "default_compliance_ratio")]
    pub compliance_ratio: f32,
}

impl Default for ExercisesConfig {
    fn default() -> Self {
        Self {
            selection_strategy: SelectionStrategy::default(),
            preferred_exercise: None,
            hold_duration_override: None,
            reps_override: None,
            timeout_seconds: default_exercise_timeout(),
            compliance_ratio: default_compliance_ratio(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SelectionStrategy {
    #[default]
    Random,
    First,
    RoundRobin,
    Preferred,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HotkeysConfig {
    #[serde(default = "default_dismiss_key")]
    pub dismiss_false_positive: String,
    #[serde(default = "default_missed_key")]
    pub mark_missed_event: String,
    #[serde(default = "default_pause_key")]
    pub pause_resume: String,
}

impl Default for HotkeysConfig {
    fn default() -> Self {
        Self {
            dismiss_false_positive: default_dismiss_key(),
            mark_missed_event: default_missed_key(),
            pause_resume: default_pause_key(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrainingConfig {
    #[serde(default)]
    pub save_frames: bool,
    #[serde(default = "default_true")]
    pub save_landmarks: bool,
    #[serde(default = "default_annotations_file")]
    pub annotations_file: PathBuf,
    #[serde(default = "default_frames_dir")]
    pub frames_dir: PathBuf,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            save_frames: false,
            save_landmarks: true,
            annotations_file: default_annotations_file(),
            frames_dir: default_frames_dir(),
        }
    }
}

// --- Default value functions ---

fn default_log_level() -> String {
    "info".to_string()
}
fn default_cooldown_seconds() -> u64 {
    30
}
fn default_stats_file() -> PathBuf {
    PathBuf::from("~/.local/share/nailbite/stats.jsonl")
}
fn default_palm_detection_path() -> PathBuf {
    PathBuf::from("./models/palm_detection.onnx")
}
fn default_hand_landmark_path() -> PathBuf {
    PathBuf::from("./models/hand_landmark.onnx")
}
fn default_face_detection_path() -> PathBuf {
    PathBuf::from("./models/face_detection.onnx")
}
fn default_face_mesh_path() -> PathBuf {
    PathBuf::from("./models/face_mesh.onnx")
}
fn default_pose_detection_path() -> PathBuf {
    PathBuf::from("./models/pose_detection.onnx")
}
fn default_pose_landmark_path() -> PathBuf {
    PathBuf::from("./models/pose_landmark.onnx")
}
fn default_cameras() -> Vec<CameraSource> {
    vec![CameraSource::default()]
}
fn default_camera_id() -> String {
    "main".to_string()
}
fn default_camera_device() -> String {
    "/dev/video0".to_string()
}
fn default_camera_role() -> CameraRole {
    CameraRole::Primary
}
fn default_camera_width() -> u32 {
    640
}
fn default_camera_height() -> u32 {
    480
}
fn default_inference_fps() -> u32 {
    8
}
fn default_intra_op_threads() -> u32 {
    2
}
fn default_inter_op_threads() -> u32 {
    1
}
fn default_proximity_threshold() -> f32 {
    0.35
}
fn default_min_sustained_ms() -> u64 {
    1500
}
fn default_confidence_threshold() -> f32 {
    0.3
}
fn default_window_ms() -> u64 {
    1500
}
fn default_positive_ratio() -> f32 {
    0.5
}
fn default_merge_tolerance_ms() -> u64 {
    100
}
fn default_true() -> bool {
    true
}
fn default_sound_file() -> String {
    "builtin".to_string()
}
fn default_volume() -> f32 {
    0.8
}
fn default_webhook_timeout_ms() -> u64 {
    5000
}
fn default_exercise_timeout() -> u64 {
    120
}
fn default_compliance_ratio() -> f32 {
    0.8
}
fn default_dismiss_key() -> String {
    "F9".to_string()
}
fn default_missed_key() -> String {
    "F10".to_string()
}
fn default_pause_key() -> String {
    "F11".to_string()
}
fn default_annotations_file() -> PathBuf {
    PathBuf::from("~/.local/share/nailbite/annotations.jsonl")
}
fn default_frames_dir() -> PathBuf {
    PathBuf::from("~/.local/share/nailbite/frames/")
}

fn default_nail_biting_config() -> BehaviorConfig {
    BehaviorConfig {
        enabled: true,
        proximity_threshold: 0.35,
        min_sustained_ms: 1500,
        confidence_threshold: 0.3,
    }
}

fn default_nail_picking_config() -> BehaviorConfig {
    BehaviorConfig {
        enabled: true,
        proximity_threshold: 0.15,
        min_sustained_ms: 1500,
        confidence_threshold: 0.3,
    }
}

// --- Loading and validation ---

impl NailbiteConfig {
    pub fn load(path: &str) -> Result<Self, ConfigError> {
        let contents = std::fs::read_to_string(path).map_err(|e| ConfigError::ReadFailed {
            path: path.to_string(),
            source: e,
        })?;
        let config: Self =
            serde_yaml_ng::from_str(&contents).map_err(|e| ConfigError::ParseFailed(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn save(&self, path: &str) -> Result<(), ConfigError> {
        let contents = serde_yaml_ng::to_string(self)
            .map_err(|e| ConfigError::ParseFailed(e.to_string()))?;
        std::fs::write(path, contents).map_err(|e| ConfigError::WriteFailed {
            path: path.to_string(),
            source: e,
        })?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.detection.temporal.positive_ratio <= 0.0
            || self.detection.temporal.positive_ratio > 1.0
        {
            return Err(ConfigError::Validation(
                "detection.temporal.positive_ratio must be in (0.0, 1.0]".to_string(),
            ));
        }

        if self.exercises.compliance_ratio <= 0.0 || self.exercises.compliance_ratio > 1.0 {
            return Err(ConfigError::Validation(
                "exercises.compliance_ratio must be in (0.0, 1.0]".to_string(),
            ));
        }

        if self.actions.sound.volume < 0.0 || self.actions.sound.volume > 1.0 {
            return Err(ConfigError::Validation(
                "actions.sound.volume must be in [0.0, 1.0]".to_string(),
            ));
        }

        if self.actions.webhook.enabled && self.actions.webhook.url.is_empty() {
            return Err(ConfigError::Validation(
                "actions.webhook.url must be set when webhook is enabled".to_string(),
            ));
        }

        if self.exercises.selection_strategy == SelectionStrategy::Preferred
            && self.exercises.preferred_exercise.is_none()
        {
            return Err(ConfigError::Validation(
                "exercises.preferred_exercise must be set when selection_strategy is 'preferred'"
                    .to_string(),
            ));
        }

        if self.camera.sources.is_empty() {
            return Err(ConfigError::Validation(
                "camera.sources must have at least one camera".to_string(),
            ));
        }

        // Check that there's exactly one primary camera
        let primary_count = self.camera.sources.iter()
            .filter(|s| s.role == CameraRole::Primary)
            .count();
        if primary_count != 1 {
            return Err(ConfigError::Validation(
                format!("camera.sources must have exactly one primary camera (found {})", primary_count),
            ));
        }

        // Validate each camera source
        for source in &self.camera.sources {
            if source.resolution_width == 0 || source.resolution_height == 0 {
                return Err(ConfigError::Validation(
                    format!("camera '{}': resolution_width and resolution_height must be > 0", source.id),
                ));
            }
            if source.id.is_empty() {
                return Err(ConfigError::Validation(
                    "camera source id must not be empty".to_string(),
                ));
            }
        }

        // Check for duplicate camera IDs
        let mut seen_ids = std::collections::HashSet::new();
        for source in &self.camera.sources {
            if !seen_ids.insert(&source.id) {
                return Err(ConfigError::Validation(
                    format!("duplicate camera id: '{}'", source.id),
                ));
            }
        }

        if self.ort.gpu.device_id > 7 {
            return Err(ConfigError::Validation(
                "ort.gpu.device_id must be in range 0-7".to_string(),
            ));
        }

        Ok(())
    }
}

