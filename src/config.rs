use serde::Deserialize;
use std::path::PathBuf;

use crate::errors::ConfigError;

#[derive(Debug, Clone, Deserialize)]
pub struct NailbiteConfig {
    #[serde(default)]
    pub general: GeneralConfig,
    #[serde(default)]
    pub models: ModelsConfig,
    #[serde(default)]
    pub cameras: Vec<CameraConfig>,
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

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
pub struct ModelsConfig {
    #[serde(default = "default_palm_detection_path")]
    pub palm_detection: PathBuf,
    #[serde(default = "default_hand_landmark_path")]
    pub hand_landmark: PathBuf,
    #[serde(default = "default_face_detection_path")]
    pub face_detection: PathBuf,
    #[serde(default = "default_face_mesh_path")]
    pub face_mesh: PathBuf,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            palm_detection: default_palm_detection_path(),
            hand_landmark: default_hand_landmark_path(),
            face_detection: default_face_detection_path(),
            face_mesh: default_face_mesh_path(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CameraConfig {
    pub id: String,
    #[serde(default = "default_device")]
    pub device: String,
    #[serde(default)]
    pub role: CameraRole,
    #[serde(default = "default_camera_width")]
    pub width: u32,
    #[serde(default = "default_camera_height")]
    pub height: u32,
    #[serde(default = "default_camera_fps")]
    pub fps: u32,
    #[serde(default = "default_inference_fps")]
    pub inference_fps: u32,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CameraRole {
    #[default]
    Full,
    FaceOnly,
    HandsOnly,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OrtConfig {
    #[serde(default = "default_intra_op_threads")]
    pub intra_op_num_threads: u32,
    #[serde(default = "default_inter_op_threads")]
    pub inter_op_num_threads: u32,
}

impl Default for OrtConfig {
    fn default() -> Self {
        Self {
            intra_op_num_threads: default_intra_op_threads(),
            inter_op_num_threads: default_inter_op_threads(),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct DetectionConfig {
    #[serde(default)]
    pub behaviors: BehaviorsConfig,
    #[serde(default)]
    pub temporal: TemporalConfig,
    #[serde(default)]
    pub false_positive: FalsePositiveConfig,
}

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FusionStrategy {
    #[default]
    Any,
    Merge,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ActionsConfig {
    #[serde(default)]
    pub sound: SoundConfig,
    #[serde(default)]
    pub notification: NotificationConfig,
    #[serde(default)]
    pub webhook: WebhookConfig,
    #[serde(default)]
    pub popup: PopupConfig,
}

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
pub struct NotificationConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
pub struct PopupConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
}

impl Default for PopupConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SelectionStrategy {
    #[default]
    Random,
    First,
    RoundRobin,
    Preferred,
}

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
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
fn default_device() -> String {
    "/dev/video0".to_string()
}
fn default_camera_width() -> u32 {
    640
}
fn default_camera_height() -> u32 {
    480
}
fn default_camera_fps() -> u32 {
    30
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

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.cameras.is_empty() {
            return Err(ConfigError::Validation(
                "At least one camera must be configured".to_string(),
            ));
        }

        for camera in &self.cameras {
            if camera.inference_fps > camera.fps {
                return Err(ConfigError::Validation(format!(
                    "Camera '{}': inference_fps ({}) cannot exceed fps ({})",
                    camera.id, camera.inference_fps, camera.fps
                )));
            }
            if camera.width == 0 || camera.height == 0 {
                return Err(ConfigError::Validation(format!(
                    "Camera '{}': width and height must be > 0",
                    camera.id
                )));
            }
        }

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

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_example_config() {
        let config = NailbiteConfig::load("config.yaml").expect("Failed to load config.yaml");
        assert_eq!(config.general.log_level, "debug");
        assert!(!config.general.show_preview);
        assert_eq!(config.general.cooldown_seconds, 30);
        assert_eq!(config.cameras.len(), 1);
        assert_eq!(config.cameras[0].id, "main");
        assert_eq!(config.cameras[0].device, "/dev/video0");
        assert_eq!(config.cameras[0].role, CameraRole::Full);
        assert_eq!(config.cameras[0].inference_fps, 8);
    }

    #[test]
    fn defaults_are_sensible() {
        let yaml = r#"
cameras:
  - id: test
"#;
        let config: NailbiteConfig = serde_yaml_ng::from_str(yaml).unwrap();
        config.validate().unwrap();

        assert_eq!(config.general.log_level, "info");
        assert_eq!(config.ort.intra_op_num_threads, 2);
        assert!(config.detection.behaviors.nail_biting.enabled);
        assert!(config.detection.behaviors.nail_picking.enabled);
        assert!(!config.detection.behaviors.hair_pulling.enabled);
        assert!(config.actions.sound.enabled);
        assert!(config.actions.notification.enabled);
        assert!(!config.actions.webhook.enabled);
        assert!(config.actions.popup.enabled);
    }

    #[test]
    fn validation_rejects_empty_cameras() {
        let yaml = "general:\n  log_level: info\n";
        let config: NailbiteConfig = serde_yaml_ng::from_str(yaml).unwrap();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("At least one camera"));
    }

    #[test]
    fn validation_rejects_inference_fps_exceeding_fps() {
        let yaml = r#"
cameras:
  - id: test
    fps: 10
    inference_fps: 20
"#;
        let config: NailbiteConfig = serde_yaml_ng::from_str(yaml).unwrap();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("inference_fps"));
    }

    #[test]
    fn validation_rejects_invalid_positive_ratio() {
        let yaml = r#"
cameras:
  - id: test
detection:
  temporal:
    positive_ratio: 1.5
"#;
        let config: NailbiteConfig = serde_yaml_ng::from_str(yaml).unwrap();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("positive_ratio"));
    }

    #[test]
    fn validation_rejects_invalid_compliance_ratio() {
        let yaml = r#"
cameras:
  - id: test
exercises:
  compliance_ratio: 0.0
"#;
        let config: NailbiteConfig = serde_yaml_ng::from_str(yaml).unwrap();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("compliance_ratio"));
    }

    #[test]
    fn validation_rejects_webhook_without_url() {
        let yaml = r#"
cameras:
  - id: test
actions:
  webhook:
    enabled: true
    url: ""
"#;
        let config: NailbiteConfig = serde_yaml_ng::from_str(yaml).unwrap();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("webhook.url"));
    }

    #[test]
    fn validation_rejects_preferred_without_exercise_id() {
        let yaml = r#"
cameras:
  - id: test
exercises:
  selection_strategy: preferred
"#;
        let config: NailbiteConfig = serde_yaml_ng::from_str(yaml).unwrap();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("preferred_exercise"));
    }

    #[test]
    fn validation_rejects_invalid_volume() {
        let yaml = r#"
cameras:
  - id: test
actions:
  sound:
    volume: 1.5
"#;
        let config: NailbiteConfig = serde_yaml_ng::from_str(yaml).unwrap();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("volume"));
    }

    #[test]
    fn validation_rejects_zero_dimensions() {
        let yaml = r#"
cameras:
  - id: test
    width: 0
"#;
        let config: NailbiteConfig = serde_yaml_ng::from_str(yaml).unwrap();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("width and height"));
    }

    #[test]
    fn partial_config_uses_defaults() {
        let yaml = r#"
cameras:
  - id: cam1
    device: /dev/video2
"#;
        let config: NailbiteConfig = serde_yaml_ng::from_str(yaml).unwrap();
        config.validate().unwrap();

        assert_eq!(config.cameras[0].device, "/dev/video2");
        assert_eq!(config.cameras[0].width, 640);
        assert_eq!(config.cameras[0].height, 480);
        assert_eq!(config.cameras[0].fps, 30);
        assert_eq!(config.cameras[0].inference_fps, 8);
    }

    #[test]
    fn camera_role_deserialization() {
        let yaml = r#"
cameras:
  - id: face_cam
    role: face_only
  - id: hand_cam
    role: hands_only
  - id: full_cam
    role: full
"#;
        let config: NailbiteConfig = serde_yaml_ng::from_str(yaml).unwrap();
        assert_eq!(config.cameras[0].role, CameraRole::FaceOnly);
        assert_eq!(config.cameras[1].role, CameraRole::HandsOnly);
        assert_eq!(config.cameras[2].role, CameraRole::Full);
    }

    #[test]
    fn fusion_strategy_deserialization() {
        let yaml = r#"
cameras:
  - id: test
fusion:
  strategy: merge
"#;
        let config: NailbiteConfig = serde_yaml_ng::from_str(yaml).unwrap();
        assert_eq!(config.fusion.strategy, FusionStrategy::Merge);
    }

    #[test]
    fn selection_strategy_deserialization() {
        let yaml = r#"
cameras:
  - id: test
exercises:
  selection_strategy: round_robin
"#;
        let config: NailbiteConfig = serde_yaml_ng::from_str(yaml).unwrap();
        assert_eq!(
            config.exercises.selection_strategy,
            SelectionStrategy::RoundRobin
        );
    }

    #[test]
    fn validation_accepts_positive_ratio_at_one() {
        let yaml = r#"
cameras:
  - id: test
detection:
  temporal:
    positive_ratio: 1.0
"#;
        let config: NailbiteConfig = serde_yaml_ng::from_str(yaml).unwrap();
        config.validate().unwrap();
    }

    #[test]
    fn validation_rejects_positive_ratio_at_zero() {
        let yaml = r#"
cameras:
  - id: test
detection:
  temporal:
    positive_ratio: 0.0
"#;
        let config: NailbiteConfig = serde_yaml_ng::from_str(yaml).unwrap();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("positive_ratio"));
    }

    #[test]
    fn validation_rejects_negative_positive_ratio() {
        let yaml = r#"
cameras:
  - id: test
detection:
  temporal:
    positive_ratio: -0.1
"#;
        let config: NailbiteConfig = serde_yaml_ng::from_str(yaml).unwrap();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("positive_ratio"));
    }

    #[test]
    fn validation_accepts_compliance_ratio_at_one() {
        let yaml = r#"
cameras:
  - id: test
exercises:
  compliance_ratio: 1.0
"#;
        let config: NailbiteConfig = serde_yaml_ng::from_str(yaml).unwrap();
        config.validate().unwrap();
    }

    #[test]
    fn validation_rejects_negative_compliance_ratio() {
        let yaml = r#"
cameras:
  - id: test
exercises:
  compliance_ratio: -0.5
"#;
        let config: NailbiteConfig = serde_yaml_ng::from_str(yaml).unwrap();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("compliance_ratio"));
    }

    #[test]
    fn validation_accepts_volume_at_zero() {
        let yaml = r#"
cameras:
  - id: test
actions:
  sound:
    volume: 0.0
"#;
        let config: NailbiteConfig = serde_yaml_ng::from_str(yaml).unwrap();
        config.validate().unwrap();
    }

    #[test]
    fn validation_accepts_volume_at_one() {
        let yaml = r#"
cameras:
  - id: test
actions:
  sound:
    volume: 1.0
"#;
        let config: NailbiteConfig = serde_yaml_ng::from_str(yaml).unwrap();
        config.validate().unwrap();
    }

    #[test]
    fn validation_rejects_negative_volume() {
        let yaml = r#"
cameras:
  - id: test
actions:
  sound:
    volume: -0.1
"#;
        let config: NailbiteConfig = serde_yaml_ng::from_str(yaml).unwrap();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("volume"));
    }

    #[test]
    fn validation_accepts_inference_fps_equal_to_fps() {
        let yaml = r#"
cameras:
  - id: test
    fps: 15
    inference_fps: 15
"#;
        let config: NailbiteConfig = serde_yaml_ng::from_str(yaml).unwrap();
        config.validate().unwrap();
    }

    #[test]
    fn valid_webhook_config() {
        let yaml = r#"
cameras:
  - id: test
actions:
  webhook:
    enabled: true
    url: "http://localhost:8080/bfrb-event"
    timeout_ms: 3000
"#;
        let config: NailbiteConfig = serde_yaml_ng::from_str(yaml).unwrap();
        config.validate().unwrap();
        assert!(config.actions.webhook.enabled);
        assert_eq!(config.actions.webhook.url, "http://localhost:8080/bfrb-event");
        assert_eq!(config.actions.webhook.timeout_ms, 3000);
    }
}
