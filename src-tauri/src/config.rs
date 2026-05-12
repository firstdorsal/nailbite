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
    #[serde(default)]
    pub history: HistoryConfig,
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
    /// When true, the in-app status circle and tray tooltip show the
    /// number of detections recorded so far today.
    #[serde(default = "default_true")]
    pub show_detection_count: bool,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            log_level: default_log_level(),
            show_preview: false,
            cooldown_seconds: default_cooldown_seconds(),
            stats_file: default_stats_file(),
            show_detection_count: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelsConfig {
    #[serde(default = "default_palm_detection_path")]
    pub palm_detection: PathBuf,
    #[serde(default = "default_hand_landmark_path")]
    pub hand_landmark: PathBuf,
    /// Higher-quality hand landmark model (RTMPose-m, SimCC).
    /// Optional: when present and loadable it replaces the MediaPipe lite
    /// model above. Used at runtime as the primary hand landmarker, with
    /// `hand_landmark` (MediaPipe lite) as automatic fallback.
    #[serde(default = "default_hand_landmark_full_path")]
    pub hand_landmark_full: PathBuf,
    /// Quality preference for hand landmarking.
    /// `auto` = use full model if downloaded and loadable, else lite.
    /// `lite` = always use MediaPipe lite (fast CPU path).
    /// `full` = require RTMPose full model; error if not available.
    #[serde(default)]
    pub hand_landmark_quality: HandLandmarkQuality,
    #[serde(default = "default_face_detection_path")]
    pub face_detection: PathBuf,
    #[serde(default = "default_face_mesh_path")]
    pub face_mesh: PathBuf,
    #[serde(default = "default_pose_landmark_path")]
    pub pose_landmark: PathBuf,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            palm_detection: default_palm_detection_path(),
            hand_landmark: default_hand_landmark_path(),
            hand_landmark_full: default_hand_landmark_full_path(),
            hand_landmark_quality: HandLandmarkQuality::default(),
            face_detection: default_face_detection_path(),
            face_mesh: default_face_mesh_path(),
            pose_landmark: default_pose_landmark_path(),
        }
    }
}

/// Hand landmark quality preference.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HandLandmarkQuality {
    /// Use the high-quality RTMPose model when available, falling back
    /// to the MediaPipe lite model otherwise.
    #[default]
    Auto,
    /// Always use the MediaPipe lite model. Skips downloading the full model.
    Lite,
    /// Require the RTMPose full model. Fails startup if it isn't available.
    Full,
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
    /// Preview FPS — how often the camera frame is pushed to the UI. Decoupled
    /// from `inference_fps` so the live preview can stay smooth even when
    /// inference runs at a slower rate. Must be >= `inference_fps`.
    #[serde(default = "default_preview_fps")]
    pub preview_fps: u32,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            sources: default_cameras(),
            inference_fps: default_inference_fps(),
            preview_fps: default_preview_fps(),
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
    /// ONNX Runtime graph-optimization level.
    ///
    /// Defaults to `extended` (Level2). Level3 enables ORT's layout
    /// transformer, which inserts `ReorderOutput_*` nodes — those trip a
    /// `GetElementType is not implemented` crash on the opencv_zoo
    /// MediaPipe palm/face detector models with ORT 1.23 on CPU. Keep on
    /// `extended` unless you have benchmarked Level3 with your stack.
    #[serde(default)]
    pub graph_optimization: GraphOptimization,
}

impl Default for OrtConfig {
    fn default() -> Self {
        Self {
            intra_op_num_threads: default_intra_op_threads(),
            inter_op_num_threads: default_inter_op_threads(),
            gpu: GpuConfig::default(),
            graph_optimization: GraphOptimization::default(),
        }
    }
}

/// Mirror of `ort::session::builder::GraphOptimizationLevel`, with serde.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GraphOptimization {
    /// Disable all optimizations.
    Disabled,
    /// Basic constant folding / redundant-node elimination.
    Basic,
    /// Extended: above + node fusion, CSE. Safe on CPU + GPU.
    #[default]
    Extended,
    /// All: above + the layout transformer. Faster on GPU; can produce
    /// `ReorderOutput_*` nodes that crash on ORT 1.23 CPU EP.
    All,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct DetectionConfig {
    #[serde(default)]
    pub behaviors: BehaviorsConfig,
    #[serde(default)]
    pub temporal: TemporalConfig,
    #[serde(default)]
    pub false_positive: FalsePositiveConfig,
    #[serde(default)]
    pub tracking: TrackingTuning,
}

/// Hand tracking stability parameters.
/// These control grace period, confirmation delay, smoothing, and confidence hysteresis.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrackingTuning {
    /// Number of unmatched frames before a hand goes invisible.
    /// Higher = more stable but hands persist longer after disappearing.
    #[serde(default = "default_tracking_grace_frames")]
    pub grace_frames: u8,
    /// Number of consecutive detection frames required before a new hand becomes visible.
    /// Higher = fewer false positives but slower to react to new hands.
    #[serde(default = "default_tracking_confirmation_frames")]
    pub confirmation_frames: u8,
    /// EMA smoothing factor for landmark updates (0 = no smoothing, 1 = no update).
    #[serde(default = "default_tracking_smoothing_alpha")]
    pub smoothing_alpha: f32,
    /// Confidence threshold for accepting a new hand (higher = more selective).
    #[serde(default = "default_tracking_new_hand_confidence")]
    pub new_hand_confidence: f32,
    /// Confidence threshold for keeping an existing tracked hand (lower = more stable).
    #[serde(default = "default_tracking_existing_hand_confidence")]
    pub existing_hand_confidence: f32,
}

impl Default for TrackingTuning {
    fn default() -> Self {
        Self {
            grace_frames: default_tracking_grace_frames(),
            confirmation_frames: default_tracking_confirmation_frames(),
            smoothing_alpha: default_tracking_smoothing_alpha(),
            new_hand_confidence: default_tracking_new_hand_confidence(),
            existing_hand_confidence: default_tracking_existing_hand_confidence(),
        }
    }
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
    /// Visual feedback shown when an alert fires (red full-screen vignette).
    /// Independent from `sound`: user can pick beep, vignette, or both.
    #[serde(default)]
    pub visual: VisualConfig,
    /// Desktop notification with built-in "Correct" / "False positive"
    /// action buttons. Clicking either button stops the alert sound
    /// immediately and persists the verdict to the recorded event.
    #[serde(default)]
    pub notification: NotificationConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NotificationConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// How long the notification stays on screen before auto-dismissing
    /// (milliseconds). Most desktops cap this at ~30s anyway.
    #[serde(default = "default_notification_timeout_ms")]
    pub timeout_ms: u32,
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            timeout_ms: default_notification_timeout_ms(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VisualConfig {
    /// Whether to render the alert vignette overlay in the UI.
    #[serde(default = "default_true")]
    pub enabled: bool,
}

impl Default for VisualConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
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

/// Event history recording configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HistoryConfig {
    /// Whether event history recording is enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Directory for event history storage.
    #[serde(default = "default_history_dir")]
    pub dir: PathBuf,
    /// Number of frames to save before the event trigger.
    #[serde(default = "default_history_frames_before")]
    pub frames_before: usize,
    /// Number of frames to save after the event trigger.
    #[serde(default = "default_history_frames_after")]
    pub frames_after: usize,
    /// Whether to draw landmarks on annotated frames.
    #[serde(default = "default_true")]
    pub annotate_landmarks: bool,
    /// Maximum number of events to keep (oldest are pruned).
    #[serde(default = "default_history_max_events")]
    pub max_events: usize,
}

impl Default for HistoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            dir: default_history_dir(),
            frames_before: default_history_frames_before(),
            frames_after: default_history_frames_after(),
            annotate_landmarks: true,
            max_events: default_history_max_events(),
        }
    }
}

// --- Default value functions ---

fn default_history_dir() -> PathBuf {
    PathBuf::from("~/.local/share/nailbite/history")
}
fn default_history_frames_before() -> usize {
    5
}
fn default_history_frames_after() -> usize {
    5
}
fn default_history_max_events() -> usize {
    100
}

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
fn default_hand_landmark_full_path() -> PathBuf {
    PathBuf::from("./models/hand_landmark_full.onnx")
}
fn default_face_detection_path() -> PathBuf {
    PathBuf::from("./models/face_detection.onnx")
}
fn default_face_mesh_path() -> PathBuf {
    PathBuf::from("./models/face_mesh.onnx")
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
fn default_preview_fps() -> u32 {
    24
}
fn default_intra_op_threads() -> u32 {
    // 8 is a good middle ground: enough to saturate inference on modern
    // 8-32 core desktops without hogging the whole machine. Users with
    // smaller CPUs can lower this in `config.yaml`.
    8
}
fn default_inter_op_threads() -> u32 {
    2
}
fn default_proximity_threshold() -> f32 {
    0.35
}
fn default_min_sustained_ms() -> u64 {
    // 3000 ms — a brief hand-near-mouth/face touch is almost always
    // accidental (adjusting glasses, scratching a chin). Real BFRB
    // episodes routinely sit much longer than this. Bumped from 1500ms
    // after users reported false positives on incidental motions.
    3000
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
fn default_notification_timeout_ms() -> u32 {
    20_000
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

fn default_tracking_grace_frames() -> u8 {
    3
}
fn default_tracking_confirmation_frames() -> u8 {
    2
}
fn default_tracking_smoothing_alpha() -> f32 {
    0.3
}
fn default_tracking_new_hand_confidence() -> f32 {
    0.30
}
fn default_tracking_existing_hand_confidence() -> f32 {
    0.15
}

fn default_nail_biting_config() -> BehaviorConfig {
    BehaviorConfig {
        enabled: true,
        proximity_threshold: 0.35,
        min_sustained_ms: 3000,
        confidence_threshold: 0.3,
    }
}

fn default_nail_picking_config() -> BehaviorConfig {
    BehaviorConfig {
        enabled: true,
        proximity_threshold: 0.15,
        min_sustained_ms: 3000,
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

        let tracking = &self.detection.tracking;
        if tracking.smoothing_alpha < 0.0 || tracking.smoothing_alpha > 1.0 {
            return Err(ConfigError::Validation(
                "detection.tracking.smoothing_alpha must be in [0.0, 1.0]".to_string(),
            ));
        }
        if tracking.new_hand_confidence < 0.0 || tracking.new_hand_confidence > 1.0 {
            return Err(ConfigError::Validation(
                "detection.tracking.new_hand_confidence must be in [0.0, 1.0]".to_string(),
            ));
        }
        if tracking.existing_hand_confidence < 0.0 || tracking.existing_hand_confidence > 1.0 {
            return Err(ConfigError::Validation(
                "detection.tracking.existing_hand_confidence must be in [0.0, 1.0]".to_string(),
            ));
        }
        if tracking.existing_hand_confidence > tracking.new_hand_confidence {
            return Err(ConfigError::Validation(
                "detection.tracking.existing_hand_confidence must be <= new_hand_confidence (hysteresis)".to_string(),
            ));
        }

        if self.ort.gpu.device_id > 7 {
            return Err(ConfigError::Validation(
                "ort.gpu.device_id must be in range 0-7".to_string(),
            ));
        }

        Ok(())
    }
}

