// TypeScript interfaces matching Rust types

export interface Landmark {
  x: number;
  y: number;
  z: number;
}

export interface HandDetection {
  landmarks: Landmark[];
  handedness: "left" | "right" | "unknown";
  confidence: number;
}

export interface FaceDetection {
  landmarks: Landmark[];
  confidence: number;
}

export interface PoseLandmark {
  x: number;
  y: number;
  z: number;
  visibility: number;
  presence: number;
}

export interface PoseDetection {
  landmarks: PoseLandmark[];
  confidence: number;
}

export type BfrbType =
  | "nail_biting"
  | "nail_picking"
  | "hair_pulling"
  | "skin_picking"
  | "lip_biting";

export interface DetectionEvent {
  bfrb_type: BfrbType;
  confidence: number;
  timestamp: string;
}

export interface FrameResult {
  hands: HandDetection[];
  face: FaceDetection | null;
  pose: PoseDetection | null;
  detections: DetectionEvent[];
  alert_active: boolean;
  paused: boolean;
}

export interface Exercise {
  id: string;
  name: string;
  instructions: string;
  category: "timed_hold" | "repetitions";
  hold_duration_secs: number;
  target_reps: number;
}

export interface VerificationResult {
  pose_correct: boolean;
  feedback: string;
  progress: number;
}

export interface ExerciseSession {
  exercise: Exercise;
  started_at: string;
  completed: boolean;
  progress: number;
  current_rep: number;
}

// Config types
export interface GeneralConfig {
  log_level: string;
  show_preview: boolean;
  cooldown_seconds: number;
  stats_file: string;
}

export interface ModelsConfig {
  palm_detection: string;
  hand_landmark: string;
  face_detection: string;
  face_mesh: string;
  pose_detection: string;
  pose_landmark: string;
}

export type CameraRole = "primary" | "auxiliary";

export interface CameraSource {
  id: string;
  device: string;
  role: CameraRole;
  resolution_width: number;
  resolution_height: number;
}

export interface CameraConfig {
  sources: CameraSource[];
  inference_fps: number;
}

export type GpuPreference = "auto" | "disabled" | "required";
export type GpuBackend = "auto" | "cuda" | "tensor_rt" | "mi_graph_x";

export interface GpuConfig {
  preference: GpuPreference;
  backend: GpuBackend;
  device_id: number;
  fp16_enable: boolean;
  memory_limit_mb: number | null;
}

export interface OrtConfig {
  intra_op_num_threads: number;
  inter_op_num_threads: number;
  gpu: GpuConfig;
}

export interface BehaviorConfig {
  enabled: boolean;
  proximity_threshold?: number;
  min_sustained_ms?: number;
  confidence_threshold?: number;
}

export interface BehaviorsConfig {
  nail_biting: BehaviorConfig;
  nail_picking: BehaviorConfig;
  hair_pulling: BehaviorConfig;
  skin_picking: BehaviorConfig;
  lip_biting: BehaviorConfig;
}

export interface TemporalConfig {
  window_ms: number;
  positive_ratio: number;
}

export interface FalsePositiveConfig {
  typing_suppression: boolean;
  chin_rest_suppression: boolean;
  eating_suppression: boolean;
}

export interface DetectionConfig {
  behaviors: BehaviorsConfig;
  temporal: TemporalConfig;
  false_positive: FalsePositiveConfig;
}

export interface FusionConfig {
  strategy: "any" | "merge";
  merge_tolerance_ms: number;
}

export interface SoundConfig {
  enabled: boolean;
  file: string;
  volume: number;
  repeat: boolean;
}

export interface WebhookConfig {
  enabled: boolean;
  url: string;
  timeout_ms: number;
}

export interface ActionsConfig {
  sound: SoundConfig;
  webhook: WebhookConfig;
}

export interface ExercisesConfig {
  selection_strategy: "random" | "first" | "round_robin" | "preferred";
  preferred_exercise: string | null;
  hold_duration_override: number | null;
  reps_override: number | null;
  timeout_seconds: number;
  compliance_ratio: number;
}

export interface HotkeysConfig {
  dismiss_false_positive: string;
  mark_missed_event: string;
  pause_resume: string;
}

export interface TrainingConfig {
  save_frames: boolean;
  save_landmarks: boolean;
  annotations_file: string;
  frames_dir: string;
}

export interface NailbiteConfig {
  general: GeneralConfig;
  models: ModelsConfig;
  camera: CameraConfig;
  ort: OrtConfig;
  detection: DetectionConfig;
  fusion: FusionConfig;
  actions: ActionsConfig;
  exercises: ExercisesConfig;
  hotkeys: HotkeysConfig;
  training: TrainingConfig;
}

// Stats types
export interface SessionEntry {
  timestamp: string;
  event_type: "detection" | "exercise_completed" | "dismissed" | "missed";
  bfrb_type?: BfrbType;
  exercise_id?: string;
  duration_secs?: number;
}

export interface SessionStats {
  entries: SessionEntry[];
  total_detections: number;
  total_exercises_completed: number;
  total_dismissed: number;
  total_missed: number;
}

// Tray state
export type TrayState = "normal" | "alert" | "paused";
