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

export type SuppressionReason =
  | "no_face"
  | "no_hands"
  | "typing_posture"
  | "chin_rest"
  | "insufficient_hands";

/** Per-hand contribution to a detector's confidence on a single frame. */
export interface HandSignal {
  hand_index: number;
  side: "Left" | "Right" | null;
  normalized_distance: number;
  distance_threshold: number;
  contributing_fingertip: number | null;
  partner_fingertip: number | null;
  curl: number | null;
  bonus: number;
  confidence: number;
}

/** Explanation of one detector's decision on a single frame. */
export interface DetectionExplanation {
  bfrb_type: BfrbType;
  hands: HandSignal[];
  suppressions: SuppressionReason[];
  frame_confidence: number;
}

export interface DetectionEvent {
  bfrb_type: BfrbType;
  confidence: number;
  timestamp: string;
  explanation?: DetectionExplanation | null;
  event_id?: string | null;
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
  show_detection_count: boolean;
}

export type HandLandmarkQuality = "auto" | "lite" | "full";

export interface ModelsConfig {
  palm_detection: string;
  hand_landmark: string;
  hand_landmark_full: string;
  hand_landmark_quality: HandLandmarkQuality;
  face_detection: string;
  face_mesh: string;
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
  preview_fps: number;
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

export interface VisualConfig {
  enabled: boolean;
}

export interface ActionsConfig {
  sound: SoundConfig;
  webhook: WebhookConfig;
  visual: VisualConfig;
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

// Event history types
export type EventTrigger = "detection" | "missed_event" | "false_positive";

export type Verdict = "true_positive" | "false_positive" | "unsure";

export interface EventHistorySummary {
  id: string;
  timestamp: string;
  trigger: EventTrigger;
  bfrb_type: string | null;
  confidence: number | null;
  frame_count: number;
  user_rating: number | null;
  verdict: Verdict | null;
  trigger_frame: string | null;
  trigger_frame_annotated: string | null;
  /** Raw frame filenames for the whole event in chronological order. The
   *  list view uses this to render a filmstrip on wide viewports. */
  frame_files: string[];
}

/** Vector overlay (hand / face / pose landmarks) stored alongside the raw
 *  JPEG in event.json. Drawn on top of the image at display time so the
 *  stored pixels stay untouched. Mirrors the live-preview detection types. */
export interface FrameOverlay {
  hands: HandDetection[];
  face: FaceDetection | null;
  pose: PoseDetection | null;
}

export interface FrameInfo {
  offset: number;
  timestamp_ms: number;
  hand_count: number;
  hand_sides: string[];
  hand_confidences: number[];
  face_detected: boolean;
  pose_detected: boolean;
  raw_filename: string;
  /** Legacy: filename of a pre-rendered annotated JPEG. May be null on
   *  events captured after we switched to vector overlays. */
  annotated_filename: string | null;
  /** New: vector overlay drawn at display time. */
  overlay: FrameOverlay | null;
  explanations: DetectionExplanation[];
}

export interface EventHistoryDetail {
  id: string;
  timestamp: string;
  trigger: EventTrigger;
  bfrb_type: string | null;
  confidence: number | null;
  user_rating: number | null;
  verdict: Verdict | null;
  verdict_reason: string | null;
  explanation: DetectionExplanation | null;
  frames: FrameInfo[];
}

// Tray state
export type TrayState = "normal" | "alert" | "paused";
