use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::frame::Frame;

/// 3D landmark point, normalized to [0.0, 1.0] relative to image dimensions.
#[derive(Debug, Clone, Copy)]
pub struct Landmark {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Which hand was detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HandSide {
    Left,
    Right,
}

/// Complete hand detection result from the hand landmark model.
#[derive(Debug, Clone)]
pub struct HandDetection {
    pub side: Option<HandSide>,
    pub landmarks: [Landmark; 21],
    pub confidence: f32,
}

/// Complete face detection result from the face mesh model.
#[derive(Debug, Clone)]
pub struct FaceDetection {
    /// 468 landmarks from face mesh
    pub landmarks: Vec<Landmark>,
    pub confidence: f32,
}

/// A single pose landmark with visibility and presence scores.
#[derive(Debug, Clone, Copy)]
pub struct PoseLandmark {
    pub landmark: Landmark,
    /// Probability that this landmark is visible (not occluded).
    pub visibility: f32,
    /// Probability that this landmark exists in the frame.
    pub presence: f32,
}

/// Complete body pose detection result from the pose landmark model.
#[derive(Debug, Clone)]
pub struct PoseDetection {
    /// 33 body landmarks (BlazePose format).
    pub landmarks: Vec<PoseLandmark>,
    pub confidence: f32,
}

/// Types of body-focused repetitive behaviors that can be detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BfrbType {
    NailBiting,
    NailPicking,
    HairPulling,
    SkinPicking,
    LipBiting,
}

impl BfrbType {
    /// Return the snake_case string representation for IPC.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NailBiting => "nail_biting",
            Self::NailPicking => "nail_picking",
            Self::HairPulling => "hair_pulling",
            Self::SkinPicking => "skin_picking",
            Self::LipBiting => "lip_biting",
        }
    }
}

impl std::fmt::Display for BfrbType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NailBiting => write!(f, "Nail Biting"),
            Self::NailPicking => write!(f, "Nail Picking"),
            Self::HairPulling => write!(f, "Hair Pulling"),
            Self::SkinPicking => write!(f, "Skin Picking"),
            Self::LipBiting => write!(f, "Lip Biting"),
        }
    }
}

/// A single frame's analysis results from one camera.
#[derive(Debug, Clone)]
pub struct FrameAnalysis {
    pub timestamp: Instant,
    /// Shared reference to avoid per-frame allocation of the camera ID string.
    pub camera_id: Arc<str>,
    pub hands: Vec<HandDetection>,
    pub face: Option<FaceDetection>,
    /// Raw RGB frame for preview rendering. `None` in tests or when preview is disabled.
    pub raw_frame: Option<Frame>,
}

/// A confirmed BFRB event after temporal analysis.
#[derive(Debug, Clone)]
pub struct DetectionEvent {
    pub bfrb_type: BfrbType,
    pub confidence: f32,
    pub started_at: Instant,
    pub duration: Duration,
    pub camera_id: String,
    /// Snapshot of the detector signals at the highest-confidence frame in
    /// the accumulation window. Used to explain *why* the alert fired.
    /// `None` when not produced (e.g. legacy callers / tests).
    pub explanation: Option<DetectionExplanation>,
}

/// Why a detector ignored or short-circuited a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuppressionReason {
    /// Detector requires a face but none was visible.
    NoFace,
    /// Detector requires hands but none were visible.
    NoHands,
    /// Hands were in a typing posture; behavior unlikely.
    TypingPosture,
    /// Hand was resting on the chin (extended fingers near face but not at mouth).
    ChinRest,
    /// Inter-hand check needs at least two hands.
    InsufficientHands,
    /// Both hands fully open / arms outstretched — looks like an arm
    /// stretch, not picking.
    StretchingPosture,
    /// Hand wrapped around an object (cup, bottle) with all fingertips
    /// clustered — looks like drinking, not biting.
    GrippingObject,
}

/// Per-hand contribution to a detector's confidence on a single frame.
///
/// Designed to be common across nail-biting and nail-picking so the frontend
/// can render one explanation widget. Detector-specific values use `None`
/// when not applicable (e.g. `curl` is biting-only).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandSignal {
    /// Index into the detector's input hand list.
    pub hand_index: usize,
    /// Identified hand side, when known.
    pub side: Option<HandSide>,
    /// Distance used by the detector, normalized to a unit-free scale
    /// (face width for biting, hand scale for picking).
    pub normalized_distance: f32,
    /// Threshold this distance was checked against. The hand contributed only
    /// when `normalized_distance < distance_threshold`.
    pub distance_threshold: f32,
    /// Fingertip landmark index that produced `normalized_distance`
    /// (one of `FINGERTIP_INDICES`). `None` when not meaningful.
    pub contributing_fingertip: Option<usize>,
    /// For nail-picking same-hand and inter-hand: the partner fingertip index.
    pub partner_fingertip: Option<usize>,
    /// Finger curl ratio [0, 1] — biting only.
    pub curl: Option<f32>,
    /// Bonus added on top of proximity (e.g. pinch bonus for picking).
    pub bonus: f32,
    /// This hand's contribution to the detector's frame confidence.
    pub confidence: f32,
}

/// Full explanation of a detector's decision on a single frame.
///
/// Produced by `BehaviorDetector::analyze_frame_explained`. Persisted into
/// `event.json`, sent over IPC for live signal display, and used by the
/// label-driven threshold analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionExplanation {
    pub bfrb_type: BfrbType,
    /// Per-hand contributions considered by the detector this frame.
    /// May be empty when a suppression fired or no hands were visible.
    pub hands: Vec<HandSignal>,
    /// Suppressions that fired this frame.
    pub suppressions: Vec<SuppressionReason>,
    /// Maximum per-hand confidence (matches the value returned by
    /// `analyze_frame`).
    pub frame_confidence: f32,
}

impl DetectionExplanation {
    #[must_use]
    pub fn empty(bfrb_type: BfrbType) -> Self {
        Self {
            bfrb_type,
            hands: Vec::new(),
            suppressions: Vec::new(),
            frame_confidence: 0.0,
        }
    }
}

/// Fingertip landmark indices in the MediaPipe hand model (21 landmarks).
pub const FINGERTIP_INDICES: [usize; 5] = [4, 8, 12, 16, 20];

/// Finger DIP (distal interphalangeal) joint indices.
pub const FINGER_DIP_INDICES: [usize; 5] = [3, 7, 11, 15, 19];

/// Finger PIP (proximal interphalangeal) joint indices.
pub const FINGER_PIP_INDICES: [usize; 4] = [6, 10, 14, 18];

/// Finger MCP (metacarpophalangeal) joint indices.
pub const FINGER_MCP_INDICES: [usize; 5] = [1, 5, 9, 13, 17];

/// Wrist landmark index.
pub const WRIST_INDEX: usize = 0;

/// Outer lip contour landmark indices in the face mesh model (468 landmarks).
pub const OUTER_LIP_INDICES: [usize; 20] = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
];

/// Inner lip contour landmark indices in the face mesh model (468 landmarks).
pub const INNER_LIP_INDICES: [usize; 20] = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
];

/// Upper inner lip center index.
pub const UPPER_LIP_CENTER: usize = 13;

/// Lower inner lip center index.
pub const LOWER_LIP_CENTER: usize = 14;

/// Left face boundary landmark (used for face width normalization).
pub const FACE_LEFT_INDEX: usize = 234;

/// Right face boundary landmark (used for face width normalization).
pub const FACE_RIGHT_INDEX: usize = 454;

/// Left ear landmark index (for decoupling exercise verification).
pub const EAR_LEFT_INDEX: usize = 234;

/// Right ear landmark index (for decoupling exercise verification).
pub const EAR_RIGHT_INDEX: usize = 454;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bfrb_type_display() {
        assert_eq!(BfrbType::NailBiting.to_string(), "Nail Biting");
        assert_eq!(BfrbType::NailPicking.to_string(), "Nail Picking");
        assert_eq!(BfrbType::HairPulling.to_string(), "Hair Pulling");
        assert_eq!(BfrbType::SkinPicking.to_string(), "Skin Picking");
        assert_eq!(BfrbType::LipBiting.to_string(), "Lip Biting");
    }

    #[test]
    fn fingertip_indices_are_correct() {
        assert_eq!(FINGERTIP_INDICES, [4, 8, 12, 16, 20]);
    }

    #[test]
    fn lip_indices_have_correct_length() {
        assert_eq!(OUTER_LIP_INDICES.len(), 20);
        assert_eq!(INNER_LIP_INDICES.len(), 20);
    }
}
