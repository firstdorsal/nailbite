use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::detection::types::BfrbType;

/// Type of annotation for training data collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnnotationType {
    /// Detection confirmed by exercise completion.
    TruePositive,
    /// User dismissed alert via hotkey (false alarm).
    FalsePositive,
    /// User flagged a missed event via hotkey.
    FalseNegative,
}

/// A training annotation saved for future model improvement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingAnnotation {
    pub timestamp: DateTime<Utc>,
    pub annotation_type: AnnotationType,
    pub bfrb_type: Option<BfrbType>,
    /// Paths to saved frame images (if frame saving is enabled).
    pub frame_paths: Vec<PathBuf>,
    /// Detection confidence at time of annotation.
    pub detection_confidence: Option<f32>,
}
