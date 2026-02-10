//! Training data collector.
//!
//! Saves annotated landmark data and optional frame images
//! for future model training. Writes JSONL format.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::Utc;
use tracing::{debug, warn};

use crate::config::TrainingConfig;
use crate::detection::types::BfrbType;
use crate::errors::TrainingError;
use crate::paths::expand_tilde;
use crate::training::annotation::{AnnotationType, TrainingAnnotation};

/// Collects and persists training annotations.
pub struct TrainingCollector {
    annotations_file: PathBuf,
    frames_dir: PathBuf,
    save_frames: bool,
    save_landmarks: bool,
}

impl TrainingCollector {
    /// Create a new training collector from configuration.
    pub fn new(config: &TrainingConfig) -> Result<Self, TrainingError> {
        let annotations_file = expand_tilde(&config.annotations_file);
        let frames_dir = expand_tilde(&config.frames_dir);

        // Ensure parent directories exist.
        if let Some(parent) = annotations_file.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                TrainingError::WriteFailed(format!(
                    "Failed to create annotation directory {}: {e}",
                    parent.display()
                ))
            })?;
        }

        if config.save_frames {
            fs::create_dir_all(&frames_dir).map_err(|e| {
                TrainingError::FrameSaveFailed(format!(
                    "Failed to create frames directory {}: {e}",
                    frames_dir.display()
                ))
            })?;
        }

        Ok(Self {
            annotations_file,
            frames_dir,
            save_frames: config.save_frames,
            save_landmarks: config.save_landmarks,
        })
    }

    /// Record an annotation (true positive, false positive, or false negative).
    pub fn record_annotation(
        &self,
        annotation_type: AnnotationType,
        bfrb_type: Option<BfrbType>,
        detection_confidence: Option<f32>,
    ) -> Result<(), TrainingError> {
        if !self.save_landmarks {
            return Ok(());
        }

        let annotation = TrainingAnnotation {
            timestamp: Utc::now(),
            annotation_type,
            bfrb_type,
            frame_paths: Vec::new(),
            detection_confidence,
        };

        let json = serde_json::to_string(&annotation).map_err(|e| {
            TrainingError::WriteFailed(format!("Failed to serialize annotation: {e}"))
        })?;

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.annotations_file)
            .map_err(|e| {
                TrainingError::WriteFailed(format!(
                    "Failed to open annotations file {}: {e}",
                    self.annotations_file.display()
                ))
            })?;

        writeln!(file, "{json}").map_err(|e| {
            TrainingError::WriteFailed(format!("Failed to write annotation: {e}"))
        })?;

        debug!(
            annotation_type = ?annotation_type,
            bfrb_type = ?bfrb_type,
            file = %self.annotations_file.display(),
            "Training annotation saved"
        );

        Ok(())
    }

    /// Save a raw frame image (if frame saving is enabled).
    ///
    /// Returns the path where the frame was saved, if any.
    pub fn save_frame(&self, rgb_data: &[u8], width: u32, height: u32) -> Option<PathBuf> {
        if !self.save_frames {
            return None;
        }

        let filename = format!("frame_{}.jpg", Utc::now().format("%Y%m%d_%H%M%S_%3f"));
        let path = self.frames_dir.join(&filename);

        match image::save_buffer(&path, rgb_data, width, height, image::ColorType::Rgb8) {
            Ok(()) => {
                debug!(path = %path.display(), "Training frame saved");
                Some(path)
            }
            Err(e) => {
                warn!(error = %e, "Failed to save training frame");
                None
            }
        }
    }

    /// Path to the annotations file.
    pub fn annotations_file(&self) -> &Path {
        &self.annotations_file
    }

    /// Whether frame saving is enabled.
    pub fn saves_frames(&self) -> bool {
        self.save_frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collector_writes_annotation() {
        let dir = tempfile::tempdir().unwrap();
        let annotations_path = dir.path().join("annotations.jsonl");

        let config = TrainingConfig {
            save_frames: false,
            save_landmarks: true,
            annotations_file: annotations_path.clone(),
            frames_dir: dir.path().join("frames"),
        };

        let collector = TrainingCollector::new(&config).unwrap();
        collector
            .record_annotation(
                AnnotationType::FalsePositive,
                Some(BfrbType::NailBiting),
                Some(0.75),
            )
            .unwrap();

        let content = fs::read_to_string(&annotations_path).unwrap();
        assert!(content.contains("FalsePositive"));
        assert!(content.contains("nail_biting"));
    }

    #[test]
    fn collector_skips_when_landmarks_disabled() {
        let dir = tempfile::tempdir().unwrap();
        let annotations_path = dir.path().join("annotations.jsonl");

        let config = TrainingConfig {
            save_frames: false,
            save_landmarks: false,
            annotations_file: annotations_path.clone(),
            frames_dir: dir.path().join("frames"),
        };

        let collector = TrainingCollector::new(&config).unwrap();
        collector
            .record_annotation(AnnotationType::TruePositive, None, None)
            .unwrap();

        // File should not be created.
        assert!(!annotations_path.exists());
    }
}
