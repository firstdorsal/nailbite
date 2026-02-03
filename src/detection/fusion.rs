//! Multi-camera detection fusion.
//!
//! Combines detection results from multiple cameras using
//! configurable strategies (any-camera OR logic, or merged face+hands).

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::config::FusionStrategy;
use crate::detection::types::FrameAnalysis;

/// Fuses frame analyses from multiple cameras into a single analysis
/// for the detection pipeline.
pub struct DetectionFusion {
    strategy: FusionStrategy,
    merge_tolerance: Duration,
}

impl DetectionFusion {
    pub fn new(strategy: FusionStrategy, merge_tolerance_ms: u64) -> Self {
        Self {
            strategy,
            merge_tolerance: Duration::from_millis(merge_tolerance_ms),
        }
    }

    /// Fuse frame analyses from multiple cameras.
    ///
    /// For `Any` strategy: returns each analysis individually (OR logic).
    /// For `Merge` strategy: combines face data from one camera with hand data
    /// from another, if timestamps are close enough.
    pub fn fuse(&self, analyses: &[FrameAnalysis]) -> Vec<FrameAnalysis> {
        match self.strategy {
            FusionStrategy::Any => analyses.to_vec(),
            FusionStrategy::Merge => self.merge_analyses(analyses),
        }
    }

    /// Merge analyses by combining the best face detection with the best hand
    /// detections from different cameras, as long as timestamps are within tolerance.
    fn merge_analyses(&self, analyses: &[FrameAnalysis]) -> Vec<FrameAnalysis> {
        if analyses.len() <= 1 {
            return analyses.to_vec();
        }

        // Find the analysis with the best face detection (highest confidence).
        let best_face = analyses.iter().filter(|a| a.face.is_some()).max_by(|a, b| {
            let conf_a = a.face.as_ref().map_or(0.0, |f| f.confidence);
            let conf_b = b.face.as_ref().map_or(0.0, |f| f.confidence);
            conf_a.partial_cmp(&conf_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find the analysis with the most hand detections.
        let best_hands = analyses.iter().max_by_key(|a| a.hands.len());

        match (best_face, best_hands) {
            (Some(face_analysis), Some(hand_analysis)) => {
                // Check timestamp tolerance.
                let time_diff = timestamp_diff(face_analysis.timestamp, hand_analysis.timestamp);
                if time_diff <= self.merge_tolerance {
                    // Merge face from one and hands from the other.
                    vec![FrameAnalysis {
                        timestamp: face_analysis.timestamp.max(hand_analysis.timestamp),
                        camera_id: Arc::from(format!(
                            "{}+{}",
                            face_analysis.camera_id, hand_analysis.camera_id
                        )),
                        hands: hand_analysis.hands.clone(),
                        face: face_analysis.face.clone(),
                        raw_frame: None,
                    }]
                } else {
                    // Timestamps too far apart, return individually.
                    analyses.to_vec()
                }
            }
            _ => analyses.to_vec(),
        }
    }
}

/// Compute absolute duration between two instants.
fn timestamp_diff(a: Instant, b: Instant) -> Duration {
    if a >= b {
        a.duration_since(b)
    } else {
        b.duration_since(a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detection::types::{FaceDetection, HandDetection, Landmark};

    fn make_face(confidence: f32) -> FaceDetection {
        FaceDetection {
            landmarks: vec![
                Landmark {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                };
                468
            ],
            confidence,
        }
    }

    fn make_hand() -> HandDetection {
        HandDetection {
            side: None,
            landmarks: [Landmark {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            }; 21],
            confidence: 1.0,
        }
    }

    #[test]
    fn any_strategy_returns_all_analyses() {
        let fusion = DetectionFusion::new(FusionStrategy::Any, 100);
        let now = Instant::now();

        let analyses = vec![
            FrameAnalysis {
                timestamp: now,
                camera_id: Arc::from("cam0"),
                hands: vec![make_hand()],
                face: None,
                raw_frame: None,
            },
            FrameAnalysis {
                timestamp: now,
                camera_id: Arc::from("cam1"),
                hands: vec![],
                face: Some(make_face(0.9)),
                raw_frame: None,
            },
        ];

        let result = fusion.fuse(&analyses);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn merge_strategy_combines_face_and_hands() {
        let fusion = DetectionFusion::new(FusionStrategy::Merge, 100);
        let now = Instant::now();

        let analyses = vec![
            FrameAnalysis {
                timestamp: now,
                camera_id: Arc::from("face_cam"),
                hands: vec![],
                face: Some(make_face(0.9)),
                raw_frame: None,
            },
            FrameAnalysis {
                timestamp: now,
                camera_id: Arc::from("hand_cam"),
                hands: vec![make_hand(), make_hand()],
                face: None,
                raw_frame: None,
            },
        ];

        let result = fusion.fuse(&analyses);
        assert_eq!(result.len(), 1);
        assert!(result[0].face.is_some());
        assert_eq!(result[0].hands.len(), 2);
        assert!(result[0].camera_id.contains("face_cam"));
        assert!(result[0].camera_id.contains("hand_cam"));
    }

    #[test]
    fn merge_falls_back_when_timestamps_diverge() {
        let fusion = DetectionFusion::new(FusionStrategy::Merge, 50);
        let now = Instant::now();

        let analyses = vec![
            FrameAnalysis {
                timestamp: now,
                camera_id: Arc::from("cam0"),
                hands: vec![],
                face: Some(make_face(0.9)),
                raw_frame: None,
            },
            FrameAnalysis {
                timestamp: now + Duration::from_millis(200),
                camera_id: Arc::from("cam1"),
                hands: vec![make_hand()],
                face: None,
                raw_frame: None,
            },
        ];

        let result = fusion.fuse(&analyses);
        // Should fall back to returning both since tolerance exceeded.
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn single_analysis_passes_through() {
        let fusion = DetectionFusion::new(FusionStrategy::Merge, 100);
        let now = Instant::now();

        let analyses = vec![FrameAnalysis {
            timestamp: now,
            camera_id: Arc::from("cam0"),
            hands: vec![make_hand()],
            face: Some(make_face(0.8)),
            raw_frame: None,
        }];

        let result = fusion.fuse(&analyses);
        assert_eq!(result.len(), 1);
    }
}
