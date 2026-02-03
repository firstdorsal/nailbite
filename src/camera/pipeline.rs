/// Per-camera inference pipeline.
///
/// Captures frames, runs ONNX models, and produces `FrameAnalysis` results.
/// Each camera runs its own pipeline on a dedicated thread.
use std::sync::Arc;
use std::time::{Duration, Instant};


use crossbeam_channel::Sender;
use tracing::{debug, error, warn};

use crate::camera::backend::CameraBackend;
use crate::config::{CameraConfig, CameraRole};
use crate::detection::types::{FaceDetection, FrameAnalysis};
use crate::inference::face_detection::FaceDetector;
use crate::inference::face_mesh::FaceMesher;
use crate::inference::hand_landmark::HandLandmarker;
use crate::inference::palm_detection::PalmDetector;
use crate::inference::session::ModelSessions;

pub struct CameraPipeline {
    camera_id: Arc<str>,
    role: CameraRole,
    palm_detector: PalmDetector,
    hand_landmarker: HandLandmarker,
    face_detector: FaceDetector,
    face_mesher: FaceMesher,
    inference_interval: Duration,
}

impl CameraPipeline {
    pub fn new(config: &CameraConfig, sessions: &ModelSessions) -> Self {
        let inference_interval = if config.inference_fps > 0 {
            Duration::from_secs_f64(1.0 / f64::from(config.inference_fps))
        } else {
            Duration::from_millis(125) // Default 8 FPS.
        };

        Self {
            camera_id: Arc::from(config.id.as_str()),
            role: config.role,
            palm_detector: PalmDetector::new(Arc::clone(&sessions.palm_detection)),
            hand_landmarker: HandLandmarker::new(Arc::clone(&sessions.hand_landmark)),
            face_detector: FaceDetector::new(Arc::clone(&sessions.face_detection)),
            face_mesher: FaceMesher::new(Arc::clone(&sessions.face_mesh)),
            inference_interval,
        }
    }

    /// Run the pipeline loop on a camera backend, sending results to the channel.
    ///
    /// This blocks the current thread. Call from a dedicated thread.
    /// Returns when `running` is set to false, or when the camera device
    /// appears to be lost (too many consecutive capture errors).
    pub fn run<B: CameraBackend>(
        &self,
        backend: &mut B,
        tx: &Sender<FrameAnalysis>,
        running: &std::sync::atomic::AtomicBool,
    ) {
        /// Consecutive capture failures before assuming the device is lost.
        const MAX_CONSECUTIVE_ERRORS: u32 = 30;
        /// EMA smoothing factor for face landmarks (higher = less smooth).
        const FACE_SMOOTH_ALPHA: f32 = 0.5;

        let mut last_inference = Instant::now() - self.inference_interval;
        let mut consecutive_errors: u32 = 0;
        let mut prev_face: Option<FaceDetection> = None;

        while running.load(std::sync::atomic::Ordering::Relaxed) {
            let frame = match backend.capture_frame() {
                Ok(f) => {
                    consecutive_errors = 0;
                    f
                }
                Err(e) => {
                    consecutive_errors = consecutive_errors.saturating_add(1);
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                        error!(
                            camera = %self.camera_id,
                            errors = consecutive_errors,
                            "Device appears lost, exiting pipeline for reconnect"
                        );
                        return;
                    }
                    error!(camera = %self.camera_id, error = %e, "Frame capture failed");
                    std::thread::sleep(Duration::from_millis(100));
                    continue;
                }
            };

            // Rate-limit inference.
            let now = Instant::now();
            if now.duration_since(last_inference) < self.inference_interval {
                continue;
            }
            last_inference = now;

            let mut analysis = match self.process_frame(&frame) {
                Ok(a) => a,
                Err(e) => {
                    warn!(camera = %self.camera_id, error = %e, "Inference failed");
                    continue;
                }
            };

            // Smooth face landmarks with exponential moving average.
            smooth_face_landmarks(&mut analysis.face, &mut prev_face, FACE_SMOOTH_ALPHA);

            // Attach raw frame for preview rendering (moved, not cloned).
            analysis.raw_frame = Some(frame);

            // Non-blocking send: drop if channel is full (bounded channel backpressure).
            if tx.try_send(analysis).is_err() {
                debug!(camera = %self.camera_id, "Detection channel full, dropping frame");
            }
        }
    }

    /// Process a single frame through the appropriate model pipeline.
    fn process_frame(
        &self,
        frame: &crate::camera::frame::Frame,
    ) -> Result<FrameAnalysis, crate::errors::InferenceError> {
        let timestamp = Instant::now();
        let mut hands = Vec::new();
        let mut face = None;

        let run_hands = self.role != CameraRole::FaceOnly;
        let run_face = self.role != CameraRole::HandsOnly;

        if run_face {
            let face_result = self.face_detector.detect(frame)?;

            // Run face mesh on the first (highest confidence) face.
            if let Some(roi) = face_result.face_rois.first() {
                match self.face_mesher.estimate(frame, roi) {
                    Ok(mesh) => face = mesh,
                    Err(e) => {
                        debug!(error = %e, "Face mesh estimation failed");
                    }
                }
            }
        }

        // Run hand pipeline. The hand landmark model filters false positives
        // (face-area palm detections get rejected at confidence 0.005-0.08).
        if run_hands {
            let palm_result = self.palm_detector.detect(frame)?;

            for roi in &palm_result.hand_rois {
                match self.hand_landmarker.estimate(frame, roi) {
                    Ok(Some(hand)) => hands.push(hand),
                    Ok(None) => {}
                    Err(e) => {
                        debug!(error = %e, "Hand landmark estimation failed for ROI");
                    }
                }
            }
        }

        Ok(FrameAnalysis {
            timestamp,
            camera_id: Arc::clone(&self.camera_id),
            hands,
            face,
            raw_frame: None, // Set by `run()` after `process_frame()` returns.
        })
    }
}

/// Apply exponential moving average smoothing to face landmarks.
///
/// Blends the current face detection landmarks with the previous frame's
/// landmarks using `alpha` as the weight for new values. Updates `prev`
/// with the smoothed result for the next frame.
fn smooth_face_landmarks(
    current: &mut Option<FaceDetection>,
    prev: &mut Option<FaceDetection>,
    alpha: f32,
) {
    let one_minus_alpha = 1.0 - alpha;

    match (current.as_mut(), prev.as_ref()) {
        (Some(cur), Some(prv)) if cur.landmarks.len() == prv.landmarks.len() => {
            for (c, p) in cur.landmarks.iter_mut().zip(&prv.landmarks) {
                c.x = alpha.mul_add(c.x, one_minus_alpha * p.x);
                c.y = alpha.mul_add(c.y, one_minus_alpha * p.y);
                c.z = alpha.mul_add(c.z, one_minus_alpha * p.z);
            }
            *prev = Some(cur.clone());
        }
        (Some(cur), _) => {
            // No previous frame or landmark count changed; adopt current as-is.
            *prev = Some(cur.clone());
        }
        (None, _) => {
            // No face detected; clear previous to avoid stale smoothing.
            *prev = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detection::types::Landmark;

    #[test]
    fn smooth_face_landmarks_blends_with_previous() {
        let lm = |x, y, z| Landmark { x, y, z };
        let mut current = Some(FaceDetection {
            landmarks: vec![lm(1.0, 1.0, 1.0)],
            confidence: 0.9,
        });
        let mut prev = Some(FaceDetection {
            landmarks: vec![lm(0.0, 0.0, 0.0)],
            confidence: 0.9,
        });

        smooth_face_landmarks(&mut current, &mut prev, 0.5);

        let cur = current.as_ref().unwrap();
        assert!((cur.landmarks[0].x - 0.5).abs() < 0.001);
        assert!((cur.landmarks[0].y - 0.5).abs() < 0.001);
        assert!((cur.landmarks[0].z - 0.5).abs() < 0.001);
    }

    #[test]
    fn smooth_face_landmarks_adopts_first_frame() {
        let lm = |x, y, z| Landmark { x, y, z };
        let mut current = Some(FaceDetection {
            landmarks: vec![lm(0.5, 0.5, 0.5)],
            confidence: 0.9,
        });
        let mut prev = None;

        smooth_face_landmarks(&mut current, &mut prev, 0.5);

        // Current should be unchanged (no blending).
        let cur = current.as_ref().unwrap();
        assert!((cur.landmarks[0].x - 0.5).abs() < 0.001);
        // prev should now hold the current value.
        assert!(prev.is_some());
    }

    #[test]
    fn smooth_face_landmarks_clears_on_no_face() {
        let lm = |x, y, z| Landmark { x, y, z };
        let mut current = None;
        let mut prev = Some(FaceDetection {
            landmarks: vec![lm(0.5, 0.5, 0.5)],
            confidence: 0.9,
        });

        smooth_face_landmarks(&mut current, &mut prev, 0.5);

        assert!(prev.is_none());
    }

    #[test]
    fn smooth_face_landmarks_alpha_weighting() {
        let lm = |x, y, z| Landmark { x, y, z };
        let mut current = Some(FaceDetection {
            landmarks: vec![lm(1.0, 0.0, 0.0)],
            confidence: 0.9,
        });
        let mut prev = Some(FaceDetection {
            landmarks: vec![lm(0.0, 1.0, 0.0)],
            confidence: 0.9,
        });

        // alpha=0.8: heavily weight new value.
        smooth_face_landmarks(&mut current, &mut prev, 0.8);

        let cur = current.as_ref().unwrap();
        // x: 0.8*1.0 + 0.2*0.0 = 0.8
        assert!((cur.landmarks[0].x - 0.8).abs() < 0.001);
        // y: 0.8*0.0 + 0.2*1.0 = 0.2
        assert!((cur.landmarks[0].y - 0.2).abs() < 0.001);
    }

}
