//! Automatic model downloading.
//!
//! Downloads ONNX models on first run if they are not already present at
//! the configured paths. Uses blocking HTTP requests with progress logging.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use tracing::{info, warn};

use crate::config::ModelsConfig;
use crate::errors::InferenceError;

/// Metadata for a single model to download.
struct ModelSpec {
    /// Human-readable name for logging.
    name: &'static str,
    /// Download URL.
    url: &'static str,
    /// Expected SHA256 hex digest for integrity verification.
    sha256: &'static str,
}

const PALM_DETECTION: ModelSpec = ModelSpec {
    name: "palm_detection",
    url: "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx",
    sha256: "78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c",
};

const HAND_LANDMARK: ModelSpec = ModelSpec {
    name: "hand_landmark",
    url: "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx",
    sha256: "db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c",
};

const FACE_DETECTION: ModelSpec = ModelSpec {
    name: "face_detection",
    url: "https://github.com/IntelliProve/face-detection-onnx/raw/refs/heads/main/fdlite/data/face_detection_short_range.onnx",
    sha256: "bb171799a4497f9d07ef40c7d08acd9b2dd5e7d80ed00bfd0ef5ab2443aab643",
};

const FACE_MESH: ModelSpec = ModelSpec {
    name: "face_mesh",
    url: "https://github.com/IntelliProve/face-detection-onnx/raw/refs/heads/main/fdlite/data/face_landmark.onnx",
    sha256: "71625efd79fd3ce448ba26db9f7f58e4f37daabf36c81a45a661844e3fdb3118",
};

const POSE_DETECTION: ModelSpec = ModelSpec {
    name: "pose_detection",
    url: "https://huggingface.co/unity/inference-engine-blaze-pose/resolve/main/models/pose_detection.onnx",
    sha256: "72081da8481170bc6d8fafa716455ee210b61a8cefed84c67fcbbf889a4c38cf",
};

const POSE_LANDMARK: ModelSpec = ModelSpec {
    name: "pose_landmark",
    url: "https://huggingface.co/unity/inference-engine-blaze-pose/resolve/main/models/pose_landmarks_detector_full.onnx",
    sha256: "ae17ee8f076a5bbc28f65b939f46139c10f10c51ec4392a011e56d06d3f76c5d",
};

/// A model paired with its configured local path.
struct ModelEntry<'a> {
    path: &'a PathBuf,
    spec: &'a ModelSpec,
}

fn model_entries(config: &ModelsConfig) -> [ModelEntry<'_>; 6] {
    [
        ModelEntry { path: &config.palm_detection, spec: &PALM_DETECTION },
        ModelEntry { path: &config.hand_landmark, spec: &HAND_LANDMARK },
        ModelEntry { path: &config.face_detection, spec: &FACE_DETECTION },
        ModelEntry { path: &config.face_mesh, spec: &FACE_MESH },
        ModelEntry { path: &config.pose_detection, spec: &POSE_DETECTION },
        ModelEntry { path: &config.pose_landmark, spec: &POSE_LANDMARK },
    ]
}

/// Ensures all required models are present at the configured paths.
///
/// For each model, if the file already exists, it is left in place.
/// Otherwise it is downloaded from the hardcoded upstream URL.
/// After download, the SHA256 checksum is verified.
pub fn ensure_models(config: &ModelsConfig) -> Result<(), InferenceError> {
    for entry in &model_entries(config) {
        validate_model_path(entry.path, entry.spec.name)?;

        if entry.path.exists() {
            info!(model = entry.spec.name, path = %entry.path.display(), "Model already present");
            continue;
        }

        download_model(entry.path, entry.spec)?;
    }

    Ok(())
}

/// Validate that a model path doesn't escape expected directories via traversal.
/// Uses canonicalization for robust path validation (SECURITY-1).
fn validate_model_path(path: &Path, model_name: &str) -> Result<(), InferenceError> {
    // First check for obvious traversal patterns in the raw path
    let path_str = path.to_string_lossy();
    if path_str.contains("..") {
        return Err(InferenceError::DownloadFailed {
            model_name: model_name.to_string(),
            reason: format!(
                "Model path contains directory traversal (..): {}",
                path.display()
            ),
        });
    }

    // Get the current working directory as base
    let cwd = std::env::current_dir().map_err(|e| InferenceError::DownloadFailed {
        model_name: model_name.to_string(),
        reason: format!("Failed to get current directory: {e}"),
    })?;

    // Resolve the path relative to cwd (or keep as absolute if already absolute)
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    };

    // Canonicalize if parent exists (can't canonicalize a path that doesn't exist yet)
    // Check that the path doesn't escape the expected boundaries
    if let Some(parent) = resolved.parent() {
        if parent.exists() {
            let canonical_parent = parent.canonicalize().map_err(|e| InferenceError::DownloadFailed {
                model_name: model_name.to_string(),
                reason: format!("Failed to canonicalize parent path: {e}"),
            })?;

            // Ensure the canonical path is within cwd or /tmp or common model directories
            let canonical_str = canonical_parent.to_string_lossy();
            let cwd_str = cwd.to_string_lossy();

            let is_safe = canonical_str.starts_with(&*cwd_str)
                || canonical_str.starts_with("/tmp")
                || canonical_str.starts_with("/home")
                || canonical_str.contains("/.local/share/nailbite");

            if !is_safe {
                return Err(InferenceError::DownloadFailed {
                    model_name: model_name.to_string(),
                    reason: format!(
                        "Model path escapes expected directories: {} resolved to {}",
                        path.display(),
                        canonical_parent.display()
                    ),
                });
            }
        }
    }

    Ok(())
}

/// Downloads a single model file from its upstream URL.
fn download_model(path: &Path, spec: &ModelSpec) -> Result<(), InferenceError> {
    info!(
        model = spec.name,
        url = spec.url,
        path = %path.display(),
        "Downloading model"
    );

    // Ensure parent directory exists.
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| InferenceError::DirectoryCreation {
            path: parent.display().to_string(),
            source: e,
        })?;
    }

    let response = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .map_err(|e| InferenceError::DownloadFailed {
            model_name: spec.name.to_string(),
            reason: e.to_string(),
        })?
        .get(spec.url)
        .send()
        .map_err(|e| InferenceError::DownloadFailed {
            model_name: spec.name.to_string(),
            reason: e.to_string(),
        })?;

    if !response.status().is_success() {
        return Err(InferenceError::DownloadFailed {
            model_name: spec.name.to_string(),
            reason: format!("HTTP {}", response.status()),
        });
    }

    let bytes = response.bytes().map_err(|e| InferenceError::DownloadFailed {
        model_name: spec.name.to_string(),
        reason: e.to_string(),
    })?;

    // Write to a temporary file first, then rename for atomicity.
    let tmp_path = path.with_extension("onnx.tmp");
    let mut file = fs::File::create(&tmp_path).map_err(|e| InferenceError::WriteFailed {
        path: tmp_path.display().to_string(),
        source: e,
    })?;

    file.write_all(&bytes)
        .map_err(|e| InferenceError::WriteFailed {
            path: tmp_path.display().to_string(),
            source: e,
        })?;

    file.flush().map_err(|e| InferenceError::WriteFailed {
        path: tmp_path.display().to_string(),
        source: e,
    })?;

    // Verify SHA256 checksum before committing the file.
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let actual_hash = format!("{:x}", hasher.finalize());
    if actual_hash != spec.sha256 {
        // Clean up the temp file on checksum mismatch.
        let _ = fs::remove_file(&tmp_path);
        return Err(InferenceError::DownloadFailed {
            model_name: spec.name.to_string(),
            reason: format!(
                "SHA256 mismatch: expected {}, got {actual_hash}",
                spec.sha256
            ),
        });
    }

    fs::rename(&tmp_path, path).map_err(|e| InferenceError::WriteFailed {
        path: path.display().to_string(),
        source: e,
    })?;

    let size_mb = bytes.len() as f64 / 1_048_576.0;
    info!(
        model = spec.name,
        size_mb = size_mb,
        sha256 = actual_hash,
        "Model downloaded and verified"
    );

    Ok(())
}

/// Checks whether all models are present without downloading.
///
/// Returns a list of model names that are missing.
pub fn check_models(config: &ModelsConfig) -> Vec<&'static str> {
    let mut missing = Vec::new();
    for entry in &model_entries(config) {
        if !entry.path.exists() {
            warn!(model = entry.spec.name, path = %entry.path.display(), "Model file missing");
            missing.push(entry.spec.name);
        }
    }
    missing
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn check_models_reports_missing() {
        let config = ModelsConfig {
            palm_detection: PathBuf::from("/tmp/nailbite_test_nonexistent/palm.onnx"),
            hand_landmark: PathBuf::from("/tmp/nailbite_test_nonexistent/hand.onnx"),
            face_detection: PathBuf::from("/tmp/nailbite_test_nonexistent/face_det.onnx"),
            face_mesh: PathBuf::from("/tmp/nailbite_test_nonexistent/face_mesh.onnx"),
            pose_detection: PathBuf::from("/tmp/nailbite_test_nonexistent/pose_det.onnx"),
            pose_landmark: PathBuf::from("/tmp/nailbite_test_nonexistent/pose_lm.onnx"),
        };
        let missing = check_models(&config);
        assert_eq!(missing.len(), 6);
        assert!(missing.contains(&"palm_detection"));
        assert!(missing.contains(&"hand_landmark"));
        assert!(missing.contains(&"face_detection"));
        assert!(missing.contains(&"face_mesh"));
        assert!(missing.contains(&"pose_detection"));
        assert!(missing.contains(&"pose_landmark"));
    }

    #[test]
    fn check_models_reports_none_missing_when_all_exist() {
        // Use files that definitely exist on the system.
        let config = ModelsConfig {
            palm_detection: PathBuf::from("/dev/null"),
            hand_landmark: PathBuf::from("/dev/null"),
            face_detection: PathBuf::from("/dev/null"),
            face_mesh: PathBuf::from("/dev/null"),
            pose_detection: PathBuf::from("/dev/null"),
            pose_landmark: PathBuf::from("/dev/null"),
        };
        let missing = check_models(&config);
        assert!(missing.is_empty());
    }

    #[test]
    fn all_specs_use_https() {
        let specs = [&PALM_DETECTION, &HAND_LANDMARK, &FACE_DETECTION, &FACE_MESH, &POSE_DETECTION, &POSE_LANDMARK];
        for spec in specs {
            assert!(
                spec.url.starts_with("https://"),
                "Model {} URL must use HTTPS",
                spec.name
            );
            assert!(
                spec.url.ends_with(".onnx"),
                "Model {} URL must end with .onnx",
                spec.name
            );
        }
    }

    #[test]
    fn rejects_path_traversal() {
        let path = PathBuf::from("../../../etc/malicious.onnx");
        let result = super::validate_model_path(&path, "test_model");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("directory traversal"));
    }

    #[test]
    fn accepts_normal_model_path() {
        let path = PathBuf::from("models/palm_detection.onnx");
        assert!(super::validate_model_path(&path, "test_model").is_ok());
    }

    #[test]
    fn all_specs_have_unique_names() {
        let specs = [&PALM_DETECTION, &HAND_LANDMARK, &FACE_DETECTION, &FACE_MESH, &POSE_DETECTION, &POSE_LANDMARK];
        let names: Vec<&str> = specs.iter().map(|s| s.name).collect();
        let mut deduped = names.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(names.len(), deduped.len(), "Model names must be unique");
    }
}
