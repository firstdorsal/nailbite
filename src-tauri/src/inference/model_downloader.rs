//! Automatic model downloading.
//!
//! Downloads ONNX models on first run if they are not already present at
//! the configured paths. Uses blocking HTTP requests with progress logging.
//!
//! Supports both raw `.onnx` URLs and zipped distributions: when a
//! [`ModelSpec`] declares an `archive_member`, the downloaded `.zip` is
//! verified by hash and the named member is extracted to the configured
//! path.

use std::fs;
use std::io::{Cursor, Read as _, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use tracing::{info, warn};

use crate::config::{HandLandmarkQuality, ModelsConfig};
use crate::errors::InferenceError;

/// Metadata for a single model to download.
struct ModelSpec {
    /// Human-readable name for logging.
    name: &'static str,
    /// Download URL.
    url: &'static str,
    /// Expected SHA256 hex digest of the *final on-disk file* (after any
    /// zip extraction).
    sha256: &'static str,
    /// If `Some`, the URL points at a `.zip` and this is the path of the
    /// `.onnx` we want to extract from within it. If `None`, the URL is
    /// downloaded directly as the model file.
    archive_member: Option<&'static str>,
}

const PALM_DETECTION: ModelSpec = ModelSpec {
    name: "palm_detection",
    url: "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx",
    sha256: "78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c",
    archive_member: None,
};

const HAND_LANDMARK: ModelSpec = ModelSpec {
    name: "hand_landmark",
    url: "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx",
    sha256: "db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c",
    archive_member: None,
};

/// Higher-quality hand landmark model (RTMPose-m, trained on 5 hand
/// datasets, 21 keypoints, SimCC decoding). Substantially more accurate
/// than the MediaPipe lite weights at ~14× the size.
///
/// Distributed as a zip alongside metadata; we extract the `end2end.onnx`.
const HAND_LANDMARK_FULL: ModelSpec = ModelSpec {
    name: "hand_landmark_full",
    url: "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip",
    sha256: "39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29",
    archive_member: Some(
        "20230831/rtmpose_onnx/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320/end2end.onnx",
    ),
};

const FACE_DETECTION: ModelSpec = ModelSpec {
    name: "face_detection",
    url: "https://github.com/IntelliProve/face-detection-onnx/raw/refs/heads/main/fdlite/data/face_detection_short_range.onnx",
    sha256: "bb171799a4497f9d07ef40c7d08acd9b2dd5e7d80ed00bfd0ef5ab2443aab643",
    archive_member: None,
};

const FACE_MESH: ModelSpec = ModelSpec {
    name: "face_mesh",
    url: "https://github.com/IntelliProve/face-detection-onnx/raw/refs/heads/main/fdlite/data/face_landmark.onnx",
    sha256: "71625efd79fd3ce448ba26db9f7f58e4f37daabf36c81a45a661844e3fdb3118",
    archive_member: None,
};

const POSE_LANDMARK: ModelSpec = ModelSpec {
    name: "pose_landmark",
    url: "https://huggingface.co/unity/inference-engine-blaze-pose/resolve/main/models/pose_landmarks_detector_full.onnx",
    sha256: "ae17ee8f076a5bbc28f65b939f46139c10f10c51ec4392a011e56d06d3f76c5d",
    archive_member: None,
};

/// A model paired with its configured local path.
struct ModelEntry<'a> {
    path: &'a PathBuf,
    spec: &'a ModelSpec,
}

/// Required vs optional entries.
///
/// `hand_landmark` (the lite MediaPipe model) is always required — it is
/// the fallback when the higher-quality model can't be loaded.
/// `hand_landmark_full` is best-effort: download failures are logged but
/// don't stop startup, so users on offline networks still get the lite path.
fn required_entries(config: &ModelsConfig) -> [ModelEntry<'_>; 5] {
    [
        ModelEntry { path: &config.palm_detection, spec: &PALM_DETECTION },
        ModelEntry { path: &config.hand_landmark, spec: &HAND_LANDMARK },
        ModelEntry { path: &config.face_detection, spec: &FACE_DETECTION },
        ModelEntry { path: &config.face_mesh, spec: &FACE_MESH },
        ModelEntry { path: &config.pose_landmark, spec: &POSE_LANDMARK },
    ]
}

fn optional_entries(config: &ModelsConfig) -> [ModelEntry<'_>; 1] {
    [ModelEntry {
        path: &config.hand_landmark_full,
        spec: &HAND_LANDMARK_FULL,
    }]
}

/// Ensures all required models are present at the configured paths.
///
/// For each required model, if the file already exists, it is left in place.
/// Otherwise it is downloaded from the hardcoded upstream URL and verified.
///
/// Optional models (e.g. the higher-quality RTMPose hand landmarker) are
/// downloaded on a best-effort basis. The decision is driven by
/// [`ModelsConfig::hand_landmark_quality`]:
///   * `Lite` — skip the optional model entirely.
///   * `Auto` — try, but log-and-continue on failure (lite is the fallback).
///   * `Full` — fail startup if the optional model can't be made available.
pub fn ensure_models(config: &ModelsConfig) -> Result<(), InferenceError> {
    for entry in &required_entries(config) {
        validate_model_path(entry.path, entry.spec.name)?;

        if entry.path.exists() {
            info!(model = entry.spec.name, path = %entry.path.display(), "Model already present");
            continue;
        }

        download_model(entry.path, entry.spec)?;
    }

    let strict_optionals = matches!(config.hand_landmark_quality, HandLandmarkQuality::Full);
    let skip_optionals = matches!(config.hand_landmark_quality, HandLandmarkQuality::Lite);

    if !skip_optionals {
        for entry in &optional_entries(config) {
            if let Err(e) = validate_model_path(entry.path, entry.spec.name) {
                if strict_optionals {
                    return Err(e);
                }
                warn!(model = entry.spec.name, error = %e, "Optional model path invalid; skipping");
                continue;
            }

            if entry.path.exists() {
                info!(model = entry.spec.name, path = %entry.path.display(), "Optional model already present");
                continue;
            }

            match download_model(entry.path, entry.spec) {
                Ok(()) => {}
                Err(e) if strict_optionals => return Err(e),
                Err(e) => {
                    warn!(
                        model = entry.spec.name,
                        error = %e,
                        "Optional model download failed; will fall back to lite model"
                    );
                }
            }
        }
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

    // Resolve final model bytes: either the downloaded payload itself
    // (raw .onnx) or the named member of a .zip archive.
    let model_bytes: Vec<u8> = if let Some(member) = spec.archive_member {
        extract_zip_member(&bytes, member, spec.name)?
    } else {
        bytes.to_vec()
    };

    // Verify SHA256 of the *final on-disk* bytes (post-extraction).
    let mut hasher = Sha256::new();
    hasher.update(&model_bytes);
    let actual_hash = format!("{:x}", hasher.finalize());
    if actual_hash != spec.sha256 {
        return Err(InferenceError::DownloadFailed {
            model_name: spec.name.to_string(),
            reason: format!(
                "SHA256 mismatch: expected {}, got {actual_hash}",
                spec.sha256
            ),
        });
    }

    // Write to a temporary file first, then rename for atomicity.
    let tmp_path = path.with_extension("onnx.tmp");
    let mut file = fs::File::create(&tmp_path).map_err(|e| InferenceError::WriteFailed {
        path: tmp_path.display().to_string(),
        source: e,
    })?;

    file.write_all(&model_bytes)
        .map_err(|e| InferenceError::WriteFailed {
            path: tmp_path.display().to_string(),
            source: e,
        })?;

    file.flush().map_err(|e| InferenceError::WriteFailed {
        path: tmp_path.display().to_string(),
        source: e,
    })?;

    fs::rename(&tmp_path, path).map_err(|e| InferenceError::WriteFailed {
        path: path.display().to_string(),
        source: e,
    })?;

    let download_mb = bytes.len() as f64 / 1_048_576.0;
    let size_mb = model_bytes.len() as f64 / 1_048_576.0;
    info!(
        model = spec.name,
        download_mb = download_mb,
        size_mb = size_mb,
        sha256 = actual_hash,
        "Model downloaded and verified"
    );

    Ok(())
}

/// Extracts a single named member from an in-memory zip archive.
fn extract_zip_member(
    archive_bytes: &[u8],
    member: &str,
    model_name: &str,
) -> Result<Vec<u8>, InferenceError> {
    let cursor = Cursor::new(archive_bytes);
    let mut archive = zip::ZipArchive::new(cursor).map_err(|e| InferenceError::DownloadFailed {
        model_name: model_name.to_string(),
        reason: format!("zip open failed: {e}"),
    })?;

    let mut entry = archive
        .by_name(member)
        .map_err(|e| InferenceError::DownloadFailed {
            model_name: model_name.to_string(),
            reason: format!("zip member '{member}' not found: {e}"),
        })?;

    if !entry.is_file() {
        return Err(InferenceError::DownloadFailed {
            model_name: model_name.to_string(),
            reason: format!("zip member '{member}' is not a file"),
        });
    }

    let size = usize::try_from(entry.size()).unwrap_or(0);
    let mut out = Vec::with_capacity(size);
    entry
        .read_to_end(&mut out)
        .map_err(|e| InferenceError::DownloadFailed {
            model_name: model_name.to_string(),
            reason: format!("zip extract '{member}' failed: {e}"),
        })?;

    Ok(out)
}

/// Checks whether all required models are present without downloading.
///
/// Returns a list of *required* model names that are missing.
/// Optional models (e.g. the high-quality hand landmarker) are checked
/// separately via [`check_optional_models`].
pub fn check_models(config: &ModelsConfig) -> Vec<&'static str> {
    let mut missing = Vec::new();
    for entry in &required_entries(config) {
        if !entry.path.exists() {
            warn!(model = entry.spec.name, path = %entry.path.display(), "Model file missing");
            missing.push(entry.spec.name);
        }
    }
    missing
}

/// Returns the list of optional model names that aren't on disk yet.
/// Callers can use this to decide whether to attempt the high-quality
/// path or stick with the lite fallback.
pub fn check_optional_models(config: &ModelsConfig) -> Vec<&'static str> {
    let mut missing = Vec::new();
    for entry in &optional_entries(config) {
        if !entry.path.exists() {
            missing.push(entry.spec.name);
        }
    }
    missing
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn missing_config() -> ModelsConfig {
        ModelsConfig {
            palm_detection: PathBuf::from("/tmp/nailbite_test_nonexistent/palm.onnx"),
            hand_landmark: PathBuf::from("/tmp/nailbite_test_nonexistent/hand.onnx"),
            hand_landmark_full: PathBuf::from("/tmp/nailbite_test_nonexistent/hand_full.onnx"),
            hand_landmark_quality: crate::config::HandLandmarkQuality::Auto,
            face_detection: PathBuf::from("/tmp/nailbite_test_nonexistent/face_det.onnx"),
            face_mesh: PathBuf::from("/tmp/nailbite_test_nonexistent/face_mesh.onnx"),
            pose_landmark: PathBuf::from("/tmp/nailbite_test_nonexistent/pose_lm.onnx"),
        }
    }

    fn devnull_config() -> ModelsConfig {
        ModelsConfig {
            palm_detection: PathBuf::from("/dev/null"),
            hand_landmark: PathBuf::from("/dev/null"),
            hand_landmark_full: PathBuf::from("/dev/null"),
            hand_landmark_quality: crate::config::HandLandmarkQuality::Auto,
            face_detection: PathBuf::from("/dev/null"),
            face_mesh: PathBuf::from("/dev/null"),
            pose_landmark: PathBuf::from("/dev/null"),
        }
    }

    #[test]
    fn check_models_reports_missing() {
        let config = missing_config();
        let missing = check_models(&config);
        assert_eq!(missing.len(), 5);
        assert!(missing.contains(&"palm_detection"));
        assert!(missing.contains(&"hand_landmark"));
        assert!(missing.contains(&"face_detection"));
        assert!(missing.contains(&"face_mesh"));
        assert!(missing.contains(&"pose_landmark"));
    }

    #[test]
    fn check_optional_models_reports_full_missing() {
        let config = missing_config();
        let missing = check_optional_models(&config);
        assert_eq!(missing, vec!["hand_landmark_full"]);
    }

    #[test]
    fn check_models_reports_none_missing_when_all_exist() {
        // Use files that definitely exist on the system.
        let config = devnull_config();
        let missing = check_models(&config);
        assert!(missing.is_empty());
        let optional = check_optional_models(&config);
        assert!(optional.is_empty());
    }

    #[test]
    fn raw_specs_use_https_and_onnx() {
        let raw = [
            &PALM_DETECTION,
            &HAND_LANDMARK,
            &FACE_DETECTION,
            &FACE_MESH,
            &POSE_LANDMARK,
        ];
        for spec in raw {
            assert!(
                spec.url.starts_with("https://"),
                "Model {} URL must use HTTPS",
                spec.name
            );
            assert!(
                spec.archive_member.is_none(),
                "Model {} should be a raw .onnx",
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
    fn archived_specs_declare_member() {
        // Models distributed as .zip archives must point at a specific
        // .onnx member inside; otherwise we have nothing to extract.
        let archived = [&HAND_LANDMARK_FULL];
        for spec in archived {
            assert!(spec.url.starts_with("https://"));
            assert!(spec.url.ends_with(".zip"));
            let member = spec.archive_member.expect("archive_member required");
            assert!(
                member.ends_with(".onnx"),
                "archive_member for {} must end with .onnx",
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
        let specs = [
            &PALM_DETECTION,
            &HAND_LANDMARK,
            &HAND_LANDMARK_FULL,
            &FACE_DETECTION,
            &FACE_MESH,
            &POSE_LANDMARK,
        ];
        let names: Vec<&str> = specs.iter().map(|s| s.name).collect();
        let mut deduped = names.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(names.len(), deduped.len(), "Model names must be unique");
    }
}
