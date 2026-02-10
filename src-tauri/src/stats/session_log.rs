//! Session logging in JSONL format.
//!
//! Records detections, exercise completions, and user annotations
//! for statistics and progress tracking.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::Utc;
use serde::Serialize;
use tracing::{debug, warn};

use crate::detection::types::BfrbType;
use crate::paths::expand_tilde;

/// A single log entry in the session log.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum LogEntry {
    /// A BFRB was detected and confirmed.
    #[serde(rename = "detection")]
    Detection {
        timestamp: String,
        bfrb_type: BfrbType,
        confidence: f32,
        camera_id: String,
    },
    /// An exercise was completed.
    #[serde(rename = "exercise_completed")]
    ExerciseCompleted {
        timestamp: String,
        exercise_id: String,
        bfrb_type: BfrbType,
        compliance_ratio: f32,
    },
    /// User dismissed an alert (false positive).
    #[serde(rename = "dismissed")]
    Dismissed {
        timestamp: String,
        bfrb_type: Option<BfrbType>,
    },
    /// User flagged a missed event (false negative).
    #[serde(rename = "missed_event")]
    MissedEvent { timestamp: String },
    /// Detection paused.
    #[serde(rename = "paused")]
    Paused { timestamp: String },
    /// Detection resumed.
    #[serde(rename = "resumed")]
    Resumed { timestamp: String },
}

/// Maximum log file size before rotation (10 MB).
const MAX_LOG_SIZE_BYTES: u64 = 10 * 1024 * 1024;

/// Writes session log entries to a JSONL file with automatic rotation (SECURITY-6).
pub struct SessionLog {
    file_path: PathBuf,
}

impl SessionLog {
    /// Create a new session log writer.
    pub fn new(file_path: &Path) -> Result<Self, String> {
        let file_path = expand_tilde(file_path);

        // Ensure parent directory exists.
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create log directory: {e}"))?;
        }

        Ok(Self { file_path })
    }

    /// Check if log file needs rotation and rotate if necessary (SECURITY-6).
    fn maybe_rotate(&self) {
        let Ok(metadata) = fs::metadata(&self.file_path) else {
            return; // File doesn't exist yet
        };

        if metadata.len() < MAX_LOG_SIZE_BYTES {
            return; // File is under limit
        }

        // Rotate: rename current file to .old, deleting any existing .old file
        let old_path = self.file_path.with_extension("jsonl.old");

        if old_path.exists() {
            if let Err(e) = fs::remove_file(&old_path) {
                warn!(error = %e, "Failed to remove old log file during rotation");
            }
        }

        if let Err(e) = fs::rename(&self.file_path, &old_path) {
            warn!(error = %e, "Failed to rotate log file");
        } else {
            debug!(
                old_file = %old_path.display(),
                "Log file rotated"
            );
        }
    }

    /// Append a log entry to the session log file.
    pub fn log(&self, entry: &LogEntry) {
        // Check for rotation before writing
        self.maybe_rotate();

        let json = match serde_json::to_string(entry) {
            Ok(j) => j,
            Err(e) => {
                warn!(error = %e, "Failed to serialize log entry");
                return;
            }
        };

        let result = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file_path)
            .and_then(|mut f| writeln!(f, "{json}"));

        match result {
            Ok(()) => {
                debug!(file = %self.file_path.display(), "Session log entry written");
            }
            Err(e) => {
                warn!(
                    error = %e,
                    file = %self.file_path.display(),
                    "Failed to write session log entry"
                );
            }
        }
    }

    /// Log a detection event.
    pub fn log_detection(&self, bfrb_type: BfrbType, confidence: f32, camera_id: &str) {
        self.log(&LogEntry::Detection {
            timestamp: Utc::now().to_rfc3339(),
            bfrb_type,
            confidence,
            camera_id: camera_id.to_string(),
        });
    }

    /// Log an exercise completion.
    pub fn log_exercise_completed(
        &self,
        exercise_id: &str,
        bfrb_type: BfrbType,
        compliance_ratio: f32,
    ) {
        self.log(&LogEntry::ExerciseCompleted {
            timestamp: Utc::now().to_rfc3339(),
            exercise_id: exercise_id.to_string(),
            bfrb_type,
            compliance_ratio,
        });
    }

    /// Log a dismissal (false positive).
    pub fn log_dismissed(&self, bfrb_type: Option<BfrbType>) {
        self.log(&LogEntry::Dismissed {
            timestamp: Utc::now().to_rfc3339(),
            bfrb_type,
        });
    }

    /// Log a missed event (false negative reported by user).
    pub fn log_missed_event(&self) {
        self.log(&LogEntry::MissedEvent {
            timestamp: Utc::now().to_rfc3339(),
        });
    }

    /// Log pause.
    pub fn log_paused(&self) {
        self.log(&LogEntry::Paused {
            timestamp: Utc::now().to_rfc3339(),
        });
    }

    /// Log resume.
    pub fn log_resumed(&self) {
        self.log(&LogEntry::Resumed {
            timestamp: Utc::now().to_rfc3339(),
        });
    }

    /// Read stats from the log file.
    ///
    /// Returns aggregated statistics from all log entries.
    pub fn read_stats(&self) -> SessionStats {
        let mut stats = SessionStats::default();

        let Ok(content) = fs::read_to_string(&self.file_path) else {
            return stats;
        };

        for line in content.lines() {
            let Ok(entry) = serde_json::from_str::<LogEntryRead>(line) else {
                continue;
            };

            stats.entries.push(SessionStatsEntry {
                timestamp: entry.timestamp.clone(),
                event_type: entry.entry_type.clone(),
                bfrb_type: entry.bfrb_type.map(|b| b.to_string()),
                exercise_id: entry.exercise_id.clone(),
                duration_secs: None,
            });

            match entry.entry_type.as_str() {
                "detection" => stats.total_detections += 1,
                "exercise_completed" => stats.total_exercises_completed += 1,
                "dismissed" => stats.total_dismissed += 1,
                "missed_event" => stats.total_missed += 1,
                _ => {}
            }
        }

        stats
    }

    /// Get the file path (for testing).
    pub fn file_path(&self) -> &Path {
        &self.file_path
    }
}

/// Stats returned by read_stats.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SessionStats {
    pub entries: Vec<SessionStatsEntry>,
    pub total_detections: u32,
    pub total_exercises_completed: u32,
    pub total_dismissed: u32,
    pub total_missed: u32,
}

/// A single entry in the stats.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionStatsEntry {
    pub timestamp: String,
    pub event_type: String,
    pub bfrb_type: Option<String>,
    pub exercise_id: Option<String>,
    pub duration_secs: Option<u64>,
}

/// Internal struct for reading log entries (subset of fields).
#[derive(Debug, serde::Deserialize)]
struct LogEntryRead {
    timestamp: String,
    #[serde(rename = "type")]
    entry_type: String,
    bfrb_type: Option<BfrbType>,
    exercise_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_detection_log_entry() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("stats.jsonl");

        let log = SessionLog::new(&log_path).unwrap();
        log.log_detection(BfrbType::NailBiting, 0.85, "main");

        let content = fs::read_to_string(&log_path).unwrap();
        assert!(content.contains("\"type\":\"detection\""));
        assert!(content.contains("nail_biting"));
    }

    #[test]
    fn writes_multiple_entries() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("stats.jsonl");

        let log = SessionLog::new(&log_path).unwrap();
        log.log_detection(BfrbType::NailBiting, 0.85, "main");
        log.log_dismissed(Some(BfrbType::NailBiting));
        log.log_missed_event();
        log.log_paused();
        log.log_resumed();

        let content = fs::read_to_string(&log_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 5);
    }

    #[test]
    fn exercise_completed_serializes() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("stats.jsonl");

        let log = SessionLog::new(&log_path).unwrap();
        log.log_exercise_completed("fist_clench", BfrbType::NailBiting, 0.92);

        let content = fs::read_to_string(&log_path).unwrap();
        assert!(content.contains("\"type\":\"exercise_completed\""));
        assert!(content.contains("fist_clench"));
    }

    #[test]
    fn read_stats_parses_log_file() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("stats.jsonl");

        let log = SessionLog::new(&log_path).unwrap();
        log.log_detection(BfrbType::NailBiting, 0.85, "main");
        log.log_detection(BfrbType::NailPicking, 0.75, "main");
        log.log_exercise_completed("fist_clench", BfrbType::NailBiting, 0.92);
        log.log_dismissed(Some(BfrbType::NailBiting));
        log.log_missed_event();

        let stats = log.read_stats();
        assert_eq!(stats.total_detections, 2);
        assert_eq!(stats.total_exercises_completed, 1);
        assert_eq!(stats.total_dismissed, 1);
        assert_eq!(stats.total_missed, 1);
        assert_eq!(stats.entries.len(), 5);
    }

    #[test]
    fn read_stats_handles_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("nonexistent.jsonl");

        let log = SessionLog::new(&log_path).unwrap();
        let stats = log.read_stats();

        assert_eq!(stats.total_detections, 0);
        assert!(stats.entries.is_empty());
    }
}
