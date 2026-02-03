//! Path utility functions shared across the codebase.

use std::path::{Path, PathBuf};

/// Expand `~` to the user's home directory.
///
/// Only expands `~/...` at the beginning of a path. If `HOME` is not set,
/// returns the path unchanged.
pub fn expand_tilde(path: &Path) -> PathBuf {
    let path_str = path.to_string_lossy();
    if let Some(rest) = path_str.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_tilde_replaces_home() {
        if std::env::var_os("HOME").is_some() {
            let expanded = expand_tilde(Path::new("~/test/path"));
            assert!(!expanded.to_string_lossy().starts_with("~/"));
            assert!(expanded.to_string_lossy().ends_with("test/path"));
        }
    }

    #[test]
    fn expand_tilde_noop_for_absolute() {
        let path = Path::new("/absolute/path");
        assert_eq!(expand_tilde(path), path);
    }

    #[test]
    fn expand_tilde_noop_for_relative() {
        let path = Path::new("relative/path");
        assert_eq!(expand_tilde(path), path);
    }

    #[test]
    fn expand_tilde_noop_for_bare_tilde() {
        // Only ~/... is expanded, not bare ~
        let path = Path::new("~");
        assert_eq!(expand_tilde(path), path);
    }
}
