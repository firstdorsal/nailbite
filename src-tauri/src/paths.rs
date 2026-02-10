//! Path utilities.

use std::path::{Path, PathBuf};

/// Expand tilde (~) to user's home directory.
pub fn expand_tilde(path: &Path) -> PathBuf {
    let path_str = path.to_string_lossy();
    if let Some(stripped) = path_str.strip_prefix("~/") {
        if let Some(home) = dirs_home() {
            return home.join(stripped);
        }
    }
    path.to_path_buf()
}

/// Get the user's home directory.
fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_tilde_expands_home() {
        let path = Path::new("~/.local/share/nailbite");
        let expanded = expand_tilde(path);
        assert!(!expanded.to_string_lossy().starts_with('~'));
    }

    #[test]
    fn expand_tilde_leaves_absolute_unchanged() {
        let path = Path::new("/etc/nailbite/config.yaml");
        let expanded = expand_tilde(path);
        assert_eq!(expanded, path);
    }

    #[test]
    fn expand_tilde_leaves_relative_unchanged() {
        let path = Path::new("./models/hand.onnx");
        let expanded = expand_tilde(path);
        assert_eq!(expanded, path);
    }
}
