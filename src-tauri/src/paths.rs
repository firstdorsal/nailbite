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

/// Resolve a relative data path (config file, model file, ...) against the
/// user's *original* working directory.
///
/// When the app is launched via `appimage-run`, the AppImage's internal `AppRun`
/// script changes cwd into `$APPDIR/usr` before exec'ing the binary, so any
/// `./foo` path would resolve into the read-only AppImage tree instead of the
/// directory the user actually launched from. `appimage-run` exports the
/// invoker cwd as `$OWD` for exactly this case; we rebase relative paths
/// against it. We can't simply `chdir(OWD)` at startup because that breaks
/// WebKitGTK's own relative lookup of `WebKitNetworkProcess` next to the GTK
/// libraries inside the AppImage.
///
/// Absolute paths are returned unchanged.
pub fn resolve_data_path<P: AsRef<Path>>(path: P) -> PathBuf {
    resolve_against_owd(path.as_ref(), std::env::var_os("OWD").as_deref())
}

/// Pure implementation of [`resolve_data_path`] for unit testing.
/// Lifting the env lookup keeps the tests from racing on `$OWD`.
fn resolve_against_owd(path: &Path, owd: Option<&std::ffi::OsStr>) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    match owd {
        Some(owd) if !owd.is_empty() => PathBuf::from(owd).join(path),
        _ => path.to_path_buf(),
    }
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

    #[test]
    fn resolve_against_owd_passes_absolute_through() {
        let path = Path::new("/etc/nailbite/config.yaml");
        let resolved = resolve_against_owd(
            path,
            Some(std::ffi::OsStr::new("/somewhere/else")),
        );
        assert_eq!(resolved, path);
    }

    #[test]
    fn resolve_against_owd_rebases_relative() {
        let resolved = resolve_against_owd(
            Path::new("config.yaml"),
            Some(std::ffi::OsStr::new("/home/user/project")),
        );
        assert_eq!(resolved, PathBuf::from("/home/user/project/config.yaml"));
    }

    #[test]
    fn resolve_against_owd_passes_relative_through_when_no_owd() {
        let resolved = resolve_against_owd(Path::new("./models/hand.onnx"), None);
        assert_eq!(resolved, PathBuf::from("./models/hand.onnx"));
    }

    #[test]
    fn resolve_against_owd_treats_empty_owd_as_unset() {
        let resolved = resolve_against_owd(
            Path::new("config.yaml"),
            Some(std::ffi::OsStr::new("")),
        );
        assert_eq!(resolved, PathBuf::from("config.yaml"));
    }
}
