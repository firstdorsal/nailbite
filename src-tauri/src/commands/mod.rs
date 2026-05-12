//! Tauri commands for the frontend.

#[cfg(target_os = "linux")]
pub mod camera;
pub mod config;
pub mod detection;
pub mod exercises;
pub mod history;
pub mod labels;
pub mod stats;
