# Nailbite

BFRB (body-focused repetitive behavior) detection and decoupling exercise system.

## Architecture

- System tray application written in Rust
- Uses ONNX models (MediaPipe hand landmarks + face mesh) via the `ort` crate for detection
- Camera capture via V4L2 (`v4l` crate) behind a `CameraBackend` trait for future cross-platform support
- Detection runs on a dedicated thread at configurable inference FPS (default 8)
- Communication between threads via `crossbeam-channel`

## Key Design Decisions

- All BFRB detectors implement the `BehaviorDetector` trait in `src/detection/behaviors/`
- All exercises implement the `Exercise` trait in `src/exercises/`
- All actions implement the `Action` trait in `src/actions/`
- Platform-specific code is behind traits (`CameraBackend`) with `#[cfg(target_os)]` gating
- Single YAML config file (`config.yaml`)

## Building

```bash
bash build.sh        # Docker-based build
cargo build          # Local build (requires system deps)
cargo test           # Run tests
cargo clippy         # Lint
```

## System Dependencies (NixOS)

- `alsa-lib` (for rodio audio)
- `gtk3` (for tray-icon)
- `libappindicator-gtk3` or `libayatana-appindicator` (for system tray on GNOME)
- ONNX Runtime shared library (`libonnxruntime.so`)

## Models

ONNX models are automatically downloaded on first run if not present at the configured paths. The download logic lives in `src/inference/model_downloader.rs`. Models are not committed to the repo.

Sources:
- Palm detection & hand landmark: [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo)
- Face detection & face mesh: [IntelliProve/face-detection-onnx](https://github.com/IntelliProve/face-detection-onnx)
