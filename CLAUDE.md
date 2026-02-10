# Nailbite

BFRB (body-focused repetitive behavior) detection and decoupling exercise system.

## Architecture

**Tauri 2 + React Frontend:**
- Desktop application built with Tauri 2 (Rust backend + React/TypeScript frontend)
- System tray with status indicator (green=ready, red=detecting, yellow=not ready)
- React frontend for camera preview, settings, exercises, and statistics

**Detection Pipeline:**
- Camera capture via V4L2 (`v4l` crate) on Linux
- ONNX inference via `ort` crate (supports CPU, CUDA, TensorRT, ROCm, MIGraphX)
- Models: Palm detection, hand landmarks, face detection, face mesh, pose detection, pose landmarks
- Detection runs on dedicated thread at configurable inference FPS (default 8)
- Hand tracking for temporal consistency and left/right hand identity

**Key Components:**
- `src-tauri/src/` - Rust backend (Tauri commands, inference, detection)
- `src/` - React frontend (TypeScript, Tailwind, shadcn/ui)
- `config.yaml` - Single YAML configuration file

## Key Design Decisions

- All BFRB detectors implement the `BehaviorDetector` trait in `src-tauri/src/detection/behaviors/`
- All exercises implement the `Exercise` trait in `src-tauri/src/exercises/`
- All actions implement the `Action` trait in `src-tauri/src/actions/`
- Platform-specific code is behind traits with `#[cfg(target_os)]` gating
- Linux-only for now (V4L2 camera), cross-platform support planned

## Building

### Development
```bash
# Enter Nix shell with all dependencies
nix-shell

# Install frontend dependencies
pnpm install

# Run in development mode
pnpm tauri dev

# Run tests
pnpm test                    # Frontend tests
cd src-tauri && cargo test   # Backend tests
```

### Production Build
```bash
# Docker-based build (recommended for reproducibility)
bash build.sh

# With GPU support
GPU_BACKEND=cuda bash build.sh     # NVIDIA CUDA
GPU_BACKEND=tensorrt bash build.sh # NVIDIA TensorRT
GPU_BACKEND=rocm bash build.sh     # AMD ROCm

# Local build (requires all system deps)
pnpm tauri build
```

## GPU Support

The `ort` crate supports multiple GPU execution providers. Configure in `config.yaml`:

```yaml
ort:
  gpu:
    preference: auto    # auto, disabled, or required
    backend: auto       # auto, cuda, tensorrt, migraphx
    device_id: 0
    fp16_enable: true
```

**Build Features:**
- `cuda` - NVIDIA CUDA support
- `tensorrt` - NVIDIA TensorRT (includes CUDA fallback)
- `migraphx` - AMD MIGraphX support
- `rocm` - AMD ROCm support

ONNX Runtime uses dynamic loading (`load-dynamic` feature). The appropriate shared library (`libonnxruntime.so`) must be available at runtime. Models download automatically on first run with SHA256 verification.

## System Dependencies (NixOS)

- `alsa-lib` (rodio audio)
- `gtk3`, `webkitgtk` (Tauri/WebKitGTK)
- `libayatana-appindicator` (system tray)
- `libv4l` (camera capture)
- `openssl` (TLS)

See `shell.nix` for complete list.

## Models

ONNX models are automatically downloaded on first run. The download logic is in `src-tauri/src/inference/model_downloader.rs`.

| Model | Source | Input Size | Purpose |
|-------|--------|------------|---------|
| Palm Detection | [opencv_zoo](https://github.com/opencv/opencv_zoo) | 192x192 | Hand bounding boxes |
| Hand Landmark | [opencv_zoo](https://github.com/opencv/opencv_zoo) | 224x224 | 21 hand keypoints |
| Face Detection | [IntelliProve](https://github.com/IntelliProve/face-detection-onnx) | 128x128 | Face bounding box |
| Face Mesh | [IntelliProve](https://github.com/IntelliProve/face-detection-onnx) | 192x192 | 468 face landmarks |
| Pose Detection | [HuggingFace](https://huggingface.co/unity/inference-engine-blaze-pose) | Variable | Body pose |
| Pose Landmark | [HuggingFace](https://huggingface.co/unity/inference-engine-blaze-pose) | 256x256 | 33 body keypoints |

## Project Structure

```
nailbite/
├── src/                    # React frontend
│   ├── components/         # UI components (shadcn/ui)
│   ├── hooks/              # React hooks (useCamera, useDetection, etc.)
│   ├── pages/              # Page components
│   └── types/              # TypeScript types
├── src-tauri/              # Tauri Rust backend
│   └── src/
│       ├── commands/       # Tauri IPC commands
│       ├── detection/      # BFRB detectors, tracker
│       ├── exercises/      # Decoupling exercises
│       ├── inference/      # ONNX model wrappers
│       ├── actions/        # Sound, webhook actions
│       └── camera/         # V4L2 camera capture
├── config.yaml             # Configuration file
├── Dockerfile*             # Docker build files
└── build.sh                # Build script
```

## Configuration

See `config.yaml` for all options. Key settings:

- `camera.sources` - Camera devices and roles
- `detection.behaviors` - Enable/configure BFRB detectors
- `ort.gpu` - GPU acceleration settings
- `actions` - Sound alerts, webhooks
- `exercises` - Selection strategy, timeouts
