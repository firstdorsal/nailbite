# n**AI**lbite

BFRB (body-focused repetitive behavior) detection system.

## Architecture

**Tauri 2 + React Frontend:**
- Desktop application built with Tauri 2 (Rust backend + React/TypeScript frontend)
- System tray with status indicator (green=monitoring, red=detecting, yellow=paused, dark gray=no one in frame, light gray=offline)
- React frontend for camera preview, event history, insights, and settings

**Detection Pipeline:**
- Camera capture via V4L2 (`v4l` crate) on Linux
- Camera controls auto-tuned at capture start (auto-exposure on, frame-rate priority off, backlight compensation max, brightness lifted ~60 %) so the subject stays well-lit regardless of background lighting; configurable under `camera.controls`. Helpers + pure stepping math live in `src-tauri/src/camera/controls.rs`.
- ONNX inference via `ort` crate (supports CPU, CUDA, TensorRT, ROCm, MIGraphX)
- Models: Palm detection, hand landmarks (MediaPipe lite or RTMPose-m full), face detection, face mesh, pose landmarks
- Detection runs on dedicated thread at configurable inference FPS (default 8)
- Hand tracking for temporal consistency and left/right hand identity
- Multi-modal presence gate (face mesh + pose torso landmarks must both agree) silences detection when the user is not in frame

**Key Components:**
- `src-tauri/src/` - Rust backend (Tauri commands, inference, detection)
- `src/` - React frontend (TypeScript, Tailwind, shadcn/ui)
- `config.yaml` - Single YAML configuration file

## Key Design Decisions

- All BFRB detectors implement the `BehaviorDetector` trait in `src-tauri/src/detection/behaviors/`
- All actions implement the `Action` trait in `src-tauri/src/actions/`
- Presence tracker (`PresenceTracker` in `src-tauri/src/commands/camera.rs`) gates the entire detection chain on a multi-signal "user in frame" vote with asymmetric hysteresis
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
│       ├── commands/       # Tauri IPC commands (camera, history, labels, ...)
│       ├── detection/      # BFRB detectors, hand tracker, fusion
│       ├── inference/      # ONNX model wrappers + downloader
│       ├── stats/          # Session log + event history + landmark annotation
│       ├── actions/        # Sound, webhook, notification actions
│       └── camera/         # V4L2 camera capture
├── config.yaml             # Configuration file
├── Dockerfile              # Docker build file
└── build.sh                # Build script
```

## Configuration

See `config.yaml` for all options. Key settings:

- `camera.sources` - Camera devices and roles
- `camera.controls` - Subject-friendly biasing applied at capture start (brightness/contrast fractions, backlight max, auto-exposure flags)
- `detection.behaviors` - Enable/configure BFRB detectors
- `detection.tracking` - Hand-tracker stability tuning
- `ort.gpu` - GPU acceleration settings
- `ort.graph_optimization` - ORT graph-optimization level (default `extended`; `all` may trip ORT 1.23 CPU-EP bugs)
- `actions` - Sound alerts, webhooks, desktop notifications
- `history` - Event recording (clip length, max events, landmark annotation)
