# Nailbite

**BFRB (Body-Focused Repetitive Behavior) detection and decoupling exercise system.**

Nailbite uses your webcam to detect nail biting, nail picking, and other BFRBs in real-time. When a behavior is detected, it guides you through evidence-based decoupling exercises from psychotherapy. Everything runs locally on your machine - no data is sent to any server.

## Features

- **Real-time BFRB Detection**: Detects nail biting and nail picking using AI-based hand and face tracking
- **Decoupling Exercises**: Guided exercises based on Habit Reversal Training (Azrin & Nunn) and the Moritz Decoupling Protocol
- **System Tray Integration**: Runs in background, tray icon shows status
- **Audio Alerts**: Configurable sound alerts when BFRB is detected
- **Statistics Tracking**: Track detections and exercise completion over time
- **GPU Acceleration**: Optional CUDA/TensorRT (NVIDIA) or ROCm/MIGraphX (AMD) support
- **Privacy-First**: All processing runs locally, no cloud services

## Supported Behaviors

- Nail Biting (fingertips near mouth)
- Nail Picking (hands together in picking gesture)
- More behaviors planned (hair pulling, skin picking, lip biting)

## Requirements

- Linux (NixOS recommended, Debian/Ubuntu also supported)
- Webcam (V4L2 compatible)
- Rust 1.85+, Node.js 22+, pnpm

### System Dependencies (Debian/Ubuntu)

```bash
sudo apt install \
    libwebkit2gtk-4.1-dev libgtk-3-dev libglib2.0-dev \
    libayatana-appindicator3-dev librsvg2-dev libsoup-3.0-dev \
    libjavascriptcoregtk-4.1-dev libasound2-dev libxdo-dev \
    libv4l-dev libssl-dev cmake clang
```

## Quick Start

### Using Nix (Recommended)

```bash
# Clone the repo
git clone https://github.com/your-username/nailbite.git
cd nailbite

# Enter the Nix shell
nix-shell

# Install dependencies and run
pnpm install
pnpm tauri dev
```

### Manual Build

```bash
# Install frontend dependencies
pnpm install

# Run in development mode
pnpm tauri dev

# Build for production
pnpm tauri build
```

### Docker Build

```bash
# CPU-only build
bash build.sh

# With NVIDIA GPU support
GPU_BACKEND=cuda bash build.sh
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# Camera settings
camera:
  inference_fps: 8
  sources:
    - id: main
      device: /dev/video0
      role: primary

# Detection settings
detection:
  behaviors:
    nail_biting:
      enabled: true
      proximity_threshold: 0.35

# GPU acceleration (optional)
ort:
  gpu:
    preference: auto  # auto, disabled, or required
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| F9 | Dismiss alert (mark as false positive) |
| F10 | Mark missed event (false negative) |
| F11 | Pause/Resume detection |

## How It Works

1. **Camera Capture**: Captures frames from your webcam using V4L2
2. **AI Detection**: Runs ONNX models for hand, face, and pose detection
3. **Behavior Analysis**: Detects BFRB patterns (e.g., fingertips near mouth)
4. **Alert & Exercise**: Plays alert sound and shows exercise instructions
5. **Verification**: Uses camera to verify you're doing the exercise correctly

ONNX models are automatically downloaded on first run (~50MB total).

## Privacy

All processing happens locally on your device:
- No video/images are sent anywhere
- No cloud services are used
- Statistics are stored locally in `~/.local/share/nailbite/`
- Webhook feature (disabled by default) is the only network feature

## Development

```bash
# Run tests
pnpm test                    # Frontend
cd src-tauri && cargo test   # Backend

# Lint
pnpm lint
cd src-tauri && cargo clippy

# Type check
pnpm typecheck
```

## License

MIT

## Acknowledgments

- Detection models from [OpenCV Zoo](https://github.com/opencv/opencv_zoo) and [IntelliProve](https://github.com/IntelliProve/face-detection-onnx)
- Decoupling exercises based on research from UKE Hamburg ([tricks-gegen-ticks.de](https://tricks-gegen-ticks.de))
- Built with [Tauri](https://tauri.app/), [React](https://react.dev/), and [ONNX Runtime](https://onnxruntime.ai/)
