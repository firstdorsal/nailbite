# Nailbite

**BFRB (Body-Focused Repetitive Behavior) detection system.**

Nailbite uses your webcam to detect nail biting, nail picking, and similar BFRBs in real time, alerts you when one happens, and lets you label each event as a true or false positive to tune the detector over time. Everything runs locally on your machine — no data is sent to any server.

## Features

- **Real-time BFRB Detection**: Detects nail biting and nail picking using AI-based hand and face tracking
- **Presence-gated**: Multi-modal presence check (face mesh + pose torso landmarks) ensures detection is silenced when the user is not in frame
- **System Tray Integration**: Runs in background; tray icon shows monitoring / detecting / paused / absent state
- **Desktop Notifications**: Inline "Correct / False positive" buttons for in-the-moment labeling
- **Event History + Insights**: Every detection captures a multi-frame clip with the contributing signals; the Insights page sweeps thresholds against your labels to suggest tuning
- **Audio Alerts**: Configurable sound alerts when BFRB is detected
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
2. **AI Detection**: Runs ONNX models for hand, face, and pose landmarks
3. **Presence Gate**: Face mesh + pose torso landmarks have to agree on "user in frame" before detection runs — gates off the empty chair
4. **Behavior Analysis**: Detects BFRB patterns (e.g., fingertips near mouth) with temporal smoothing and per-hand explanations
5. **Alert + Label**: Plays the alert sound, surfaces a verdict notification, and records the captured clip + signals to the event history for later review and threshold tuning

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
- Built with [Tauri](https://tauri.app/), [React](https://react.dev/), and [ONNX Runtime](https://onnxruntime.ai/)
