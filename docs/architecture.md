# Architecture

Nailbite is a desktop application for detecting and interrupting body-focused repetitive behaviors (BFRBs) like nail biting and nail picking.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Tauri Application                         │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (React + TypeScript)          │  Backend (Rust)        │
│  ┌─────────────────────────────────┐    │  ┌──────────────────┐  │
│  │  Pages                          │    │  │  Camera Capture  │  │
│  │  - PreviewPage                  │    │  │  (V4L2)          │  │
│  │  - SettingsPage                 │    │  └────────┬─────────┘  │
│  │  - StatsPage                    │    │           │            │
│  │  - ExercisePage                 │    │  ┌────────▼─────────┐  │
│  └─────────────────────────────────┘    │  │  ONNX Inference  │  │
│  ┌─────────────────────────────────┐    │  │  - Hand Landmark │  │
│  │  Hooks                          │    │  │  - Face Mesh     │  │
│  │  - useCamera (frame events)     │◄───┼──│  - Pose          │  │
│  │  - useConfig                    │    │  └────────┬─────────┘  │
│  │  - useExercise                  │    │           │            │
│  │  - useDetection                 │    │  ┌────────▼─────────┐  │
│  └─────────────────────────────────┘    │  │  BFRB Detectors  │  │
│                                         │  │  - Nail Biting   │  │
│  ┌─────────────────────────────────┐    │  │  - Nail Picking  │  │
│  │  Components                     │    │  └────────┬─────────┘  │
│  │  - LandmarkCanvas               │    │           │            │
│  │  - AlertModal                   │    │  ┌────────▼─────────┐  │
│  │  - StatusIndicator              │    │  │  Actions         │  │
│  └─────────────────────────────────┘    │  │  - Sound Alert   │  │
│                                         │  │  - Webhook       │  │
│                                         │  │  - Exercise      │  │
│                                         │  └──────────────────┘  │
├─────────────────────────────────────────┴────────────────────────┤
│                         System Tray                               │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### Frontend (src/)

The frontend is built with React 19 and TypeScript, using:
- **Vite** for bundling
- **Tailwind CSS** for styling
- **shadcn/ui** for UI components
- **Tauri API** for backend communication

Key files:
- `src/App.tsx` - Main application with routing
- `src/hooks/useCamera.ts` - Camera streaming and detection events
- `src/hooks/useConfig.ts` - Configuration management
- `src/hooks/useExercise.ts` - Exercise session management
- `src/components/LandmarkCanvas.tsx` - Hand/face landmark visualization

### Backend (src-tauri/src/)

The backend is written in Rust and handles:
- Camera capture via V4L2
- ONNX model inference
- BFRB behavior detection
- Alert actions (sound, webhook)
- Exercise verification

Key modules:
- `camera/` - V4L2 camera capture
- `inference/` - ONNX model loading and inference
- `detection/` - Behavior detection algorithms
- `exercises/` - Decoupling exercise definitions
- `actions/` - Alert actions (sound, webhook)
- `commands/` - Tauri IPC commands

### Communication

Frontend and backend communicate via:
1. **Tauri Commands** - Request/response pattern for actions
2. **Tauri Events** - Real-time streaming of frame data and detections

```
Frontend                              Backend
   │                                     │
   │──── invoke("start_camera") ────────►│
   │                                     │
   │◄──── event("frame-update") ─────────│ (repeating at inference_fps)
   │      { frame_base64, hands,         │
   │        face, detections, ... }      │
   │                                     │
   │──── invoke("dismiss_alert") ───────►│
   │                                     │
```

## Detection Pipeline

1. **Frame Capture** - V4L2 captures frames from webcam
2. **Preprocessing** - Resize and normalize for ONNX models
3. **Inference** - Run hand/face/pose detection models
4. **Analysis** - Calculate spatial relationships (hand-to-mouth distance, etc.)
5. **Temporal Tracking** - Confirm sustained behavior over time window
6. **Alert Trigger** - Play sound, send webhook, show exercise

## Data Flow

```
Camera Frame
     │
     ▼
┌────────────┐    ┌────────────┐    ┌────────────┐
│   Palm     │───►│   Hand     │───►│  Behavior  │
│ Detection  │    │ Landmark   │    │  Detector  │
└────────────┘    └────────────┘    └────────────┘
                                          │
┌────────────┐    ┌────────────┐          │
│   Face     │───►│   Face     │──────────┤
│ Detection  │    │   Mesh     │          │
└────────────┘    └────────────┘          │
                                          ▼
                                   ┌────────────┐
                                   │  Temporal  │
                                   │  Tracker   │
                                   └────────────┘
                                          │
                                          ▼
                                   ┌────────────┐
                                   │   Alert    │
                                   │  Actions   │
                                   └────────────┘
```

## State Management

### Backend State (AppState)

Centralized state in `src-tauri/src/state.rs`:
- Model sessions (shared, read-only)
- Behavior detectors (RwLock for hot-reload)
- Detection tracker (temporal state machine)
- Configuration (RwLock for hot-reload)
- Camera handles
- Session log

### Frontend State

- React Context for global detection state
- React hooks for local component state
- Tauri events for real-time updates

## Security Considerations

- **Camera permissions** - WebKit permissions filtered to camera only
- **Webhook SSRF protection** - Internal IPs blocked, CRLF validation
- **Path traversal** - Model paths validated with canonicalization
- **Log rotation** - Session logs rotated at 10MB

## Platform Support

Currently Linux-only:
- V4L2 for camera capture
- GTK for system tray
- WebKitGTK for webview

Future: macOS (AVFoundation) and Windows (Media Foundation) support possible via `CameraBackend` trait.
