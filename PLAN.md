# Nailbite: BFRB Detection & Decoupling System

## Overview

A Rust system tray application that uses webcam(s) to detect body-focused repetitive behaviors and guides the user through evidence-based decoupling exercises from psychotherapy. Runs fully local.

**MVP scope**: Single camera, nail biting + nail picking detection only. But all traits, interfaces, and config structures are designed for extensibility to: multiple cameras, additional BFRBs (hair pulling, skin picking, lip/cheek biting), and cross-platform support (Windows, macOS).

**Current target**: Linux only. Cross-platform support is a future goal -- all platform-specific code is behind abstraction traits to enable this.

---

## Architecture Summary

```
Camera Thread(s)  -->  Detection Thread  -->  Action Thread
  (V4L2 capture       (behavior analysis      (sound, notification,
   + ONNX inference)    + temporal tracking)    webhook, exercise popup)
       ^                                              |
       |                                              v
       +---- Exercise Verification <---- Exercise UI (egui)

Global Hotkey Listener (global-hotkey) --> Dismiss/Annotate events --> Training Data Log
System Tray (tray-icon + GTK) --> Preview/Settings windows (egui)
```

**Thread model**: Camera capture+inference threads (one per camera), detection thread, UI thread (GTK event loop + egui), global hotkey listener thread. Communication via `crossbeam-channel`.

---

## Crate Stack

| Component | Crate | Version |
|---|---|---|
| ONNX inference | `ort` | `2.0.0-rc.11` |
| Camera (V4L2) | `v4l` | `0.14.0` (behind `CameraBackend` trait for future cross-platform) |
| System tray | `tray-icon` | `0.21` |
| GUI/preview | `eframe` + `egui` | `0.33` |
| Audio | `rodio` | `0.21` |
| Notifications | `notify-rust` | `4.12` |
| HTTP client | `reqwest` | `0.13` (blocking+json) |
| Config | `serde` + `serde_yaml_ng` | `0.10.0` |
| Errors | `thiserror` | `2.0` |
| Logging | `tracing` + `tracing-subscriber` | `0.1` / `0.3` |
| Image processing | `image` | `0.25` |
| Channels | `crossbeam-channel` | `0.5` |
| CLI args | `clap` | `4.5` (derive) |
| Global hotkeys | `global-hotkey` | `0.7` (Tauri team, cross-platform: Linux/Windows/macOS) |
| Randomization | `rand` | `0.9` |
| Time | `chrono` | `0.4` |

---

## ONNX Model Pipeline

Four models from PINTO_model_zoo / OpenCV Zoo, pre-converted to ONNX:

1. **Palm Detection** -- input `[1,192,192,3]` float32 `[0,1]` -- outputs bounding boxes + anchors
2. **Hand Landmark** -- input `[1,224,224,3]` float32 `[0,1]` -- outputs `[1,63]` = 21 landmarks x 3
3. **Face Detection (BlazeFace)** -- input `[1,128,128,3]` float32 `[0,1]` -- outputs bounding boxes
4. **Face Mesh** -- input `[1,192,192,3]` float32 `[-1,1]` -- outputs `[1,1404]` = 468 landmarks x 3

Key landmark indices:
- Fingertips: 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky)
- Outer lips: `[61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146]`
- Inner lips: `[78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]`

Models are automatically downloaded on first run by `src/inference/model_downloader.rs` if not present at the configured paths. Downloads include SHA256 integrity verification.

---

## Project Structure

```
nailbite/
  src/
    main.rs                    # CLI args, config load, bootstrap
    lib.rs                     # Module declarations
    config.rs                  # YAML config struct + validation
    errors.rs                  # thiserror error types (scoped per subsystem)
    paths.rs                   # Shared path utilities (expand_tilde)
    app.rs                     # Application lifecycle orchestrator
    camera/
      mod.rs
      backend.rs               # CameraBackend trait (platform abstraction)
      v4l_backend.rs           # Linux V4L2 implementation (cfg(target_os = "linux"))
      frame.rs                 # Frame type + pixel format conversion
      pipeline.rs              # Per-camera inference pipeline
    inference/
      mod.rs
      session.rs               # ort::Session management + shared config
      model_downloader.rs      # Automatic model downloading with SHA256 verification
      preprocessing.rs         # Resize, normalize, tensor creation
      palm_detection.rs        # Palm detection model wrapper
      hand_landmark.rs         # Hand landmark model wrapper
      face_detection.rs        # BlazeFace model wrapper
      face_mesh.rs             # Face mesh model wrapper
      postprocessing.rs        # NMS, anchor decoding, coordinate mapping
    detection/
      mod.rs
      types.rs                 # Landmark, HandDetection, FaceDetection, BfrbType, DetectionEvent
      tracker.rs               # Temporal state machine (sliding window)
      analyzer.rs              # Spatial proximity + pose classification
      fusion.rs                # Multi-camera detection merging
      behaviors/
        mod.rs                 # BehaviorDetector trait
        nail_biting.rs         # Fingertip-near-mouth
        nail_picking.rs        # Hand-on-hand fingertip proximity
        hair_pulling.rs        # Hand near head with grip posture (stub)
        skin_picking.rs        # Pinching gesture near face (stub)
        lip_biting.rs          # Jaw/lip landmark analysis (stub)
    actions/
      mod.rs
      types.rs                 # Action trait
      sound.rs                 # rodio looping alert
      notification.rs          # notify-rust desktop notification
      webhook.rs               # HTTP POST with JSON payload
      popup.rs                 # Bridge to exercise UI
    exercises/
      mod.rs
      types.rs                 # Exercise trait, ExercisePhase, VerificationResult
      registry.rs              # Available exercises + selection strategy
      verification.rs          # Camera-based pose verification loop
      fist_clench.rs           # 60s hold, all fingertips collapsed to palm
      flat_hand_press.rs       # 60s hold, fingers spread on desk
      interlocked_squeeze.rs   # 60s hold, hands clasped together
      ear_touch.rs             # 10 reps, Moritz decoupling deflection
      finger_flick.rs          # 10 reps, finger extension away from body
      palm_press.rs            # 60s hold, palms pressed together
      fingertip_massage.rs     # 30s per hand, rubbing fingertips on palm
    ui/
      mod.rs
      tray.rs                  # System tray icon + context menu
      preview.rs               # egui camera preview with landmark overlays
      exercise_window.rs       # Exercise instructions + live verification feedback
    hotkeys/
      mod.rs
      listener.rs              # global-hotkey based capture (cross-platform)
    training/
      mod.rs
      collector.rs             # Save annotated frames + landmarks
      annotation.rs            # Annotation types (false_positive, false_negative, true_positive)
    stats/
      mod.rs
      session_log.rs           # JSONL logging of detections, exercises, annotations
  models/                      # ONNX model files (auto-downloaded)
  assets/
    sounds/alert.wav
    icons/tray_{normal,active,alert}.png
  tests/
    integration/
    fixtures/
  config.yaml                  # Default/example config
  Cargo.toml
  Dockerfile
  build.sh
```

---

## Cross-Platform Abstraction

Platform-specific code is isolated behind traits so that Windows/macOS support can be added without changing the core logic:

```rust
/// Camera backend abstraction (src/camera/backend.rs)
pub trait CameraBackend: Send {
    fn open(device: &str, width: u32, height: u32, fps: u32) -> Result<Self, CameraError> where Self: Sized;
    fn capture_frame(&mut self) -> Result<Frame, CameraError>;
    fn device_list() -> Result<Vec<CameraDevice>, CameraError> where Self: Sized;
}
```

| Abstraction | MVP (Linux) | Future Windows | Future macOS |
|---|---|---|---|
| Camera | `v4l` (V4L2) | Media Foundation / `nokhwa` | AVFoundation / `nokhwa` |
| Hotkeys | `global-hotkey` | `global-hotkey` | `global-hotkey` |
| Tray icon | `tray-icon` | `tray-icon` | `tray-icon` |
| Audio | `rodio` (ALSA) | `rodio` (WASAPI) | `rodio` (CoreAudio) |
| Notifications | `notify-rust` (D-Bus) | `notify-rust` (WinRT) | `notify-rust` (macOS) |
| GUI | `eframe`/`egui` | `eframe`/`egui` | `eframe`/`egui` |

Most crates (`tray-icon`, `rodio`, `eframe`, `global-hotkey`, `notify-rust`) are already cross-platform. Only the camera backend needs platform-specific implementations. The `v4l_backend.rs` file is gated with `#[cfg(target_os = "linux")]`.

---

## Core Traits

### BehaviorDetector

```rust
pub trait BehaviorDetector: Send + Sync {
    fn bfrb_type(&self) -> BfrbType;
    fn name(&self) -> &str;
    fn analyze_frame(&self, analysis: &FrameAnalysis) -> Option<f32>;  // confidence [0,1]
    fn min_sustained_duration(&self) -> Duration;
    fn confidence_threshold(&self) -> f32;
    fn requires_face(&self) -> bool;
    fn requires_hands(&self) -> bool;
}
```

New BFRBs are added by implementing this trait. The detection thread holds a `Vec<Box<dyn BehaviorDetector>>` and runs all enabled detectors per frame.

### Exercise

```rust
pub trait Exercise: Send + Sync {
    fn id(&self) -> &str;
    fn name(&self) -> &str;
    fn instructions(&self) -> &str;
    fn category(&self) -> ExerciseCategory;  // TimedHold | Repetitions
    fn hold_duration(&self) -> Duration;
    fn target_reps(&self) -> u32;
    fn applicable_to(&self) -> &[BfrbType];
    fn verify(&self, hands: &[HandDetection], face: &Option<FaceDetection>) -> VerificationResult;
    fn timeout(&self) -> Duration;
}
```

### Action

```rust
pub trait Action: Send + Sync {
    fn start(&mut self, event: &DetectionEvent) -> Result<(), ActionError>;
    fn stop(&mut self) -> Result<(), ActionError>;
    fn is_active(&self) -> bool;
}
```

---

## Detection Logic

### Nail Biting
1. Require 1+ hands + face with mouth landmarks
2. Compute distance from each fingertip to mouth center (average of inner lip landmarks)
3. Normalize by face width (landmarks 234-454)
4. Positive if any fingertip within `proximity_threshold` (default 0.35x face width)
5. Hand pose filter: fingers must be partially curled (not flat/spread)
6. Suppress if typing posture detected (both hands same Y level, fingers extended down)
7. Suppress if chin rest detected (wrist/palm near chin, not fingertips at mouth)

### Nail Picking
1. Require 2 hands detected
2. Compute fingertip-to-fingertip distances between hands
3. Check for pinching gesture (thumb+index close on one hand)
4. Distinguish from clasped hands (all fingers interleaved) and typing (hands side-by-side but on keyboard plane)

### Temporal Tracker
- Sliding window (default 1.5s) of per-frame confidence values
- Requires 40% of frames above threshold to confirm (configurable, code default 50%)
- State machine: `Idle -> Accumulating -> Confirmed -> Alerting -> Cooldown`
- Cooldown (default 30s) prevents immediate re-trigger after exercise

---

## Hotkeys & Training Data Collection

### Global Hotkey Listener (`src/hotkeys/listener.rs`)
Uses `global-hotkey` crate (Tauri team, cross-platform). Configurable keys in config.yaml.

**Hotkeys:**
- **Dismiss / False Positive** (default: `F9`): Stops all active alerts, marks current detection as false positive
- **Mark Missed Event** (default: `F10`): User flags that a BFRB is happening but was not detected (false negative)
- **Pause/Resume** (default: `F11`): Temporarily pause/resume detection

### Training Data Collector (`src/training/collector.rs`)

When a hotkey is pressed or an event is confirmed, the system saves an annotation:

```rust
pub struct TrainingAnnotation {
    pub timestamp: DateTime<Utc>,
    pub annotation_type: AnnotationType,
    pub bfrb_type: Option<BfrbType>,
    /// Raw frame data (if save_frames enabled)
    pub frame_paths: Vec<PathBuf>,
    /// Detection confidence at time of annotation
    pub detection_confidence: Option<f32>,
}

pub enum AnnotationType {
    TruePositive,    // Detection confirmed by exercise completion
    FalsePositive,   // User dismissed via hotkey
    FalseNegative,   // User flagged missed event via hotkey
}
```

**Storage:**
- Annotations saved as JSONL to `~/.local/share/nailbite/annotations.jsonl`
- If `training.save_frames: true` in config, raw frames are saved as JPEG to `~/.local/share/nailbite/frames/` (with annotation reference)
- Landmark data is always saved (small, just numbers)
- This dataset can be used to train a lightweight classifier on top of the landmark features (post-MVP: replace rule-based detection with ML classifier)

---

## Decoupling Exercises (Evidence-Based)

Based on Habit Reversal Training (Azrin & Nunn, 1973) and the Moritz Decoupling Protocol (UKE Hamburg). Reference: tricks-gegen-ticks.de.

| Exercise | Type | Duration/Reps | Verification | For |
|---|---|---|---|---|
| Fist Clench | Hold | 60s | All fingertips collapsed to palm | All BFRBs |
| Flat Hand Press | Hold | 60s | Fingers spread, hand flat | All BFRBs |
| Interlocked Squeeze | Hold | 60s | Both hands clasped, fingers interleaved | All BFRBs |
| Ear Touch Deflection | Reps | 10x | Fingertip reaches ear landmark | Nail biting |
| Finger Extension Flick | Reps | 10x per hand | Rapid curl-to-extend transition | Nail picking |
| Palm-to-Palm Press | Hold | 60s | Both palms aligned at midline | All BFRBs |
| Fingertip-on-Palm Massage | Hold | 30s/hand | Fingertips on opposite palm, motion detected | Nail picking |

Selection strategy configurable: random, first, round_robin, preferred.

Exercises must be held/completed to stop the alert. Minimum therapeutic duration is 60s (per Twohig & Woods, 2001). Compliance ratio (fraction of time pose must be correct) defaults to 80%.

---

## Multi-Camera Support (Post-MVP, interfaces created now)

Each camera in config gets its own capture+inference thread with a configurable `role`:
- `full`: Runs face + hand pipeline (default, single camera)
- `face_only`: Only face detection + mesh
- `hands_only`: Only palm detection + hand landmarks

**Fusion strategies** (configurable):
- `any`: Each camera runs independently; any camera detecting BFRB triggers alert (OR logic)
- `merge`: Combine face data from face-role camera with hand data from hands-role camera into a single `FrameAnalysis`. Requires time synchronization (tolerance configurable, default 100ms)

---

## Performance Strategy

| Technique | Impact |
|---|---|
| Inference FPS cap (default 8) | Runs models on every Nth frame, not all 30 |
| Alternating model scheduling | Even frames: hand pipeline, odd frames: face pipeline |
| ROI tracking (post-MVP) | After initial detection, skip palm/face detection; reuse tracked bounding box |
| ONNX thread limits | `intra_op_num_threads: 2`, `inter_op_num_threads: 1` |
| V4L2 MMAP buffers | Zero-copy frame access from kernel |
| Bounded channels (size 2) | Old frames dropped under backpressure |
| Screen lock detection (post-MVP) | Pause pipelines via D-Bus ScreenSaver interface |
| Tensor reuse | Pre-allocate input/output tensors |

---

## Configuration (Single YAML File)

```yaml
general:
  log_level: info
  show_preview: false
  cooldown_seconds: 30
  stats_file: ~/.local/share/nailbite/stats.jsonl

models:
  palm_detection: ./models/palm_detection.onnx
  hand_landmark: ./models/hand_landmark.onnx
  face_detection: ./models/face_detection.onnx
  face_mesh: ./models/face_mesh.onnx

cameras:
  - id: main
    device: /dev/video0
    role: full
    width: 640
    height: 480
    fps: 30
    inference_fps: 8

ort:
  intra_op_num_threads: 2
  inter_op_num_threads: 1

detection:
  behaviors:
    nail_biting:
      enabled: true
      proximity_threshold: 0.35
      min_sustained_ms: 1500
      confidence_threshold: 0.3
    nail_picking:
      enabled: true
      proximity_threshold: 0.15
      min_sustained_ms: 1500
      confidence_threshold: 0.3
    hair_pulling:
      enabled: false
    skin_picking:
      enabled: false
    lip_biting:
      enabled: false
  temporal:
    window_ms: 1500
    positive_ratio: 0.4
  false_positive:
    typing_suppression: true
    chin_rest_suppression: true
    eating_suppression: true

fusion:
  strategy: any
  merge_tolerance_ms: 100

actions:
  sound:
    enabled: true
    file: builtin
    volume: 0.8
    repeat: true
  notification:
    enabled: true
  webhook:
    enabled: false
    url: ""
    timeout_ms: 5000
    headers: {}  # Optional custom HTTP headers
  popup:
    enabled: true

exercises:
  selection_strategy: random
  preferred_exercise: null
  hold_duration_override: null
  reps_override: null
  timeout_seconds: 120
  compliance_ratio: 0.8

hotkeys:
  dismiss_false_positive: F9
  mark_missed_event: F10
  pause_resume: F11

training:
  save_frames: false
  save_landmarks: true
  annotations_file: ~/.local/share/nailbite/annotations.jsonl
  frames_dir: ~/.local/share/nailbite/frames/
```

---

## Build & Deployment

- **Dockerfile**: `rust:1.85.0-bookworm` + cargo-chef + sccache + upx --best --lzma, final stage from scratch
- **build.sh**: Docker buildx bake, extract binary from container
- **GitHub Actions**: Test + clippy + cargo audit on push, Docker build on tag
- **Release profile**: `lto = true`, `strip = "symbols"`, clippy lints enforced

---

## Implementation Phases

### MVP (Single Camera, Nail Biting + Nail Picking)

**Phase 1: Foundation**
1. Project scaffolding (Cargo.toml, module declarations, CLAUDE.md)
2. Error types (thiserror, scoped per subsystem)
3. Config struct + YAML loading + validation + tests
4. Tracing setup (env-filter)
5. Model downloader (automatic download with SHA256 verification)

**Phase 2: Camera + Inference**
6. `CameraBackend` trait + `V4lBackend` implementation (single camera)
7. Frame type + YUYV/MJPEG -> RGB conversion
8. ONNX session management with `ort`
9. Preprocessing (resize, normalize for both `[0,1]` and `[-1,1]` ranges)
10. Palm detection model wrapper + anchor decoding + NMS
11. Hand landmark model wrapper + coordinate mapping
12. Face detection (BlazeFace) model wrapper
13. Face mesh model wrapper
14. Camera pipeline: capture -> preprocess -> infer -> assemble `FrameAnalysis`

**Phase 3: Detection**
15. `BehaviorDetector` trait definition
16. `NailBitingDetector` (fingertip-to-mouth proximity + hand pose filter)
17. `NailPickingDetector` (hand-to-hand fingertip proximity + pinch gesture)
18. False positive suppression (typing posture, chin rest)
19. Temporal tracker (sliding window state machine)
20. Detection loop: receive `FrameAnalysis` -> run detectors -> emit `DetectionEvent`

**Phase 4: Actions + Exercises**
21. `Action` trait + sound action (rodio, looping alert)
22. Notification action (notify-rust)
23. Webhook action (reqwest blocking POST)
24. `Exercise` trait + `ExerciseRegistry` + selection strategies
25. Fist clench exercise with landmark-based verification
26. Flat hand press exercise with verification
27. Remaining 5 exercises (interlocked squeeze, ear touch, finger flick, palm press, fingertip massage)
28. Exercise session manager (phase tracking, compliance ratio, timeout)
29. Action orchestrator (start all on detection, stop all on exercise completion)

**Phase 5: UI + Hotkeys + Training**
30. System tray (tray-icon) with context menu (pause/resume, show preview, quit)
31. Camera preview window (egui, toggle-able, landmark overlays)
32. Exercise guidance window (egui, instructions + live verification feedback)
33. Global hotkey listener (global-hotkey): dismiss/false-positive, mark missed, pause
34. Training data collector (JSONL annotations + optional frame saving)

**Phase 6: Build + CI**
35. Dockerfile (rust:bookworm + cargo-chef + upx + scratch)
36. build.sh (docker buildx bake, extract binary)
37. GitHub Actions (test + clippy + cargo audit on push, Docker build on tag)

### Post-MVP (Future Work)

- Multi-camera support (camera pipeline per device, detection fusion)
- Additional BFRBs (hair pulling, skin picking, lip biting detectors)
- Settings window (egui config editor)
- ROI tracking + alternating model scheduling (performance)
- Screen lock idle detection (D-Bus)
- Windows + macOS camera backends
- ML classifier trained on collected annotation data (replace rule-based detection)

---

## Verification

After each phase, verify by:
- `cargo test` -- all unit + integration tests pass
- `cargo clippy -- -D warnings` -- no warnings
- Phase 2+: run with `--show-preview` to visually confirm landmark detection on live camera feed
- Phase 3+: trigger detection by bringing hand to mouth, verify console log output
- Phase 4+: verify sound plays and exercise window opens on detection
- Phase 5+: verify hotkeys dismiss alerts and annotations are written to JSONL
- Phase 6: `bash build.sh` produces a working static binary
