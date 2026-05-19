# Configuration Reference

n**AI**lbite is configured via `config.yaml` in the project root.

## Complete Configuration

```yaml
general:
  log_level: info              # trace, debug, info, warn, error
  show_preview: false          # Show camera preview window on startup
  cooldown_seconds: 30         # Cooldown after exercise before re-triggering
  stats_file: ~/.local/share/nailbite/stats.jsonl

models:
  palm_detection: ./models/palm_detection.onnx
  hand_landmark: ./models/hand_landmark.onnx
  face_detection: ./models/face_detection.onnx
  face_mesh: ./models/face_mesh.onnx
  pose_detection: ./models/pose_detection.onnx
  pose_landmark: ./models/pose_landmark.onnx

camera:
  inference_fps: 8             # Frames per second for inference
  preview_fps: 24              # Live preview FPS (must be >= inference_fps)
  controls:                    # Camera biasing applied at capture start
    gamma_reset: true
    auto_exposure: true
    exposure_auto_priority: true    # Image priority — exposure may stretch in dim light
    auto_white_balance: true
    auto_gain: true
    backlight_compensation_max: false   # Opt-in for backlit scenes only
    brightness_fraction: null           # Fraction of [min,max], or null (default) for camera default
    contrast_fraction: null
  sources:
    - id: main
      device: /dev/video0
      role: primary            # primary or auxiliary
      resolution_width: 640
      resolution_height: 480

ort:
  intra_op_num_threads: 2      # Threads within an operation
  inter_op_num_threads: 1      # Threads between operations
  gpu:
    preference: auto           # disabled, auto, preferred, required
    backend: auto              # auto, cuda, tensorrt, rocm, migraphx
    device_id: 0
    fp16_enable: true
    memory_limit_mb: null      # null for unlimited

detection:
  behaviors:
    nail_biting:
      enabled: true
      proximity_threshold: 0.35  # Max distance from fingertip to mouth
      min_sustained_ms: 1500     # How long behavior must persist
      confidence_threshold: 0.3
    nail_picking:
      enabled: true
      proximity_threshold: 0.25
      min_sustained_ms: 1500
      confidence_threshold: 0.3
    hair_pulling:
      enabled: false
    skin_picking:
      enabled: false
    lip_biting:
      enabled: false
  temporal:
    window_ms: 1500            # Sliding window for detection
    positive_ratio: 0.4        # Fraction of positive frames needed
  false_positive:
    typing_suppression: true   # Suppress when typing detected
    chin_rest_suppression: true
    eating_suppression: true

fusion:
  strategy: any                # any, all, merge
  merge_tolerance_ms: 100

actions:
  sound:
    enabled: true
    file: builtin              # builtin or path to WAV file
    volume: 0.8                # 0.0 to 1.0
    repeat: true               # Loop until dismissed
  webhook:
    enabled: false
    url: ""
    timeout_ms: 5000
    headers: {}                # Custom headers

exercises:
  selection_strategy: random   # random, round_robin, preferred
  preferred_exercise: null     # Exercise ID if strategy is preferred
  hold_duration_override: null # Override timed hold duration
  reps_override: null          # Override rep count
  timeout_seconds: 120         # Max time for exercise
  compliance_ratio: 0.8        # Required pose accuracy

hotkeys:
  dismiss_false_positive: F9
  mark_missed_event: F10
  pause_resume: F11

training:
  save_frames: false           # Save frames for training data
  save_landmarks: true         # Save landmarks for training data
  annotations_file: ~/.local/share/nailbite/annotations.jsonl
  frames_dir: ~/.local/share/nailbite/frames/
```

## Section Details

### general

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| log_level | string | info | Logging verbosity |
| show_preview | bool | false | Open preview window on start |
| cooldown_seconds | int | 30 | Seconds between alerts |
| stats_file | string | ~/.local/share/nailbite/stats.jsonl | Session log path |

### camera

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| inference_fps | int | 8 | Detection rate (higher = more CPU) |
| preview_fps | int | 24 | Live preview FPS — must be `>= inference_fps` |
| controls | object | see below | Subject-friendly camera biasing applied at capture start |
| sources | array | - | Camera configurations |

#### camera.controls

Applied best-effort: unsupported V4L2 controls on a given camera are skipped, not fatal.
Defaults are minimal — just enable the camera's own auto-features and reset gamma to
the driver default. The lift biases (max backlight compensation, brightness/contrast
pushes) are opt-in because on normally-lit scenes they over-expose the image.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| gamma_reset | bool | true | Write gamma back to driver default at startup. Other apps (browsers, conferencing tools) often leave gamma cranked up; auto-exposure can't recover from that. |
| auto_exposure | bool | true | Let the driver track room brightness |
| exposure_auto_priority | bool | true | V4L2 image-priority mode: when `true`, exposure may stretch in dim or backlit scenes (FPS may drop briefly); when `false`, exposure is capped to preserve FPS and the subject crushes to black in dim light |
| auto_white_balance | bool | true | Track ambient colour temperature |
| auto_gain | bool | true | Track ambient gain |
| backlight_compensation_max | bool | false | Opt-in: pin backlight compensation to MAX. Only useful when the subject is genuinely backlit (bright window behind them); on a normally-lit scene this over-exposes. |
| brightness_fraction | float or null | null | Manual override. Position within the control's `[min,max]` range, rounded to its step. `null` keeps the camera's own default — usually the right choice. |
| contrast_fraction | float or null | null | Same semantics as `brightness_fraction` |

#### camera.sources

Each camera source:

| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique identifier |
| device | string | V4L2 device path |
| role | string | `primary` or `auxiliary` |
| resolution_width | int | Capture width |
| resolution_height | int | Capture height |

### ort (ONNX Runtime)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| intra_op_num_threads | int | 2 | Parallelism within ops |
| inter_op_num_threads | int | 1 | Parallelism between ops |

GPU settings:
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| preference | string | auto | GPU usage preference |
| backend | string | auto | GPU backend to use |
| device_id | int | 0 | GPU device index |
| fp16_enable | bool | true | Use FP16 for speed |
| memory_limit_mb | int/null | null | GPU memory limit |

GPU preference values:
- `disabled` - CPU only
- `auto` - Use GPU if available, fallback to CPU
- `preferred` - Prefer GPU, warn if unavailable
- `required` - Fail if GPU unavailable

### detection.behaviors

Each behavior can be configured:
| Field | Type | Description |
|-------|------|-------------|
| enabled | bool | Enable detection |
| proximity_threshold | float | Max normalized distance |
| min_sustained_ms | int | Required duration |
| confidence_threshold | float | Minimum confidence |

### actions.sound

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| enabled | bool | true | Enable sound alerts |
| file | string | builtin | `builtin` or WAV path |
| volume | float | 0.8 | Volume level (0-1) |
| repeat | bool | true | Loop until dismissed |

### actions.webhook

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| enabled | bool | false | Enable webhooks |
| url | string | "" | Webhook URL (HTTPS required) |
| timeout_ms | int | 5000 | Request timeout |
| headers | object | {} | Custom HTTP headers |

### exercises

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| selection_strategy | string | random | How to pick exercises |
| preferred_exercise | string/null | null | ID for preferred strategy |
| timeout_seconds | int | 120 | Max exercise duration |
| compliance_ratio | float | 0.8 | Required accuracy |

Selection strategies:
- `random` - Random applicable exercise
- `round_robin` - Cycle through exercises
- `preferred` - Always use preferred_exercise

## Hot Reload

Configuration changes are applied immediately without restart for:
- Detection thresholds
- Action settings
- Exercise settings

Camera and model changes require app restart.
