# Nailbite

**A quiet companion that notices when your hands wander to your face — and gently lets you know.**

If you've ever caught yourself mid-bite and thought "oh, again" — Nailbite is built for that
moment. It watches your webcam, recognises a small set of body-focused repetitive behaviours
(BFRBs) — currently nail biting and nail picking — and surfaces a soft alert when it sees one.
That's it. No streaks, no scolding, no leaderboards. You decide what to do with the nudge.

Everything happens on your own machine. Nothing is uploaded.

## Why this exists

BFRBs are common, often soothing, and usually not a moral failing — they're just hard to
*notice* in the moment. The hand is at the mouth before the conscious mind catches up. A
gentle external signal can close that gap, which is the whole idea behind the [habit-reversal
training](https://en.wikipedia.org/wiki/Habit_reversal_training) and decoupling traditions
this tool draws from.

Nailbite is a personal experiment, not a medical device. If you're working with a therapist on
a BFRB, this can be a useful pair of extra eyes — but it isn't a substitute for that work.

## What it does

- **Notices the gesture.** Hand/face/pose tracking with on-device ONNX models, running at a
  modest ~8 FPS so it's friendly to your CPU.
- **Waits for sustained contact** before alerting (a brief brush of the lip won't trigger it).
- **Plays a soft sound and a desktop notification** so you can choose to redirect the gesture.
- **Lets you label each alert** "correct" or "false positive" with one click, so the thresholds
  can be tuned to *your* hands over time.
- **Keeps a short clip and the landmark trace** of each event in `~/.local/share/nailbite/` so
  you can review what happened if you want to — and delete it if you don't.
- **Stays quiet when you're not in frame.** A multi-modal presence check (face mesh + torso
  pose) gates the whole pipeline, so an empty chair never sets off the detector.
- **Runs entirely locally.** No telemetry, no cloud, no account. The only network feature is
  an optional webhook that ships off by default.

## What it doesn't do (yet)

- Hair pulling, skin picking, lip biting — these have detector stubs but aren't enabled. The
  generic pipeline is there; per-behaviour signals need tuning before they're trustworthy.
- macOS / Windows. Linux x86_64 only at the moment; the camera and tray code are behind
  platform traits so the door is open.
- "Coaching" or gamification. By design — a counter that nudges you toward a number can turn
  into another source of shame, which is the opposite of what helps.

## Screenshot

<p align="center">
  <img src="docs/images/hero.png" alt="Nailbite live preview with hand, face mesh, and pose torso landmarks overlaid; tray status reads 'Monitoring'" width="780">
</p>

The live preview during monitoring. Hand landmarks (green = left, orange = right)
ride on top of the face mesh and pose torso so you can see exactly what the
detector is seeing — useful when you're tuning thresholds, and reassuring when
you want to know what is and isn't being tracked.

## Install

Pre-built binaries are on the [Releases page](../../releases/latest) for every tagged
version. The release notes include copy-paste install commands with exact URLs.

Pick whichever format you prefer; both contain the same app.

### AppImage (works on most distros, no install needed)

Download the AppImage for your hardware, then:

```bash
chmod +x nailbite_*_amd64-cpu.AppImage
./nailbite_*_amd64-cpu.AppImage
```

On **NixOS**:

```bash
nix-shell -p appimage-run --run './nailbite_*_amd64-cpu.AppImage'
```

On **Ubuntu 24.04** the AppImage needs the legacy FUSE shim:

```bash
sudo apt install libfuse2t64
```

### Debian / Ubuntu (.deb)

```bash
sudo apt install ./nailbite_*_amd64-cpu.deb
```

### GPU builds

CPU is fine for most people — the detection pipeline is light. If you have a discrete GPU and
want the headroom, replace `cpu` in the filename with the backend matching your hardware:

| Backend | Filename suffix | Runtime requirement              |
|---------|-----------------|----------------------------------|
| CPU     | `-cpu`          | none                             |
| NVIDIA CUDA     | `-cuda`     | NVIDIA driver 525+, CUDA 12.x    |
| NVIDIA TensorRT | `-tensorrt` | NVIDIA driver 525+, CUDA 12.x    |
| AMD ROCm        | `-rocm`     | AMD ROCm 6.0+                    |
| AMD MIGraphX    | `-migraphx` | AMD ROCm 6.0+                    |

### Verifying downloads

Every release ships a `SHA256SUMS.txt`. Drop it next to the binary you downloaded and run:

```bash
sha256sum -c SHA256SUMS.txt --ignore-missing
```

### Requirements

- Linux x86_64 (NixOS, Arch, Debian 12+, Ubuntu 22.04+ tested)
- A V4L2-compatible webcam (most USB and laptop cameras work out of the box)
- Membership of the `video` group (most distros add the primary user automatically)

## Using it

The first launch downloads a small set of ONNX models (~50 MB) and caches them locally. After
that, Nailbite lives in your system tray:

- 🟢 monitoring
- 🟡 paused
- 🔴 BFRB currently detected
- ⚫ no one in frame
- ⚪ offline

A few keys you'll probably want:

| Key | Action |
|-----|--------|
| F9  | Dismiss the latest alert as a false positive (helps the detector learn) |
| F10 | Mark a missed event (false negative) |
| F11 | Pause / resume detection |

When an alert fires you get an inline "Correct / False positive" choice on the notification
itself. The history page lets you scrub through past events with the landmarks overlaid, and
the insights page sweeps thresholds against your labels to suggest tuning.

## Configuration

The single source of truth is `config.yaml`. The defaults are tuned to be conservative
(prefers missing a real event to firing a false one), and the camera bias is set up so your
face stays well-lit even with a bright window behind you.

A small taste:

```yaml
camera:
  inference_fps: 8
  controls:
    auto_exposure: true
    exposure_auto_priority: false   # let exposure stretch so dim rooms don't dim you
    backlight_compensation_max: true
    brightness_fraction: 0.60       # null disables this bias
    contrast_fraction: 0.55
  sources:
    - id: main
      device: /dev/video0
      role: primary

detection:
  behaviors:
    nail_biting:
      enabled: true
      proximity_threshold: 0.35
      min_sustained_ms: 4000        # 4s of dwell before alerting

ort:
  gpu:
    preference: auto                # auto / disabled / required
```

See `docs/configuration.md` for the full reference.

## How it works under the hood

1. **Camera** → frames pulled from V4L2.
2. **AI** → a small stack of ONNX models (palm detection + hand landmarks, face detection +
   mesh, pose landmarks) runs each frame.
3. **Presence gate** → face mesh and pose torso landmarks both have to agree "user in frame"
   before anything alerts.
4. **Behaviour analysis** → temporal smoothing + per-hand explanations decide whether the
   geometry actually looks like nail biting or picking, rather than typing, eating, scratching
   an itch, or resting on a chin.
5. **Alert + label** → soft sound, desktop notification with inline labels, snapshot saved to
   your local history.

More detail in `docs/architecture.md` and `docs/detection.md`.

## Privacy

This is software that points a camera at you. Privacy isn't a bullet point — it's the whole
shape of the thing.

- No video frames or images ever leave your machine.
- No cloud, no account, no telemetry.
- Event clips and stats live under `~/.local/share/nailbite/`. They are yours; delete them
  whenever you like.
- The only optional network feature is a webhook (disabled by default) so you can route alerts
  into your own systems if you want.

## Build from source

Required: Rust 1.88+, Node.js 22+, pnpm 10+.

System packages (Debian / Ubuntu):

```bash
sudo apt install \
    libwebkit2gtk-4.1-dev libgtk-3-dev libglib2.0-dev \
    libayatana-appindicator3-dev librsvg2-dev libsoup-3.0-dev \
    libjavascriptcoregtk-4.1-dev libasound2-dev libxdo-dev \
    libv4l-dev libssl-dev cmake clang
```

On NixOS, `nix-shell` handles all of this for you.

```bash
git clone https://github.com/firstdorsal/nailbite.git
cd nailbite
pnpm install
pnpm tauri dev
```

### Reproducible release builds (Docker)

The release pipeline uses these scripts; running them locally produces byte-identical
artifacts to the published ones.

```bash
bash build.sh                       # CPU-only AppImage + .deb in ./dist
GPU_BACKEND=cuda bash build.sh
GPU_BACKEND=tensorrt bash build.sh
GPU_BACKEND=rocm bash build.sh
GPU_BACKEND=migraphx bash build.sh

bash scripts/run-appimage.sh        # build + launch in one step
```

## Development

```bash
pnpm test                       # Frontend tests (vitest)
cd src-tauri && cargo test      # Backend tests
pnpm lint && pnpm typecheck     # Frontend checks
cd src-tauri && cargo clippy    # Backend lint
```

## Contributing

This is a personal project, but issues and small PRs are welcome — especially:

- Reports of false positives / negatives with the event clip attached (it's already saved
  locally, no extra work).
- New behaviour detectors implementing the `BehaviorDetector` trait.
- macOS or Windows camera backends behind the existing platform traits.

If you're trying it out and something feels wrong (it scolds you too much, the sound is too
loud, the alert dwell is too short), that's exactly the feedback that helps the defaults get
kinder over time.

## License

GNU Affero General Public License v3.0 — see `LICENSE`.

In short: you're free to use, modify, and redistribute Nailbite. If you run a
modified version as a network service, you have to make your source available
to its users too. The aim is to keep the tool — and any improvements made to
it — open for the community that depends on it.

## Acknowledgments

- Hand landmark + palm detection models from [OpenCV Zoo](https://github.com/opencv/opencv_zoo).
- Face detection + mesh from [IntelliProve](https://github.com/IntelliProve/face-detection-onnx).
- Pose landmark from [BlazePose via Unity Inference Engine](https://huggingface.co/unity/inference-engine-blaze-pose).
- Built on the shoulders of [Tauri](https://tauri.app/), [React](https://react.dev/), and
  [ONNX Runtime](https://onnxruntime.ai/).

And a quiet thank-you to anyone whose research on habit reversal and decoupling made this
worth building in the first place.
