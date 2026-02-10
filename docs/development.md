# Development Guide

## Prerequisites

### NixOS

Use the provided `shell.nix`:

```bash
nix-shell
```

This provides all dependencies including:
- Rust 1.85+
- Node.js 22+
- pnpm
- GTK3, WebKitGTK
- ONNX Runtime
- V4L2 utilities

### Other Linux Distributions

Install system dependencies:

```bash
# Debian/Ubuntu
sudo apt install \
  libwebkit2gtk-4.1-dev \
  libgtk-3-dev \
  libayatana-appindicator3-dev \
  librsvg2-dev \
  libv4l-dev \
  libasound2-dev \
  pkg-config \
  curl \
  wget

# Fedora
sudo dnf install \
  webkit2gtk4.1-devel \
  gtk3-devel \
  libappindicator-gtk3-devel \
  librsvg2-devel \
  libv4l-devel \
  alsa-lib-devel
```

Install Rust and Node.js:

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default 1.85

# Node.js (via nvm)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 22
nvm use 22

# pnpm
npm install -g pnpm
```

## Project Setup

```bash
# Clone repository
git clone https://github.com/your-username/nailbite.git
cd nailbite

# Install frontend dependencies
pnpm install

# Download ONNX models (happens automatically on first run)
# Or manually:
mkdir -p models
# Models are downloaded from opencv_zoo and HuggingFace
```

## Development Commands

```bash
# Start development server (frontend + backend)
pnpm tauri dev

# Run frontend only (no Rust backend)
pnpm dev

# Run tests
pnpm test              # Frontend tests
pnpm test:watch        # Watch mode
cd src-tauri && cargo test  # Backend tests

# Linting
pnpm lint              # ESLint
pnpm typecheck         # TypeScript
cd src-tauri && cargo clippy  # Rust

# Full verification
pnpm verify            # All checks
```

## Project Structure

```
nailbite/
├── src/                    # Frontend (React + TypeScript)
│   ├── components/         # UI components
│   ├── hooks/              # React hooks
│   ├── pages/              # Page components
│   ├── lib/                # Utilities
│   ├── types/              # TypeScript types
│   └── test/               # Test setup
├── src-tauri/              # Backend (Rust)
│   ├── src/
│   │   ├── actions/        # Alert actions
│   │   ├── camera/         # V4L2 capture
│   │   ├── commands/       # Tauri IPC commands
│   │   ├── detection/      # BFRB detection
│   │   ├── exercises/      # Decoupling exercises
│   │   ├── inference/      # ONNX models
│   │   ├── stats/          # Session logging
│   │   └── training/       # Training data collection
│   ├── Cargo.toml
│   └── tauri.conf.json
├── docs/                   # Documentation
├── models/                 # ONNX models (gitignored)
├── config.yaml             # Configuration
└── package.json
```

## Adding a New BFRB Detector

1. Create detector in `src-tauri/src/detection/behaviors/`:

```rust
// src-tauri/src/detection/behaviors/hair_pulling.rs
use super::{BehaviorDetector, DetectionContext};
use crate::detection::types::{AnalysisResult, BfrbType};

pub struct HairPullingDetector {
    proximity_threshold: f32,
}

impl HairPullingDetector {
    pub fn new(proximity_threshold: f32) -> Self {
        Self { proximity_threshold }
    }
}

impl BehaviorDetector for HairPullingDetector {
    fn bfrb_type(&self) -> BfrbType {
        BfrbType::HairPulling
    }

    fn analyze(&self, ctx: &DetectionContext) -> Option<AnalysisResult> {
        // Detection logic here
        None
    }
}
```

2. Register in `src-tauri/src/detection/behaviors/mod.rs`

3. Add configuration in `src-tauri/src/config.rs`

4. Add tests

## Adding a New Exercise

1. Create exercise in `src-tauri/src/exercises/`:

```rust
// src-tauri/src/exercises/new_exercise.rs
use super::types::{Exercise, ExerciseCategory, VerificationContext};
use crate::detection::types::BfrbType;

pub fn new_exercise() -> Exercise {
    Exercise {
        id: "new_exercise".to_string(),
        name: "New Exercise".to_string(),
        instructions: "Instructions here".to_string(),
        category: ExerciseCategory::TimedHold,
        hold_duration_secs: 30,
        target_reps: 0,
        applicable_to: vec![BfrbType::NailBiting],
        verify: Box::new(verify_new_exercise),
    }
}

fn verify_new_exercise(ctx: &VerificationContext) -> (bool, String) {
    // Verification logic
    (true, "Good!".to_string())
}
```

2. Register in `src-tauri/src/exercises/registry.rs`

3. Add tests

## Debugging

### Enable Debug Logging

```bash
RUST_LOG=debug pnpm tauri dev
```

### Camera Issues

```bash
# List cameras
v4l2-ctl --list-devices

# Test camera
v4l2-ctl -d /dev/video0 --list-formats-ext
```

### Model Issues

Models are downloaded to `./models/`. Delete and restart to re-download:

```bash
rm -rf models/
pnpm tauri dev
```

## Testing

### Frontend Tests

```bash
pnpm test                    # Run once
pnpm test:watch              # Watch mode
pnpm test:coverage           # With coverage
```

Tests use Vitest + React Testing Library. Test files are colocated with source (`*.test.ts`).

### Backend Tests

```bash
cd src-tauri
cargo test                   # All tests
cargo test detection         # Filter by name
cargo test -- --nocapture    # Show output
```

## Building for Production

```bash
# Standard build
pnpm tauri build

# Debug build (faster, includes debug symbols)
pnpm tauri build --debug

# With GPU support
GPU_BACKEND=cuda bash build.sh
GPU_BACKEND=rocm bash build.sh
```

Output is in `src-tauri/target/release/bundle/`.

## Code Style

### Rust

- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Use `thiserror` for error types
- Use `tracing` for logging

### TypeScript

- Use ESLint configuration
- Use Prettier for formatting
- Prefer functional components with hooks
- Use TypeScript strict mode

## Commit Guidelines

- Use conventional commits
- Sign commits with GPG
- Run `pnpm verify` before committing
