ARG PROFILE="release"
ARG SERVICE_NAME="nailbite"
ARG SERVICE_VERSION="0.1.0"
ARG BINARY_NAME="nailbite"

# --- Chef builder: Rust + system deps + cargo-chef + sccache ---
# Pin the Rust image version for reproducible builds.
FROM rust:1.85.0-bookworm AS chef-builder
RUN cargo install cargo-chef --locked
RUN cargo install --locked --no-default-features sccache
ENV RUSTC_WRAPPER=sccache SCCACHE_DIR=/sccache
# Disable incremental compilation for reproducible release builds.
ENV CARGO_INCREMENTAL=0
# Reproducible timestamps: Unix epoch for deterministic builds.
ENV SOURCE_DATE_EPOCH=0

# Install all system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    cmake \
    clang \
    libclang-dev \
    upx \
    # Audio (rodio -> ALSA)
    libasound2-dev \
    # System tray (tray-icon -> GTK3 + AppIndicator)
    libgtk-3-dev \
    libglib2.0-dev \
    libayatana-appindicator3-dev \
    # GUI (eframe/egui -> X11 + Wayland + OpenGL)
    libwayland-dev \
    libxkbcommon-dev \
    libx11-dev \
    libxcursor-dev \
    libxrandr-dev \
    libxi-dev \
    libxcb1-dev \
    libgl-dev \
    libvulkan-dev \
    # Global hotkeys (global-hotkey -> libxdo)
    libxdo-dev \
    # Camera (v4l)
    libv4l-dev \
    linux-libc-dev \
    # TLS
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*


# --- Planner: prepare cargo-chef recipe ---
FROM chef-builder AS planner
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src src
RUN cargo chef prepare --recipe-path recipe.json


# --- Builder: cook deps + build binary ---
FROM chef-builder AS builder
ARG PROFILE
ARG BINARY_NAME
WORKDIR /build

# Cook dependencies (cached layer)
COPY --from=planner /build/recipe.json recipe.json
COPY Cargo.toml Cargo.lock ./
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo chef cook --recipe-path recipe.json --profile=${PROFILE}

# Build application
COPY src src
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo build --bin ${BINARY_NAME} --profile=${PROFILE}

# Move binary to predictable location and compress
RUN if [ "${PROFILE}" = "dev" ]; then \
      cp /build/target/debug/${BINARY_NAME} /${BINARY_NAME}; \
    else \
      cp /build/target/${PROFILE}/${BINARY_NAME} /${BINARY_NAME}; \
    fi
# Stripping is handled by Cargo.toml [profile.release] strip = "symbols".
RUN if [ "${PROFILE}" = "release" ]; then upx --best --lzma /${BINARY_NAME}; fi


# --- Export: minimal image with just the binary ---
# Note: nailbite is a desktop application requiring GTK3, ALSA, X11, etc.
# at runtime. This stage exists for binary extraction via build.sh,
# not for running as a container service.
FROM scratch
ARG SERVICE_NAME
ARG SERVICE_VERSION
ARG BINARY_NAME
COPY --from=builder /${BINARY_NAME} /nailbite
LABEL org.opencontainers.image.title="${SERVICE_NAME}"
LABEL org.opencontainers.image.version="${SERVICE_VERSION}"
ENTRYPOINT ["/nailbite"]
