# Nailbite Tauri Build Dockerfile
#
# Unified Dockerfile for all build variants (CPU, CUDA, TensorRT, ROCm, MIGraphX).
# Produces: AppImage and deb packages.
#
# Uses Ubuntu 24.04 with WebKitGTK pinned to 2.44.0-2 (last version before the
# DMA-BUF renderer regression that causes "Could not create default EGL display:
# EGL_BAD_PARAMETER" on many Linux systems including NixOS, Arch, etc.).
# See: https://github.com/nicehash/NiceHashQuickMiner/issues/4
#      https://github.com/nicehash/NiceHashQuickMiner/issues/4
#      https://github.com/tauri-apps/tauri/issues/11994
#
# Usage via build.sh (recommended):
#   bash build.sh                        # CPU-only build
#   GPU_BACKEND=cuda bash build.sh       # NVIDIA CUDA build
#   GPU_BACKEND=rocm bash build.sh       # AMD ROCm/MIGraphX build
#
# Or directly:
#   docker build -t nailbite:cpu .
#   docker build --build-arg BASE_IMAGE=nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 \
#                --build-arg CARGO_FEATURES="--features cuda" -t nailbite:cuda .
#
ARG BASE_IMAGE=ubuntu:24.04
ARG NODE_VERSION="22"
ARG RUST_VERSION="1.88.0"
ARG PNPM_VERSION="10"

# =============================================================================
# Stage 1: Base builder with all system dependencies
# =============================================================================
FROM ${BASE_IMAGE} AS builder-base

ARG NODE_VERSION
ARG RUST_VERSION
ARG PNPM_VERSION

# Install base build tools and Node.js (bare Ubuntu and GPU base images need both)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates gnupg && \
    if ! command -v node >/dev/null 2>&1; then \
        mkdir -p /etc/apt/keyrings && \
        curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | \
            gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
        echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_VERSION}.x nodistro main" | \
            tee /etc/apt/sources.list.d/nodesource.list && \
        apt-get update && apt-get install -y nodejs; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain ${RUST_VERSION} && \
    rm -rf /root/.cargo/registry /root/.cargo/git
ENV PATH="/root/.cargo/bin:${PATH}"

# Install pnpm (use npm instead of corepack to avoid network issues in Docker)
RUN npm install -g pnpm@${PNPM_VERSION}

# Install cargo tools for caching
RUN cargo install cargo-chef --locked && \
    cargo install --locked --no-default-features sccache
ENV RUSTC_WRAPPER=sccache SCCACHE_DIR=/sccache
ENV CARGO_INCREMENTAL=0
ENV SOURCE_DATE_EPOCH=0

# Pin WebKitGTK to 2.44.0-2 (last version before the EGL/DMA-BUF regression).
# WebKitGTK >= 2.46 introduced a DMA-BUF renderer that hard-aborts when EGL
# initialization fails, which happens in AppImage sandboxed environments.
ARG WEBKIT_VERSION="2.44.0-2"

# Install system build dependencies for Tauri + Nailbite
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    pkg-config \
    cmake \
    clang \
    libclang-dev \
    # AppImage bundling (linuxdeploy requirements)
    file \
    libfuse2t64 \
    patchelf \
    squashfs-tools \
    # Tauri/WebKitGTK dependencies (pinned to avoid EGL regression)
    libwebkit2gtk-4.1-0=${WEBKIT_VERSION} \
    libwebkit2gtk-4.1-dev=${WEBKIT_VERSION} \
    libjavascriptcoregtk-4.1-0=${WEBKIT_VERSION} \
    libjavascriptcoregtk-4.1-dev=${WEBKIT_VERSION} \
    gir1.2-javascriptcoregtk-4.1=${WEBKIT_VERSION} \
    gir1.2-webkit2-4.1=${WEBKIT_VERSION} \
    libgtk-3-dev \
    libglib2.0-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev \
    libsoup-3.0-dev \
    # Audio (rodio -> ALSA)
    libasound2-dev \
    # GUI/Display (X11 + Wayland + OpenGL)
    libwayland-dev \
    libxkbcommon-dev \
    libx11-dev \
    libxcursor-dev \
    libxrandr-dev \
    libxi-dev \
    libxcb1-dev \
    libgl-dev \
    libvulkan-dev \
    # Global hotkeys (libxdo)
    libxdo-dev \
    # Camera (v4l)
    libv4l-dev \
    linux-libc-dev \
    # TLS
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime (ort crate uses load-dynamic / dlopen at runtime).
#
# Two install modes, selected by ROCM_ORT:
#   * ROCM_ORT=0 (default): upstream Microsoft CPU tarball.
#   * ROCM_ORT=1: community-built MIGraphX-enabled tarball from the
#     Looong01/onnxruntime-rocm-build releases. This ships
#     libonnxruntime.so (matching ORT_VERSION ABI), plus
#     libonnxruntime_providers_migraphx.so and the shared loader, in a
#     drop-in layout identical to Microsoft's tarball. Microsoft does
#     not publish ROCm/MIGraphX tarballs; the community build tracks
#     the same release tags so we get an ABI-compatible 1.23+ binary
#     ort 2.0.0-rc.11 needs.
ARG ORT_VERSION="1.23.0"
ARG ROCM_ORT="0"
RUN if [ "${ROCM_ORT}" = "1" ]; then \
        echo "Installing MIGraphX ONNX Runtime ${ORT_VERSION} (community tarball)" && \
        curl -fsSL --retry 3 \
            "https://github.com/Looong01/onnxruntime-rocm-build/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-migraphx-${ORT_VERSION}.tgz" \
            | tar xz --strip-components=1 -C /usr/local \
                onnxruntime-linux-x64-rocm-${ORT_VERSION}/lib && \
        ls -la /usr/local/lib/libonnxruntime* && \
        ldconfig; \
    else \
        echo "Installing upstream CPU ONNX Runtime tarball v${ORT_VERSION}" && \
        curl -sL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
            | tar xz --strip-components=1 -C /usr/local onnxruntime-linux-x64-${ORT_VERSION}/lib && \
        ldconfig; \
    fi

WORKDIR /app

# =============================================================================
# Stage 2: Prepare cargo-chef recipe for Rust dependency caching
# =============================================================================
FROM builder-base AS planner

# Copy Tauri backend sources for recipe
COPY src-tauri/Cargo.toml src-tauri/Cargo.lock ./src-tauri/
COPY src-tauri/src ./src-tauri/src
COPY src-tauri/build.rs ./src-tauri/

WORKDIR /app/src-tauri
RUN cargo chef prepare --recipe-path /recipe.json

# =============================================================================
# Stage 3: Build frontend and backend
# =============================================================================
FROM builder-base AS builder

ARG CARGO_FEATURES=""
ARG TAURI_BUNDLES="appimage,deb"

# ---- Frontend dependencies (cached layer) ----
COPY package.json pnpm-lock.yaml ./
RUN --mount=type=cache,target=/root/.local/share/pnpm/store \
    pnpm install --frozen-lockfile

# ---- Rust dependencies (cached layer via cargo-chef) ----
COPY --from=planner /recipe.json /recipe.json
COPY src-tauri/Cargo.toml src-tauri/Cargo.lock ./src-tauri/
COPY src-tauri/build.rs ./src-tauri/

WORKDIR /app/src-tauri
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo chef cook --release --recipe-path /recipe.json ${CARGO_FEATURES}

# ---- Build frontend ----
WORKDIR /app
COPY index.html vite.config.ts tsconfig.json tsconfig.node.json tailwind.config.ts postcss.config.js components.json ./
COPY src ./src
RUN pnpm build

# ---- Build Tauri backend ----
COPY src-tauri ./src-tauri
WORKDIR /app/src-tauri
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo build --release ${CARGO_FEATURES}

# ---- Pre-cache and patch AppImage tools ----
# glibc >= 2.36 rejects the non-standard ABI version (65 = 'A' from AppImage "AI"
# magic) in AppImage ELF headers. Patch byte 8 to 0 for compatibility.
ENV APPIMAGE_EXTRACT_AND_RUN=1
RUN mkdir -p /root/.cache/tauri && cd /root/.cache/tauri && \
    curl -sL -o AppRun-x86_64 https://github.com/tauri-apps/binary-releases/releases/download/apprun-old/AppRun-x86_64 && \
    curl -sL -o linuxdeploy-x86_64.AppImage https://github.com/tauri-apps/binary-releases/releases/download/linuxdeploy/linuxdeploy-x86_64.AppImage && \
    curl -sL -o linuxdeploy-plugin-gtk.sh https://raw.githubusercontent.com/tauri-apps/linuxdeploy-plugin-gtk/master/linuxdeploy-plugin-gtk.sh && \
    curl -sL -o linuxdeploy-plugin-gstreamer.sh https://raw.githubusercontent.com/tauri-apps/linuxdeploy-plugin-gstreamer/master/linuxdeploy-plugin-gstreamer.sh && \
    curl -sL -o linuxdeploy-plugin-appimage.AppImage https://github.com/linuxdeploy/linuxdeploy-plugin-appimage/releases/download/continuous/linuxdeploy-plugin-appimage-x86_64.AppImage && \
    chmod +x * && \
    printf '\x00' | dd of=linuxdeploy-x86_64.AppImage bs=1 count=1 seek=8 conv=notrunc 2>/dev/null && \
    printf '\x00' | dd of=linuxdeploy-plugin-appimage.AppImage bs=1 count=1 seek=8 conv=notrunc 2>/dev/null

# ---- Build Tauri bundles ----
WORKDIR /app
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    pnpm tauri build --no-bundle && \
    pnpm tauri build --bundles ${TAURI_BUNDLES}

# ---- Inject ONNX Runtime into AppImage ----
# The ort crate uses load-dynamic (dlopen) which is invisible to linuxdeploy.
# Extract the squashfs payload, add libonnxruntime.so + provider libs, and
# repack. For ROCm builds we ship the migraphx + rocm provider plugins so
# that ort can register them at runtime; the host system must still provide
# ROCm runtime libraries (libamdhip64, libMIOpen, …) at the standard search
# paths.
ARG ORT_VERSION
ARG ROCM_ORT
COPY scripts/repack-appimage.sh /usr/local/bin/repack-appimage
RUN chmod +x /usr/local/bin/repack-appimage && \
    if [ "${ROCM_ORT}" = "1" ]; then \
        repack-appimage \
            src-tauri/target/release/bundle/appimage/*.AppImage \
            --dir /usr/local/lib \
            'libonnxruntime.so' 'libonnxruntime.so.1' 'libonnxruntime.so.*' \
            'libonnxruntime_providers_*.so'; \
    else \
        repack-appimage \
            src-tauri/target/release/bundle/appimage/*.AppImage \
            /usr/local/lib/libonnxruntime.so.${ORT_VERSION} \
            libonnxruntime.so.1 libonnxruntime.so; \
    fi

# ---- Verify bundles ----
RUN echo "Verifying binary dependencies..." && \
    test -f src-tauri/target/release/bundle/appimage/*.AppImage && \
    echo "Verifying deb..." && \
    test -f src-tauri/target/release/bundle/deb/*.deb && \
    echo "All bundle checks passed."

# =============================================================================
# Stage 4: Export artifacts
# =============================================================================
FROM scratch AS export

# Copy all build artifacts (binary is named after the package: nailbite-tauri)
COPY --from=builder /app/src-tauri/target/release/nailbite-tauri /nailbite
COPY --from=builder /app/src-tauri/target/release/bundle /bundle

LABEL org.opencontainers.image.title="nailbite"
LABEL org.opencontainers.image.description="BFRB detection and decoupling exercise system"
