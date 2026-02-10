{ pkgs ? import <nixpkgs> {} }:

let
  # ort 2.0.0-rc.11 requires ONNX Runtime >= 1.23.x; nixpkgs has 1.22.2.
  onnxruntime_1_23 = pkgs.stdenv.mkDerivation rec {
    pname = "onnxruntime";
    version = "1.23.0";

    src = pkgs.fetchurl {
      url = "https://github.com/microsoft/onnxruntime/releases/download/v${version}/onnxruntime-linux-x64-${version}.tgz";
      sha256 = "sha256-tt7qfy4iwQwEMBnylKDqTSpsCuUqAJw0hHZA23XsVYA=";
    };

    nativeBuildInputs = [ pkgs.autoPatchelfHook ];
    buildInputs = [ pkgs.stdenv.cc.cc.lib ];

    dontBuild = true;
    dontConfigure = true;

    installPhase = ''
      mkdir -p $out/lib $out/include
      cp -r lib/* $out/lib/
      cp -r include/* $out/include/
    '';
  };
in

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Rust toolchain
    rustc
    cargo
    clippy
    rustfmt

    # Build tools
    pkg-config
    cmake
    openssl
    llvmPackages.libclang
    clang

    # Audio (rodio/cpal -> alsa-sys)
    alsa-lib

    # Tauri 2 dependencies
    gtk3
    glib
    libayatana-appindicator
    webkitgtk_4_1
    glib-networking
    libsoup_3
    dbus

    # Display (Tauri uses winit -> wayland + x11)
    wayland
    libxkbcommon
    xorg.libX11
    xorg.libXcursor
    xorg.libXrandr
    xorg.libXi
    xorg.libxcb
    libGL
    vulkan-loader

    # Global hotkeys (tauri-plugin-global-shortcut -> libxdo)
    xdotool

    # Node.js for frontend
    nodejs_22
    nodePackages.pnpm

    # GStreamer for WebKitGTK camera access (getUserMedia)
    gst_all_1.gstreamer
    gst_all_1.gst-plugins-base
    gst_all_1.gst-plugins-good
    gst_all_1.gst-plugins-bad
    gst_all_1.gst-plugins-ugly
    gst_all_1.gst-libav

    # PipeWire for modern Linux camera access
    pipewire
    wireplumber

    # V4L2 camera capture (linux headers for videodev2.h)
    linuxHeaders
  ];

  LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

  # Bindgen needs to find linux headers for V4L2
  BINDGEN_EXTRA_CLANG_ARGS = "-I${pkgs.linuxHeaders}/include -I${pkgs.glibc.dev}/include";

  LD_LIBRARY_PATH = with pkgs; lib.makeLibraryPath [
    wayland
    libxkbcommon
    xorg.libX11
    xorg.libXcursor
    xorg.libXrandr
    xorg.libXi
    xorg.libxcb
    libGL
    vulkan-loader
    libayatana-appindicator
    gtk3
    glib
    webkitgtk_4_1
    libsoup_3
    onnxruntime_1_23
  ];

  ORT_DYLIB_PATH = "${onnxruntime_1_23}/lib/libonnxruntime.so";

  # GIO modules for HTTPS in WebKitGTK
  GIO_MODULE_DIR = "${pkgs.glib-networking}/lib/gio/modules";

  # GStreamer plugin paths for WebKitGTK camera access
  GST_PLUGIN_PATH = with pkgs; lib.makeSearchPath "lib/gstreamer-1.0" [
    gst_all_1.gstreamer
    gst_all_1.gst-plugins-base
    gst_all_1.gst-plugins-good
    gst_all_1.gst-plugins-bad
    gst_all_1.gst-plugins-ugly
    gst_all_1.gst-libav
    pipewire
  ];

  shellHook = ''
    echo "Nailbite Tauri development shell"
    echo ""
    echo "Commands:"
    echo "  pnpm install         # Install frontend dependencies"
    echo "  pnpm tauri dev       # Run in development mode"
    echo "  pnpm tauri build     # Build for production"
    echo ""
  '';
}
