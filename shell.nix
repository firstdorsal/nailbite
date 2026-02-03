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

    # System tray (tray-icon -> gtk3 + libappindicator)
    gtk3
    glib
    libayatana-appindicator

    # GUI (eframe/egui -> winit -> wayland + x11)
    wayland
    libxkbcommon
    xorg.libX11
    xorg.libXcursor
    xorg.libXrandr
    xorg.libXi
    xorg.libxcb
    libGL
    vulkan-loader

    # Global hotkeys (global-hotkey -> libxdo)
    xdotool

    # Camera (v4l)
    v4l-utils
    linuxHeaders
  ];

  LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
  BINDGEN_EXTRA_CLANG_ARGS = "-isystem ${pkgs.linuxHeaders}/include -isystem ${pkgs.glibc.dev}/include";

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
    onnxruntime_1_23
  ];

  ORT_DYLIB_PATH = "${onnxruntime_1_23}/lib/libonnxruntime.so";
}
