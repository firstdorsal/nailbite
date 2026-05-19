# AMD GPU Acceleration Runbook

## How the GPU build is wired

`bash build.sh GPU_BACKEND=migraphx` (or `=rocm`) produces an AppImage that:

1. Compiles the Rust backend with `--features migraphx` (or `--features amd-gpu`),
   wiring in `MIGraphXExecutionProvider` (and `ROCmExecutionProvider`).
2. Downloads
   `onnxruntime-linux-x64-migraphx-${ORT_VERSION}.tgz` from the
   community-maintained
   [Looong01/onnxruntime-rocm-build](https://github.com/Looong01/onnxruntime-rocm-build)
   releases. Microsoft does **not** publish ROCm/MIGraphX tarballs;
   this fork tracks upstream tags so the ABI matches the `ort` crate
   (we need 1.23.x for `ort 2.0.0-rc.11`). The PyPI `onnxruntime-rocm`
   wheel is older (1.22) and would fail with `not compatible with the
   ONNX Runtime binary found at libonnxruntime.so`.
3. Extracts `libonnxruntime.so{,.1,.1.23.0}` +
   `libonnxruntime_providers_migraphx.so` +
   `libonnxruntime_providers_shared.so` into `/usr/local/lib`.
4. Bundles those libs into the AppImage via `scripts/repack-appimage.sh`.

The AppImage is **not** self-contained. The host system must provide:

- `libamdhip64.so.6`   (HIP runtime, from `rocmPackages.clr`)
- `libmigraphx_c.so.3` (MIGraphX runtime, from `rocmPackages.migraphx`)
- `libMIOpen.so.1`     (MIOpen, transitively via `migraphx`)
- AMD GPU device nodes `/dev/kfd` and `/dev/dri/renderD128` with
  `render`/`video` group access.

Bundling these would push the AppImage over 2 GB and create version-skew
risk against the kernel `amdgpu` module — it's safer to require them on
the host.

## Required NixOS configuration

Replace the relevant block in `configuration.nix` with the following.
The key change is `rocmPackages.clr` (full HIP runtime, not just `.icd`)
plus `rocmPackages.migraphx`:

```nix
hardware.graphics = {
  enable = true;
  enable32Bit = true;
  extraPackages = with pkgs; [
    rocmPackages.clr             # full HIP runtime (provides libamdhip64.so.6)
    rocmPackages.clr.icd         # OpenCL ICD
    rocmPackages.migraphx        # MIGraphX runtime (libmigraphx_c.so.3, MIOpen, hipBLAS)
    rocmPackages.miopen          # explicit dep guard
    rocmPackages.hipblas         # explicit dep guard
    amdvlk
  ];
};

systemd.tmpfiles.rules = [
  "L+ /opt/rocm - - - - ${pkgs.rocmPackages.clr}"
];

environment.sessionVariables = {
  HSA_OVERRIDE_GFX_VERSION = "10.3.0";  # spoof gfx1032 (Navi 23) → gfx1030 (Navi 21, officially supported)
  ROCM_PATH = "/opt/rocm";
};

users.users.paul.extraGroups = [ "video" "render" ];
```

After `sudo nixos-rebuild switch && reboot`, verify:

```bash
rocminfo                                    # should list "Agent 2: Type=GPU, Marketing Name=Navi 23"
ls /opt/rocm/lib/libamdhip64.so.6           # should resolve to a real file
ls /opt/rocm/lib/libmigraphx_c.so.3         # ditto
groups | grep -E 'render|video'             # both required for /dev/kfd access
```

## Running the GPU AppImage

```bash
# After the rebuild + reboot:
bash build.sh GPU_BACKEND=migraphx
./dist/bundle/appimage/nailbite_*.AppImage
```

Watch the log for the provider line — you should see:

```
INFO  …inference::execution_provider: MIGraphX execution provider available
INFO  …inference::session: Model session created provider=MIGraphX
INFO  …commands::camera: Hand landmark backend selected backend="rtmpose_m_hand5"
```

If you instead see `"No GPU providers available, falling back to CPU"`, run:

```bash
LIBGL_DEBUG=verbose ROCR_VISIBLE_DEVICES=0 \
    ./dist/bundle/appimage/nailbite_*.AppImage 2>&1 | head -80
```

The most common failure modes:

| Symptom | Cause | Fix |
|---|---|---|
| `cannot open libamdhip64.so.6` | `clr` not in `hardware.graphics.extraPackages` | add it, rebuild |
| `MIGraphX execution provider not available` | `migraphx` package missing | add `rocmPackages.migraphx`, rebuild |
| `HSA_STATUS_ERROR_INVALID_AGENT` | gfx1032 not spoofed | set `HSA_OVERRIDE_GFX_VERSION=10.3.0` |
| `Permission denied` on `/dev/kfd` | user not in `render` group | `sudo gpasswd -a paul render && relogin` |

## Performance reference

Measured on Ryzen 7950X + Navi 23 (gfx1032) / RTMPose-m hand5, 256×256:

| Backend | Latency / call | Pipeline FPS (full) |
|---|---|---|
| CPU (8 threads) | ~30 ms | ~5–7 FPS |
| MIGraphX        | ~5–8 ms | ~20–25 FPS |
| ROCm            | ~6–9 ms | ~18–22 FPS |

Latency is single-hand. Both hands present roughly doubles the
landmarker cost since `palm_detection → hand_landmark` runs once per
ROI.

## Why we don't ship the full ROCm provider lib

The community tarball is MIGraphX-only (~8 MB compressed); the wheel
distribution that bundles `libonnxruntime_providers_rocm.so` adds
~1.4 GB uncompressed for marginal benefit, since MIGraphX is tried
first in the auto-selection chain and is essentially equivalent in
speed for our workload. If you really need the ROCm provider (e.g.
for ops MIGraphX doesn't yet support), pin `ROCM_ORT_VERSION` in the
Dockerfile to a wheel build and add `libonnxruntime_providers_rocm.so`
to the repack glob — but keep the ABI matched to `ort 2.0.0-rc.11`'s
expected version (`>= 1.23.x`).
