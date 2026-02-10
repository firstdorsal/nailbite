# Troubleshooting

## Camera Issues

### Camera not detected
- Check that your camera is connected: `ls /dev/video*`
- Verify V4L2 compatibility: `v4l2-ctl --list-devices`
- Ensure the device path in `config.yaml` matches your camera

### Permission denied accessing camera
```bash
# Add your user to the video group
sudo usermod -aG video $USER
# Log out and back in for changes to take effect
```

### No video in preview
- WebKitGTK may need a reload. Try restarting the app
- Check that no other application is using the camera
- Try a different camera if you have multiple

## Model Download Issues

### Download failed
- Check your internet connection
- Models are downloaded from GitHub and HuggingFace - ensure these aren't blocked
- Manually download models:
  ```bash
  mkdir -p models
  curl -L -o models/palm_detection.onnx "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx"
  # Repeat for other models (see src-tauri/src/inference/model_downloader.rs for URLs)
  ```

### Checksum mismatch
- The model file may be corrupted or truncated
- Delete the model file and let it re-download:
  ```bash
  rm models/palm_detection.onnx
  pnpm tauri dev  # Will re-download
  ```

## ONNX Runtime Issues

### libonnxruntime.so not found
ONNX Runtime uses dynamic loading. Ensure the library is available:

**NixOS**: Library is provided by `shell.nix`

**Other Linux**:
```bash
# Download ONNX Runtime
ORT_VERSION="1.20.1"
curl -L -o onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"
tar xzf onnxruntime.tgz
export LD_LIBRARY_PATH="$PWD/onnxruntime-linux-x64-${ORT_VERSION}/lib:$LD_LIBRARY_PATH"
```

### GPU not detected
- Ensure you built with GPU features: `GPU_BACKEND=cuda bash build.sh`
- Check CUDA/ROCm installation
- Set `ort.gpu.preference: required` in config to see errors

## Detection Issues

### False positives (detecting when not biting nails)
- Increase `detection.behaviors.nail_biting.proximity_threshold` (e.g., 0.25 → 0.15)
- Increase `detection.temporal.positive_ratio` (e.g., 0.4 → 0.6)
- Enable `detection.false_positive.chin_rest_suppression`

### False negatives (not detecting when biting)
- Decrease proximity threshold
- Decrease `detection.temporal.positive_ratio`
- Ensure good lighting for camera

### Hands not tracked correctly
- Ensure hands are visible in frame
- Check that face is also visible (helps with spatial reference)
- Try different camera position/angle

## Audio Issues

### No sound on alert
- Check system volume
- Verify ALSA is working: `aplay /usr/share/sounds/alsa/Front_Center.wav`
- Check `actions.sound.enabled: true` in config

### Sound doesn't stop
- Press F9 to dismiss the alert
- Complete the exercise to stop alert

## Build Issues

### Cargo build fails with GTK errors
Ensure WebKitGTK 4.1 and dependencies are installed:
```bash
sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev
```

### pnpm tauri build fails
- Check Node.js version: `node --version` (should be 22+)
- Check Rust version: `rustc --version` (should be 1.85+)
- Run `pnpm install` to ensure dependencies are installed

## Performance Issues

### High CPU usage
- Reduce `camera.inference_fps` (e.g., 8 → 4)
- Reduce camera resolution in config
- Enable GPU acceleration if available

### Choppy video preview
- Reduce camera resolution
- Close other applications using webcam
- Reduce inference FPS

## Still Having Issues?

1. Check logs: `RUST_LOG=debug pnpm tauri dev`
2. File an issue: https://github.com/your-username/nailbite/issues
3. Include:
   - OS and version
   - Camera model
   - Error messages from console
   - `config.yaml` (remove any sensitive data)
