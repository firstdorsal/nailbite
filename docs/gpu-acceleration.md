# GPU Acceleration

n**AI**lbite supports GPU acceleration for ONNX model inference via CUDA, TensorRT, ROCm, and MIGraphX.

## Supported Backends

| Backend | GPU | Notes |
|---------|-----|-------|
| CUDA | NVIDIA | Requires CUDA 11.8+ |
| TensorRT | NVIDIA | Fastest, requires TensorRT 8.6+ |
| ROCm | AMD | Requires ROCm 5.4+ |
| MIGraphX | AMD | AMD's graph optimizer |

## Configuration

In `config.yaml`:

```yaml
ort:
  gpu:
    preference: auto      # disabled, auto, preferred, required
    backend: auto         # auto, cuda, tensorrt, rocm, migraphx
    device_id: 0          # GPU index
    fp16_enable: true     # Use FP16 for faster inference
    memory_limit_mb: null # Limit GPU memory (null = unlimited)
```

### Preference Levels

- **disabled** - CPU only, ignore GPU even if available
- **auto** - Use GPU if available, silently fallback to CPU
- **preferred** - Use GPU if available, log warning if not
- **required** - Fail startup if GPU not available

### Backend Selection

- **auto** - Try backends in order: TensorRT → CUDA → MIGraphX → ROCm → CPU
- **cuda** - NVIDIA CUDA only
- **tensorrt** - NVIDIA TensorRT only (fastest)
- **rocm** - AMD ROCm only
- **migraphx** - AMD MIGraphX only

## Building with GPU Support

### NVIDIA (CUDA/TensorRT)

```bash
# Build with CUDA
GPU_BACKEND=cuda bash build.sh

# Build with TensorRT
GPU_BACKEND=tensorrt bash build.sh
```

Required:
- CUDA Toolkit 11.8+
- cuDNN 8.6+
- TensorRT 8.6+ (for TensorRT backend)

### AMD (ROCm/MIGraphX)

```bash
# Build with ROCm
GPU_BACKEND=rocm bash build.sh

# Build with MIGraphX
GPU_BACKEND=migraphx bash build.sh
```

Required:
- ROCm 5.4+
- MIGraphX (for MIGraphX backend)

## Docker Images

Pre-built Docker images with GPU support:

```bash
# NVIDIA
docker build -f Dockerfile.cuda -t nailbite:cuda .

# AMD
docker build -f Dockerfile.rocm -t nailbite:rocm .
```

Run with GPU access:

```bash
# NVIDIA
docker run --gpus all -v /dev/video0:/dev/video0 nailbite:cuda

# AMD
docker run --device=/dev/kfd --device=/dev/dri nailbite:rocm
```

## Performance Comparison

Typical inference times on 640x480 frames:

| Backend | Hand Detection | Face Mesh | Total Pipeline |
|---------|---------------|-----------|----------------|
| CPU (4 threads) | ~25ms | ~30ms | ~80ms |
| CUDA | ~5ms | ~6ms | ~20ms |
| TensorRT | ~3ms | ~4ms | ~12ms |
| ROCm | ~6ms | ~7ms | ~22ms |

*Results vary by hardware. Tested on RTX 3080 / RX 6800.*

## Troubleshooting

### GPU Not Detected

1. Check GPU is visible:
   ```bash
   # NVIDIA
   nvidia-smi

   # AMD
   rocm-smi
   ```

2. Set preference to `required` to see error messages:
   ```yaml
   ort:
     gpu:
       preference: required
   ```

3. Check ONNX Runtime was built with GPU support:
   ```bash
   RUST_LOG=debug pnpm tauri dev 2>&1 | grep -i "execution provider"
   ```

### Out of Memory

Limit GPU memory:

```yaml
ort:
  gpu:
    memory_limit_mb: 1024  # 1GB limit
```

Or reduce batch size / resolution:

```yaml
camera:
  sources:
    - resolution_width: 320
      resolution_height: 240
```

### TensorRT Compilation Slow

TensorRT compiles optimized engines on first run. This can take several minutes. Engines are cached for subsequent runs.

### ROCm Permission Denied

Add user to video and render groups:

```bash
sudo usermod -aG video,render $USER
# Log out and back in
```

## FP16 Mode

FP16 (half precision) reduces memory usage and increases speed with minimal accuracy loss:

```yaml
ort:
  gpu:
    fp16_enable: true
```

Recommended for:
- RTX 20/30/40 series (Tensor Cores)
- AMD RDNA2+ GPUs

Not recommended for:
- Older GPUs without native FP16 support
- When maximum accuracy is needed

## Multi-GPU

Select GPU by device ID:

```yaml
ort:
  gpu:
    device_id: 1  # Use second GPU
```

Currently only single-GPU inference is supported. Multi-GPU would require model sharding.
