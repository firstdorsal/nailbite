//! Execution provider selection and GPU detection.
//!
//! Handles runtime detection of GPU availability and constructs the
//! appropriate execution provider chain with fallback to CPU.

use tracing::{info, warn};

use crate::config::{GpuBackend, GpuConfig, GpuPreference};
use crate::errors::InferenceError;

/// The active execution provider for a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveProvider {
    Cpu,
    Cuda,
    TensorRt,
    MiGraphX,
    RoCm,
}

impl std::fmt::Display for ActiveProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Cuda => write!(f, "CUDA"),
            Self::TensorRt => write!(f, "TensorRT"),
            Self::MiGraphX => write!(f, "MIGraphX"),
            Self::RoCm => write!(f, "ROCm"),
        }
    }
}

/// Build execution providers based on configuration.
///
/// Returns a vector of execution provider dispatches in priority order,
/// along with the expected active provider. The `ort` crate will try each
/// provider in order and fall back through the list.
///
/// # Errors
///
/// Returns `InferenceError::GpuRequired` if GPU is required but no GPU
/// execution providers are available (either not compiled or not detected).
pub fn build_execution_providers(
    gpu_config: &GpuConfig,
) -> Result<(Vec<ort::execution_providers::ExecutionProviderDispatch>, ActiveProvider), InferenceError>
{
    match gpu_config.preference {
        GpuPreference::Disabled => {
            info!("GPU disabled by configuration, using CPU");
            Ok((vec![], ActiveProvider::Cpu))
        }
        GpuPreference::Auto | GpuPreference::Required => {
            let (providers, expected) = build_gpu_providers(gpu_config);

            if providers.is_empty() && gpu_config.preference == GpuPreference::Required {
                return Err(InferenceError::GpuRequired {
                    reason: "No GPU execution providers available (not compiled with GPU support or GPU not detected)".to_string(),
                });
            }

            if providers.is_empty() {
                info!("No GPU providers available, falling back to CPU");
            }

            Ok((providers, expected))
        }
    }
}

/// Build GPU providers based on backend configuration.
#[allow(unused_variables, unused_mut)]
fn build_gpu_providers(
    gpu_config: &GpuConfig,
) -> (Vec<ort::execution_providers::ExecutionProviderDispatch>, ActiveProvider) {
    let mut providers = Vec::new();
    let mut expected = ActiveProvider::Cpu;

    match gpu_config.backend {
        GpuBackend::Auto => {
            // Priority: TensorRT > CUDA > MIGraphX > ROCm > CPU
            #[cfg(feature = "tensorrt")]
            {
                if let Some(ep) = try_build_tensorrt(gpu_config) {
                    providers.push(ep);
                    if expected == ActiveProvider::Cpu {
                        expected = ActiveProvider::TensorRt;
                    }
                }
            }

            #[cfg(feature = "cuda")]
            {
                if let Some(ep) = try_build_cuda(gpu_config) {
                    providers.push(ep);
                    if expected == ActiveProvider::Cpu {
                        expected = ActiveProvider::Cuda;
                    }
                }
            }

            #[cfg(feature = "migraphx")]
            {
                if let Some(ep) = try_build_migraphx(gpu_config) {
                    providers.push(ep);
                    if expected == ActiveProvider::Cpu {
                        expected = ActiveProvider::MiGraphX;
                    }
                }
            }

            #[cfg(feature = "rocm")]
            {
                if let Some(ep) = try_build_rocm(gpu_config) {
                    providers.push(ep);
                    if expected == ActiveProvider::Cpu {
                        expected = ActiveProvider::RoCm;
                    }
                }
            }
        }

        GpuBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                if let Some(ep) = try_build_cuda(gpu_config) {
                    providers.push(ep);
                    expected = ActiveProvider::Cuda;
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                warn!("CUDA backend requested but not compiled with 'cuda' feature");
            }
        }

        GpuBackend::TensorRt => {
            #[cfg(feature = "tensorrt")]
            {
                if let Some(ep) = try_build_tensorrt(gpu_config) {
                    providers.push(ep);
                    expected = ActiveProvider::TensorRt;
                }
            }
            #[cfg(not(feature = "tensorrt"))]
            {
                warn!("TensorRT backend requested but not compiled with 'tensorrt' feature");
            }

            // TensorRT falls back to CUDA
            #[cfg(feature = "cuda")]
            {
                if let Some(ep) = try_build_cuda(gpu_config) {
                    providers.push(ep);
                    if expected == ActiveProvider::Cpu {
                        expected = ActiveProvider::Cuda;
                    }
                }
            }
        }

        GpuBackend::MiGraphX => {
            #[cfg(feature = "migraphx")]
            {
                if let Some(ep) = try_build_migraphx(gpu_config) {
                    providers.push(ep);
                    expected = ActiveProvider::MiGraphX;
                }
            }
            #[cfg(not(feature = "migraphx"))]
            {
                warn!("MIGraphX backend requested but not compiled with 'migraphx' feature");
            }
        }
    }

    (providers, expected)
}

#[cfg(feature = "cuda")]
fn try_build_cuda(
    gpu_config: &GpuConfig,
) -> Option<ort::execution_providers::ExecutionProviderDispatch> {
    use ort::execution_providers::CUDAExecutionProvider;

    let cuda = CUDAExecutionProvider::default();

    if !cuda.is_available() {
        warn!("CUDA execution provider not available (missing CUDA libraries or GPU)");
        return None;
    }

    info!(device_id = gpu_config.device_id, "CUDA execution provider available");

    let mut builder = CUDAExecutionProvider::default()
        .with_device_id(i32::try_from(gpu_config.device_id).unwrap_or(0));

    if let Some(limit_mb) = gpu_config.memory_limit_mb {
        let limit_bytes = usize::try_from(limit_mb).unwrap_or(0) * 1024 * 1024;
        builder = builder.with_memory_limit(limit_bytes);
    }

    Some(builder.build())
}

#[cfg(feature = "tensorrt")]
fn try_build_tensorrt(
    gpu_config: &GpuConfig,
) -> Option<ort::execution_providers::ExecutionProviderDispatch> {
    use ort::execution_providers::TensorRTExecutionProvider;

    let trt = TensorRTExecutionProvider::default();

    if !trt.is_available() {
        warn!("TensorRT execution provider not available");
        return None;
    }

    info!(device_id = gpu_config.device_id, "TensorRT execution provider available");

    let mut builder = TensorRTExecutionProvider::default()
        .with_device_id(i32::try_from(gpu_config.device_id).unwrap_or(0));

    if gpu_config.fp16_enable {
        builder = builder.with_fp16(true);
    }

    Some(builder.build())
}

#[cfg(feature = "migraphx")]
fn try_build_migraphx(
    gpu_config: &GpuConfig,
) -> Option<ort::execution_providers::ExecutionProviderDispatch> {
    use ort::execution_providers::MIGraphXExecutionProvider;

    let migraphx = MIGraphXExecutionProvider::default();

    if !migraphx.is_available() {
        warn!("MIGraphX execution provider not available (missing ROCm or AMD GPU)");
        return None;
    }

    info!(device_id = gpu_config.device_id, "MIGraphX execution provider available");

    let mut builder = MIGraphXExecutionProvider::default()
        .with_device_id(i32::try_from(gpu_config.device_id).unwrap_or(0));

    if gpu_config.fp16_enable {
        builder = builder.with_fp16(true);
    }

    Some(builder.build())
}

#[cfg(feature = "rocm")]
fn try_build_rocm(
    gpu_config: &GpuConfig,
) -> Option<ort::execution_providers::ExecutionProviderDispatch> {
    use ort::execution_providers::ROCmExecutionProvider;

    let rocm = ROCmExecutionProvider::default();

    if !rocm.is_available() {
        warn!("ROCm execution provider not available");
        return None;
    }

    info!(device_id = gpu_config.device_id, "ROCm execution provider available");

    Some(
        ROCmExecutionProvider::default()
            .with_device_id(i32::try_from(gpu_config.device_id).unwrap_or(0))
            .build(),
    )
}

/// Check which GPU providers are available at runtime.
///
/// Returns a list of (provider, is_available) tuples for each compiled
/// GPU backend. Useful for UI to show GPU status.
#[allow(unused_mut)]
pub fn detect_available_gpus() -> Vec<(ActiveProvider, bool)> {
    let mut results = Vec::new();

    #[cfg(feature = "tensorrt")]
    {
        use ort::execution_providers::TensorRTExecutionProvider;
        let available = TensorRTExecutionProvider::default().is_available();
        results.push((ActiveProvider::TensorRt, available));
    }

    #[cfg(feature = "cuda")]
    {
        use ort::execution_providers::CUDAExecutionProvider;
        let available = CUDAExecutionProvider::default().is_available();
        results.push((ActiveProvider::Cuda, available));
    }

    #[cfg(feature = "migraphx")]
    {
        use ort::execution_providers::MIGraphXExecutionProvider;
        let available = MIGraphXExecutionProvider::default().is_available();
        results.push((ActiveProvider::MiGraphX, available));
    }

    #[cfg(feature = "rocm")]
    {
        use ort::execution_providers::ROCmExecutionProvider;
        let available = ROCmExecutionProvider::default().is_available();
        results.push((ActiveProvider::RoCm, available));
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_gpu_returns_empty_providers() {
        let config = GpuConfig {
            preference: GpuPreference::Disabled,
            backend: GpuBackend::Auto,
            device_id: 0,
            fp16_enable: true,
            memory_limit_mb: None,
        };

        let (providers, active) = build_execution_providers(&config).unwrap();
        assert!(providers.is_empty());
        assert_eq!(active, ActiveProvider::Cpu);
    }

    #[test]
    #[cfg(not(any(
        feature = "cuda",
        feature = "migraphx",
        feature = "rocm",
        feature = "tensorrt"
    )))]
    fn required_gpu_fails_without_gpu_features() {
        let config = GpuConfig {
            preference: GpuPreference::Required,
            backend: GpuBackend::Auto,
            device_id: 0,
            fp16_enable: true,
            memory_limit_mb: None,
        };

        let result = build_execution_providers(&config);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, InferenceError::GpuRequired { .. }));
    }

    #[test]
    fn detect_available_gpus_returns_vec() {
        // Should not panic, returns empty vec without GPU features
        let gpus = detect_available_gpus();
        // Just verify it doesn't crash and returns a vec
        let _ = gpus.len();
    }

    #[test]
    fn gpu_config_default_is_auto() {
        let config = GpuConfig::default();
        assert_eq!(config.preference, GpuPreference::Auto);
        assert_eq!(config.backend, GpuBackend::Auto);
        assert_eq!(config.device_id, 0);
        assert!(config.fp16_enable);
        assert!(config.memory_limit_mb.is_none());
    }

    #[test]
    fn active_provider_display() {
        assert_eq!(ActiveProvider::Cpu.to_string(), "CPU");
        assert_eq!(ActiveProvider::Cuda.to_string(), "CUDA");
        assert_eq!(ActiveProvider::TensorRt.to_string(), "TensorRT");
        assert_eq!(ActiveProvider::MiGraphX.to_string(), "MIGraphX");
        assert_eq!(ActiveProvider::RoCm.to_string(), "ROCm");
    }
}
