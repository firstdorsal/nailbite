//! Camera control biasing.
//!
//! Knobs that we ask the camera driver to apply once at capture start so
//! the user's face is well-lit regardless of background lighting. The
//! V4L2 implementation lives in [`v4l2`]; the pure arithmetic
//! ([`compute_stepped_value`], [`ControlRange`]) is platform-agnostic so
//! it can be unit tested without a real device and reused when other
//! backends (AVFoundation, MediaFoundation) land.
//!
//! Cross-platform port: the [`ImageControlBackend`] trait is the seam
//! the future macOS/Windows ports will implement.

use crate::config::CameraControlsConfig;

/// Describes the [min, max] range, step, and default value the camera
/// driver reports for a single control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlRange {
    pub min: i64,
    pub max: i64,
    pub step: u64,
    pub default: i64,
}

/// What happened when we tried to apply one biased control. Used to
/// produce a single human-readable summary line at the end of init —
/// silent skips on every camera are otherwise invisible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlOutcome {
    /// Wrote the listed value successfully.
    Applied(i64),
    /// The control isn't exposed by this camera — nothing to write.
    Unsupported,
    /// The control exists but the driver rejected the write.
    Failed,
    /// The control was deliberately skipped (config opted out).
    Skipped,
}

/// Per-startup report of what biases the camera accepted. Useful for
/// telemetry and for the eventual "Controls applied" UI surface.
#[derive(Debug, Clone, Default)]
pub struct ControlsReport {
    pub gamma_reset: Option<ControlOutcome>,
    pub auto_exposure: Option<ControlOutcome>,
    pub exposure_auto_priority: Option<ControlOutcome>,
    pub auto_white_balance: Option<ControlOutcome>,
    pub auto_gain: Option<ControlOutcome>,
    pub backlight_compensation: Option<ControlOutcome>,
    pub brightness: Option<ControlOutcome>,
    pub contrast: Option<ControlOutcome>,
}

/// Platform-agnostic seam for the subject-friendly camera biasing.
///
/// Today only the V4L2 backend exists; the trait lives here so the
/// macOS / Windows ports have an obvious place to land instead of
/// re-deriving "what biases do we want" from scratch.
pub trait ImageControlBackend {
    /// Apply the requested biases. Failures on individual controls
    /// MUST be folded into the returned report rather than propagated
    /// as errors — capture must never abort just because one knob is
    /// unsupported on a given camera.
    fn apply_subject_friendly_biases(
        &self,
        profile: &CameraControlsConfig,
    ) -> ControlsReport;
}

/// Compute the integer value that sits `fraction` of the way between
/// `min` and `max`, rounded to the nearest `step`-aligned position and
/// clamped back into the inclusive `[min, max]` range.
///
/// Math is in `f64` so the helper is well-defined across the full i64
/// range that V4L2 advertises; the only platform precision risk is the
/// final `as i64` cast which is bounded by the clamp.
///
/// Returns `min` when `max <= min` (degenerate range) or when
/// `fraction` is NaN — the alternative would be silently producing a
/// garbage value the caller then writes to the device. Callers that
/// want to skip writing on degenerate ranges should detect that case
/// before calling.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
)]
pub fn compute_stepped_value(min: i64, max: i64, step: u64, fraction: f32) -> i64 {
    if max <= min || !fraction.is_finite() {
        return min;
    }
    let min_f = min as f64;
    let max_f = max as f64;
    let span = max_f - min_f;
    let clamped = f64::from(fraction.clamp(0.0, 1.0));
    let raw = min_f + clamped * span;
    let stepped = if step > 1 {
        let s = step as f64;
        ((raw - min_f) / s).round() * s + min_f
    } else {
        raw.round()
    };
    // Cast is bounded by the clamp; truncation only happens on f64
    // values outside i64 range, which the clamp rules out.
    let result = stepped.clamp(min_f, max_f) as i64;
    result.clamp(min, max)
}

// V4L2 control IDs (linux/videodev2.h). Defined inline; the list is
// stable kernel UAPI but if it grows past ~10 entries we should switch
// to a sys binding rather than keep copying them.
#[cfg(target_os = "linux")]
pub(crate) const V4L2_CID_BRIGHTNESS: u32 = 0x0098_0900;
#[cfg(target_os = "linux")]
pub(crate) const V4L2_CID_CONTRAST: u32 = 0x0098_0901;
#[cfg(target_os = "linux")]
pub(crate) const V4L2_CID_AUTO_WHITE_BALANCE: u32 = 0x0098_090C;
#[cfg(target_os = "linux")]
pub(crate) const V4L2_CID_AUTOGAIN: u32 = 0x0098_0912;
#[cfg(target_os = "linux")]
pub(crate) const V4L2_CID_BACKLIGHT_COMPENSATION: u32 = 0x0098_091C;
#[cfg(target_os = "linux")]
pub(crate) const V4L2_CID_GAMMA: u32 = 0x0098_0910;
#[cfg(target_os = "linux")]
pub(crate) const V4L2_CID_EXPOSURE_AUTO: u32 = 0x009A_0901;
#[cfg(target_os = "linux")]
pub(crate) const V4L2_CID_EXPOSURE_AUTO_PRIORITY: u32 = 0x009A_0903;

// V4L2_CID_EXPOSURE_AUTO menu values (v4l2_exposure_auto_type in
// linux/v4l2-controls.h):
//   0 = AUTO              (auto exposure time, auto iris)
//   1 = MANUAL
//   2 = SHUTTER_PRIORITY  (manual exposure, auto iris)
//   3 = APERTURE_PRIORITY (auto exposure time, manual iris)
//
// We try FULL_AUTO first and fall back to APERTURE_PRIORITY. Both are
// "auto-exposure" in the colloquial sense, but on UVC firmwares that
// implement only the {MANUAL, APERTURE_PRIORITY} subset, aperture
// priority is sometimes interpreted as "exposure stays at whatever the
// driver last set it to" rather than tracking scene brightness — i.e.
// not actually responsive to changing light. Full auto is the value
// that real-world webcams reliably interpret as "adapt to the scene."
#[cfg(target_os = "linux")]
pub(crate) const EXPOSURE_AUTO_FULL_AUTO: i64 = 0;
#[cfg(target_os = "linux")]
pub(crate) const EXPOSURE_AUTO_APERTURE_PRIORITY: i64 = 3;

#[cfg(target_os = "linux")]
pub use linux_impl::enable_auto_image_controls;

#[cfg(target_os = "linux")]
mod linux_impl {
    use super::{
        compute_stepped_value, ControlOutcome, ControlRange, ControlsReport,
        EXPOSURE_AUTO_APERTURE_PRIORITY, EXPOSURE_AUTO_FULL_AUTO,
        V4L2_CID_AUTOGAIN, V4L2_CID_AUTO_WHITE_BALANCE, V4L2_CID_BACKLIGHT_COMPENSATION,
        V4L2_CID_BRIGHTNESS, V4L2_CID_CONTRAST, V4L2_CID_EXPOSURE_AUTO,
        V4L2_CID_EXPOSURE_AUTO_PRIORITY, V4L2_CID_GAMMA,
    };
    use crate::config::CameraControlsConfig;
    use std::collections::HashMap;
    use tracing::{debug, info, warn};
    use v4l::control::{Control, Description, Value as ControlValue};
    use v4l::Device;

    /// Apply the subject-friendly biases to a V4L2 device. Returns a
    /// per-control outcome report. Failures on individual controls are
    /// recorded but never propagated — capture must not abort just
    /// because one knob is unsupported.
    pub fn enable_auto_image_controls(
        dev: &Device,
        profile: &CameraControlsConfig,
    ) -> ControlsReport {
        let mut report = ControlsReport::default();

        // Cache the device's control table once. Avoids issuing a
        // VIDIOC_QUERY_EXT_CTRL ioctl per control read.
        let ranges = match dev.query_controls() {
            Ok(list) => list
                .into_iter()
                .map(|d| (d.id, description_to_range(&d)))
                .collect::<HashMap<u32, ControlRange>>(),
            Err(e) => {
                warn!(
                    error = %e,
                    "Could not query camera control table; subject-friendly biases skipped"
                );
                return report;
            }
        };

        // --- Reset stuck controls back to driver default ---
        // Other apps (browsers, conferencing tools) sometimes leave
        // gamma cranked way above the manufacturer default; the
        // auto-exposure loop can't recover from that and the image
        // arrives permanently over-bright. Writing the driver-reported
        // default back is a load-bearing reset, not a bias.
        report.gamma_reset = Some(if profile.gamma_reset {
            reset_to_default(dev, &ranges, V4L2_CID_GAMMA, "gamma")
        } else {
            ControlOutcome::Skipped
        });

        // --- Auto-exposure ---
        if profile.auto_exposure {
            report.auto_exposure = Some(apply_auto_exposure(dev, &ranges));
        } else {
            report.auto_exposure = Some(ControlOutcome::Skipped);
        }

        // Frame-rate priority: caller usually wants this OFF (false) so
        // the camera can use long exposures to expose the subject.
        report.exposure_auto_priority = Some(set_boolean(
            dev,
            &ranges,
            V4L2_CID_EXPOSURE_AUTO_PRIORITY,
            profile.exposure_auto_priority,
            "exposure_auto_priority",
            true,
        ));

        report.auto_white_balance = Some(if profile.auto_white_balance {
            set_boolean(
                dev,
                &ranges,
                V4L2_CID_AUTO_WHITE_BALANCE,
                true,
                "auto_white_balance",
                false,
            )
        } else {
            ControlOutcome::Skipped
        });

        report.auto_gain = Some(if profile.auto_gain {
            set_boolean(
                dev,
                &ranges,
                V4L2_CID_AUTOGAIN,
                true,
                "auto_gain",
                false,
            )
        } else {
            ControlOutcome::Skipped
        });

        // --- Subject-friendly biases ---
        // For the controls we *might* bias, reset to driver default
        // when no bias is requested. V4L2 controls are persistent
        // across sessions: a prior run that wrote backlight=MAX or
        // brightness=+13 leaves those values stuck until something
        // (kernel reload, another app, us) writes them back. If the
        // user has turned a bias off, they want the camera default
        // — not "whatever nailbite set yesterday."
        report.backlight_compensation = Some(if profile.backlight_compensation_max {
            set_to_max(
                dev,
                &ranges,
                V4L2_CID_BACKLIGHT_COMPENSATION,
                "backlight_compensation",
                true,
            )
        } else {
            reset_to_default(dev, &ranges, V4L2_CID_BACKLIGHT_COMPENSATION, "backlight_compensation")
        });

        report.brightness = Some(match profile.brightness_fraction {
            Some(fraction) => set_to_fraction(
                dev,
                &ranges,
                V4L2_CID_BRIGHTNESS,
                fraction,
                "brightness",
                false,
            ),
            None => reset_to_default(dev, &ranges, V4L2_CID_BRIGHTNESS, "brightness"),
        });

        report.contrast = Some(match profile.contrast_fraction {
            Some(fraction) => set_to_fraction(
                dev,
                &ranges,
                V4L2_CID_CONTRAST,
                fraction,
                "contrast",
                false,
            ),
            None => reset_to_default(dev, &ranges, V4L2_CID_CONTRAST, "contrast"),
        });

        info!(
            ?report.gamma_reset,
            ?report.auto_exposure,
            ?report.exposure_auto_priority,
            ?report.auto_white_balance,
            ?report.auto_gain,
            ?report.backlight_compensation,
            ?report.brightness,
            ?report.contrast,
            "Camera control biases applied"
        );

        report
    }

    fn description_to_range(d: &Description) -> ControlRange {
        ControlRange {
            min: d.minimum,
            max: d.maximum,
            step: d.step,
            default: d.default,
        }
    }

    /// Try FULL_AUTO first, then APERTURE_PRIORITY. Real-world UVC
    /// webcams interpret APERTURE_PRIORITY inconsistently — on several
    /// firmwares it means "exposure stays at its last manual value,
    /// only the iris adapts" rather than tracking scene brightness, so
    /// the image stops responding to changing light. FULL_AUTO is the
    /// value cameras reliably interpret as "adapt to the scene." If
    /// the driver rejects FULL_AUTO we fall back to APERTURE_PRIORITY
    /// because some firmwares only expose the {MANUAL, APERTURE_PRIORITY}
    /// subset of the menu.
    fn apply_auto_exposure(
        dev: &Device,
        ranges: &HashMap<u32, ControlRange>,
    ) -> ControlOutcome {
        if !ranges.contains_key(&V4L2_CID_EXPOSURE_AUTO) {
            debug!(label = "exposure_auto", "Control unsupported; skipping");
            return ControlOutcome::Unsupported;
        }
        if let Err(e) = dev.set_control(Control {
            id: V4L2_CID_EXPOSURE_AUTO,
            value: ControlValue::Integer(EXPOSURE_AUTO_FULL_AUTO),
        }) {
            debug!(
                error = %e,
                "exposure_auto=full_auto rejected; trying aperture_priority fallback"
            );
            if let Err(e2) = dev.set_control(Control {
                id: V4L2_CID_EXPOSURE_AUTO,
                value: ControlValue::Integer(EXPOSURE_AUTO_APERTURE_PRIORITY),
            }) {
                warn!(
                    error = %e2,
                    "Could not enable auto-exposure — image won't adapt to brightness changes"
                );
                return ControlOutcome::Failed;
            }
            info!("Camera auto-exposure enabled (aperture priority — full auto unavailable)");
            return ControlOutcome::Applied(EXPOSURE_AUTO_APERTURE_PRIORITY);
        }
        info!("Camera auto-exposure enabled (full auto)");
        ControlOutcome::Applied(EXPOSURE_AUTO_FULL_AUTO)
    }

    /// Write a control back to the driver-reported `default` value.
    /// Used for controls that another app may have left in a stuck
    /// state (gamma is the canonical offender). Treated as
    /// `Unsupported` if the driver doesn't expose the control.
    fn reset_to_default(
        dev: &Device,
        ranges: &HashMap<u32, ControlRange>,
        id: u32,
        label: &str,
    ) -> ControlOutcome {
        let Some(range) = ranges.get(&id) else {
            debug!(label, "Control unsupported; skipping reset");
            return ControlOutcome::Unsupported;
        };
        let value = range.default;
        match dev.set_control(Control {
            id,
            value: ControlValue::Integer(value),
        }) {
            Ok(()) => {
                info!(label, value, "Camera control reset to driver default");
                ControlOutcome::Applied(value)
            }
            Err(e) => {
                warn!(error = %e, label, value, "Reset to driver default rejected");
                ControlOutcome::Failed
            }
        }
    }

    fn set_boolean(
        dev: &Device,
        ranges: &HashMap<u32, ControlRange>,
        id: u32,
        value: bool,
        label: &str,
        load_bearing: bool,
    ) -> ControlOutcome {
        if !ranges.contains_key(&id) {
            debug!(label, "Control unsupported; skipping");
            return ControlOutcome::Unsupported;
        }
        match dev.set_control(Control {
            id,
            value: ControlValue::Boolean(value),
        }) {
            Ok(()) => ControlOutcome::Applied(i64::from(value)),
            Err(e) => {
                if load_bearing {
                    warn!(error = %e, label, "Control set rejected (load-bearing)");
                } else {
                    debug!(error = %e, label, "Control set rejected");
                }
                ControlOutcome::Failed
            }
        }
    }

    fn set_to_max(
        dev: &Device,
        ranges: &HashMap<u32, ControlRange>,
        id: u32,
        label: &str,
        load_bearing: bool,
    ) -> ControlOutcome {
        let Some(range) = ranges.get(&id) else {
            debug!(label, "Control unsupported; skipping");
            return ControlOutcome::Unsupported;
        };
        let value = range.max;
        match dev.set_control(Control {
            id,
            value: ControlValue::Integer(value),
        }) {
            Ok(()) => {
                info!(label, value, "Camera control set to max");
                ControlOutcome::Applied(value)
            }
            Err(e) => {
                if load_bearing {
                    warn!(error = %e, label, value, "Control set rejected (load-bearing)");
                } else {
                    debug!(error = %e, label, value, "Control set rejected");
                }
                ControlOutcome::Failed
            }
        }
    }

    fn set_to_fraction(
        dev: &Device,
        ranges: &HashMap<u32, ControlRange>,
        id: u32,
        fraction: f32,
        label: &str,
        load_bearing: bool,
    ) -> ControlOutcome {
        let Some(range) = ranges.get(&id) else {
            debug!(label, "Control unsupported; skipping");
            return ControlOutcome::Unsupported;
        };
        if range.max <= range.min {
            debug!(label, ?range, "Control range degenerate; leaving at default");
            return ControlOutcome::Failed;
        }
        let value = compute_stepped_value(range.min, range.max, range.step, fraction);
        match dev.set_control(Control {
            id,
            value: ControlValue::Integer(value),
        }) {
            Ok(()) => {
                info!(
                    label,
                    value,
                    min = range.min,
                    max = range.max,
                    default = range.default,
                    "Camera control biased toward subject"
                );
                ControlOutcome::Applied(value)
            }
            Err(e) => {
                if load_bearing {
                    warn!(error = %e, label, value, "Control set rejected (load-bearing)");
                } else {
                    debug!(error = %e, label, value, "Control set rejected");
                }
                ControlOutcome::Failed
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fraction_zero_returns_min() {
        assert_eq!(compute_stepped_value(0, 100, 1, 0.0), 0);
        assert_eq!(compute_stepped_value(-50, 50, 1, 0.0), -50);
    }

    #[test]
    fn fraction_one_returns_max() {
        assert_eq!(compute_stepped_value(0, 100, 1, 1.0), 100);
        assert_eq!(compute_stepped_value(-50, 50, 1, 1.0), 50);
    }

    #[test]
    fn fraction_half_returns_midpoint() {
        assert_eq!(compute_stepped_value(0, 100, 1, 0.5), 50);
        assert_eq!(compute_stepped_value(72, 500, 1, 0.5), 286);
    }

    #[test]
    fn fraction_60_percent() {
        // The brightness_fraction default. 60% between 0..=255 is 153.
        assert_eq!(compute_stepped_value(0, 255, 1, 0.60), 153);
    }

    #[test]
    fn step_alignment() {
        // 70% of [0,100] step=10 rounds to 70.
        assert_eq!(compute_stepped_value(0, 100, 10, 0.70), 70);
        // 73% of [0,100] step=10 rounds to 70 (nearest step from min).
        assert_eq!(compute_stepped_value(0, 100, 10, 0.73), 70);
        // 75% of [0,100] step=10 rounds to 80 (banker's rounding edge —
        // f64 round() is round-half-away-from-zero).
        assert_eq!(compute_stepped_value(0, 100, 10, 0.78), 80);
    }

    #[test]
    fn step_alignment_with_nonzero_min() {
        // Stepping is anchored at `min`, not zero, so a min=1, step=2
        // range only produces odd numbers.
        assert_eq!(compute_stepped_value(1, 11, 2, 0.0), 1);
        assert_eq!(compute_stepped_value(1, 11, 2, 1.0), 11);
        assert_eq!(compute_stepped_value(1, 11, 2, 0.5), 7);
    }

    #[test]
    fn fraction_clamped_outside_range() {
        // Out-of-range fractions clamp into [0,1] before scaling.
        assert_eq!(compute_stepped_value(0, 100, 1, -2.0), 0);
        assert_eq!(compute_stepped_value(0, 100, 1, 5.0), 100);
    }

    #[test]
    fn nan_returns_min() {
        assert_eq!(compute_stepped_value(0, 100, 1, f32::NAN), 0);
    }

    #[test]
    fn degenerate_range_returns_min() {
        assert_eq!(compute_stepped_value(10, 10, 1, 0.5), 10);
        assert_eq!(compute_stepped_value(50, 10, 1, 0.5), 50);
    }

    #[test]
    fn large_step_does_not_overflow() {
        // 128-step range still clamps inside [min,max].
        let value = compute_stepped_value(0, 1000, 128, 0.5);
        assert!((0..=1000).contains(&value));
        // 0 + step*round((500)/128) = 0 + 128*4 = 512.
        assert_eq!(value, 512);
    }

    #[test]
    fn result_stays_within_bounds_for_all_fractions() {
        for percent in 0..=100 {
            let f = percent as f32 / 100.0;
            let v = compute_stepped_value(-100, 100, 7, f);
            assert!((-100..=100).contains(&v), "fraction={f} produced {v}");
        }
    }
}
