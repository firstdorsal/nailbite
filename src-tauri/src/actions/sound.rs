//! Sound alert action using rodio.
//!
//! Plays a configurable WAV file in a loop until stopped.
//! Uses a builtin beep tone if no custom file is specified.

use std::io::Cursor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use rodio::{OutputStream, OutputStreamBuilder, Sink, Source};
use tracing::debug;

use crate::actions::types::Action;
use crate::detection::types::DetectionEvent;
use crate::errors::ActionError;

/// Generates a beep-silence pattern for looping alerts.
///
/// Produces `beep_ms` of sine wave followed by `silence_ms` of silence.
fn generate_beep_pattern(
    frequency: f32,
    beep_ms: u32,
    silence_ms: u32,
    sample_rate: u32,
) -> Vec<f32> {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let beep_samples = (f64::from(sample_rate) * f64::from(beep_ms) / 1000.0) as usize;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let silence_samples = (f64::from(sample_rate) * f64::from(silence_ms) / 1000.0) as usize;
    let total = beep_samples + silence_samples;
    let mut samples = Vec::with_capacity(total);

    for i in 0..beep_samples {
        let t = i as f32 / sample_rate as f32;
        let envelope = if i < 200 {
            i as f32 / 200.0
        } else if i > beep_samples.saturating_sub(200) {
            (beep_samples - i) as f32 / 200.0
        } else {
            1.0
        };
        samples.push((t * frequency * 2.0 * std::f32::consts::PI).sin() * envelope);
    }

    // Append silence.
    samples.resize(total, 0.0);
    samples
}

/// A simple sine wave source for rodio.
struct BeepSource {
    samples: Vec<f32>,
    position: usize,
    sample_rate: u32,
}

impl Iterator for BeepSource {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        let sample = self.samples.get(self.position).copied();
        if sample.is_some() {
            self.position += 1;
        }
        sample
    }
}

impl Source for BeepSource {
    fn current_span_len(&self) -> Option<usize> {
        Some(self.samples.len().saturating_sub(self.position))
    }

    fn channels(&self) -> u16 {
        1
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<Duration> {
        let remaining = self.samples.len().saturating_sub(self.position);
        Some(Duration::from_secs_f32(
            remaining as f32 / self.sample_rate as f32,
        ))
    }
}

pub struct SoundAction {
    file_path: Option<String>,
    volume: f32,
    repeat: bool,
    active: Arc<AtomicBool>,
    _stream: Option<OutputStream>,
    sink: Option<Sink>,
}

impl SoundAction {
    pub fn new(file_path: &str, volume: f32, repeat: bool) -> Self {
        let file_path = if file_path == "builtin" {
            None
        } else {
            Some(file_path.to_string())
        };

        Self {
            file_path,
            volume,
            repeat,
            active: Arc::new(AtomicBool::new(false)),
            _stream: None,
            sink: None,
        }
    }

    fn create_sink(&mut self) -> Result<(), ActionError> {
        let stream = OutputStreamBuilder::open_default_stream()
            .map_err(|e| ActionError::Sound(e.to_string()))?;
        let sink = Sink::connect_new(stream.mixer());
        sink.set_volume(self.volume);
        self._stream = Some(stream);
        self.sink = Some(sink);
        Ok(())
    }
}

impl Action for SoundAction {
    fn start(&mut self, event: &DetectionEvent) -> Result<(), ActionError> {
        if self.active.load(Ordering::Relaxed) {
            return Ok(());
        }

        debug!(bfrb_type = %event.bfrb_type, "Starting sound alert");

        self.create_sink()?;
        let Some(sink) = &self.sink else {
            return Err(ActionError::Sound("Failed to create audio sink".to_string()));
        };

        match &self.file_path {
            Some(path) => {
                let data = std::fs::read(path)
                    .map_err(|e| ActionError::Sound(format!("Failed to read {path}: {e}")))?;
                let cursor = Cursor::new(data);
                let source = rodio::Decoder::new(cursor)
                    .map_err(|e| ActionError::Sound(format!("Failed to decode {path}: {e}")))?;
                if self.repeat {
                    sink.append(source.repeat_infinite());
                } else {
                    sink.append(source);
                }
            }
            None => {
                let sample_rate = 44100;
                let samples = if self.repeat {
                    // Beep with pauses: 500ms beep, 500ms silence.
                    generate_beep_pattern(800.0, 500, 500, sample_rate)
                } else {
                    generate_beep_pattern(800.0, 500, 0, sample_rate)
                };
                let source = BeepSource {
                    samples,
                    position: 0,
                    sample_rate,
                };
                if self.repeat {
                    sink.append(source.repeat_infinite());
                } else {
                    sink.append(source);
                }
            }
        }

        self.active.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn stop(&mut self) -> Result<(), ActionError> {
        if !self.active.load(Ordering::Relaxed) {
            return Ok(());
        }

        debug!("Stopping sound alert");

        if let Some(sink) = self.sink.take() {
            sink.stop();
        }
        self._stream = None;
        self.active.store(false, Ordering::Relaxed);
        Ok(())
    }

    fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beep_pattern_produces_correct_length() {
        let samples = generate_beep_pattern(440.0, 100, 0, 44100);
        assert!(!samples.is_empty());
        assert_eq!(samples.len(), 4410);
    }

    #[test]
    fn beep_pattern_with_silence() {
        let samples = generate_beep_pattern(440.0, 100, 100, 44100);
        // 100ms beep + 100ms silence = 8820 samples.
        assert_eq!(samples.len(), 8820);
    }

    #[test]
    fn beep_source_iterates() {
        let samples = generate_beep_pattern(440.0, 50, 0, 44100);
        let len = samples.len();
        let source = BeepSource {
            samples,
            position: 0,
            sample_rate: 44100,
        };
        assert_eq!(source.channels(), 1);
        assert_eq!(source.sample_rate(), 44100);
        let collected: Vec<f32> = source.collect();
        assert_eq!(collected.len(), len);
    }

    #[test]
    fn sound_action_starts_inactive() {
        let action = SoundAction::new("builtin", 0.8, true);
        assert!(!action.is_active());
    }
}
