// Cross-platform notification sounds using programmatic tone generation
// Uses rodio's built-in sources - no external audio files needed

use rodio::source::Source;
use rodio::{OutputStream, Sink};
use std::time::Duration;

/// A simple sine wave source for generating tones
struct SineWave {
    freq: f32,
    sample_rate: u32,
    sample_idx: u64,
    duration_samples: u64,
}

impl SineWave {
    fn new(freq: f32, duration_ms: u64, sample_rate: u32) -> Self {
        Self {
            freq,
            sample_rate,
            sample_idx: 0,
            duration_samples: (sample_rate as u64 * duration_ms) / 1000,
        }
    }
}

impl Iterator for SineWave {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.sample_idx >= self.duration_samples {
            return None;
        }
        let t = self.sample_idx as f32 / self.sample_rate as f32;
        self.sample_idx += 1;

        // Apply envelope to avoid clicks (fade in/out)
        let envelope = {
            let fade_samples = self.sample_rate as u64 / 100; // 10ms fade
            let idx = self.sample_idx;
            let remaining = self.duration_samples - idx;
            if idx < fade_samples {
                idx as f32 / fade_samples as f32
            } else if remaining < fade_samples {
                remaining as f32 / fade_samples as f32
            } else {
                1.0
            }
        };

        Some((t * self.freq * 2.0 * std::f32::consts::PI).sin() * 0.1 * envelope)
    }
}

impl Source for SineWave {
    fn current_frame_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> u16 {
        1
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<Duration> {
        Some(Duration::from_millis(
            (self.duration_samples * 1000) / self.sample_rate as u64,
        ))
    }
}

/// Play a low tone for pause (descending)
pub fn play_pause() {
    std::thread::spawn(|| {
        if let Ok((_stream, handle)) = OutputStream::try_default() {
            if let Ok(sink) = Sink::try_new(&handle) {
                // Descending two-tone: high to low
                let tone1 = SineWave::new(880.0, 80, 44100); // A5
                let tone2 = SineWave::new(440.0, 120, 44100); // A4
                sink.append(tone1);
                sink.append(tone2);
                sink.sleep_until_end();
            }
        }
        // Fail silently if audio output unavailable
    });
}

/// Play a high tone for resume (ascending)
pub fn play_resume() {
    std::thread::spawn(|| {
        if let Ok((_stream, handle)) = OutputStream::try_default() {
            if let Ok(sink) = Sink::try_new(&handle) {
                // Ascending two-tone: low to high
                let tone1 = SineWave::new(440.0, 80, 44100); // A4
                let tone2 = SineWave::new(880.0, 120, 44100); // A5
                sink.append(tone1);
                sink.append(tone2);
                sink.sleep_until_end();
            }
        }
        // Fail silently if audio output unavailable
    });
}

/// Play a chord for mode change
pub fn play_mode_change() {
    std::thread::spawn(|| {
        if let Ok((_stream, handle)) = OutputStream::try_default() {
            if let Ok(sink) = Sink::try_new(&handle) {
                // Three quick tones (C-E-G arpeggio)
                let tone1 = SineWave::new(523.25, 60, 44100); // C5
                let tone2 = SineWave::new(659.25, 60, 44100); // E5
                let tone3 = SineWave::new(783.99, 100, 44100); // G5
                sink.append(tone1);
                sink.append(tone2);
                sink.append(tone3);
                sink.sleep_until_end();
            }
        }
        // Fail silently if audio output unavailable
    });
}

/// Play error/warning sound
#[allow(dead_code)]
pub fn play_error() {
    std::thread::spawn(|| {
        if let Ok((_stream, handle)) = OutputStream::try_default() {
            if let Ok(sink) = Sink::try_new(&handle) {
                // Low buzzy tone
                let tone = SineWave::new(220.0, 200, 44100); // A3
                sink.append(tone);
                sink.sleep_until_end();
            }
        }
    });
}
