// Shared control logic
// Eliminates code duplication between CLI, TUI, and main loop

use anyhow::{Context, Result};

/// Audio playback helper using rodio (in-memory, no external windows)
pub fn playback_audio(
    samples: &[i16],
    sample_rate: u32,
    channels: u16,
    _wait_for_completion: bool,
) -> Result<()> {
    use rodio::{buffer::SamplesBuffer, OutputStream, Sink};

    // Convert i16 samples to f32 for rodio
    let samples_f32: Vec<f32> = samples
        .iter()
        .map(|&s| s as f32 / i16::MAX as f32)
        .collect();

    // Create audio output stream
    let (_stream, stream_handle) =
        OutputStream::try_default().context("Failed to open audio output device")?;

    let sink = Sink::try_new(&stream_handle).context("Failed to create audio sink")?;

    // Create buffer source from samples
    let source = SamplesBuffer::new(channels, sample_rate, samples_f32);
    sink.append(source);

    // Wait for playback to complete
    sink.sleep_until_end();

    Ok(())
}
