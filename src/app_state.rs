// Centralized application state management
// Single source of truth for all state mutations

use crate::processors::VoiceMode;
use crate::tui::AppState as TuiState;
use std::sync::{Arc, Mutex};

/// Sync state bidirectionally with TUI and tray
/// Returns true if any state changed
pub fn sync_state(
    enabled: &Arc<Mutex<bool>>,
    paused: &Arc<Mutex<bool>>,
    output_enabled: &Arc<Mutex<bool>>,
    mode: &Arc<Mutex<VoiceMode>>,
    tui_state: &Option<Arc<Mutex<TuiState>>>,
    tray_counter: &Option<Arc<Mutex<u32>>>,
    last_enabled: &mut bool,
    last_paused: &mut bool,
    last_output_enabled: &mut bool,
    last_mode: &mut VoiceMode,
) -> bool {
    let mut changed = false;

    if let Some(ref state) = tui_state {
        if let Ok(mut s) = state.lock() {
            // Sync enabled state
            let mut e = enabled.lock().unwrap();
            if s.is_enabled != *e {
                if s.is_enabled != *last_enabled {
                    // TUI changed -> update main & tray
                    *e = s.is_enabled;
                    *last_enabled = s.is_enabled;
                    changed = true;
                } else {
                    // Main/tray changed -> update TUI
                    s.is_enabled = *e;
                    *last_enabled = *e;
                }
            } else {
                *last_enabled = *e;
            }
            drop(e);

            // Sync output enabled state
            let mut o = output_enabled.lock().unwrap();
            if s.output_enabled != *o {
                if s.output_enabled != *last_output_enabled {
                    // TUI changed -> update main & tray
                    *o = s.output_enabled;
                    *last_output_enabled = s.output_enabled;
                    changed = true;
                } else {
                    // Main/tray changed -> update TUI
                    s.output_enabled = *o;
                    *last_output_enabled = *o;
                }
            } else {
                *last_output_enabled = *o;
            }
            drop(o);

            // Sync pause state
            let mut p = paused.lock().unwrap();
            if s.is_paused != *p {
                if s.is_paused != *last_paused {
                    // TUI changed -> update main & tray
                    *p = s.is_paused;
                    *last_paused = s.is_paused;
                    changed = true;
                } else {
                    // Main/tray changed -> update TUI
                    s.is_paused = *p;
                    *last_paused = *p;
                }
            } else {
                *last_paused = *p;
            }
            drop(p);

            // Sync mode
            let mut m = mode.lock().unwrap();
            if !matches_mode(&s.mode, &*m) {
                if !matches_mode(&s.mode, last_mode) {
                    // TUI changed -> update main & tray
                    *m = s.mode.clone();
                    *last_mode = s.mode.clone();
                    changed = true;
                } else {
                    // Main/tray changed -> update TUI
                    s.mode = m.clone();
                    *last_mode = m.clone();
                }
            } else {
                *last_mode = m.clone();
            }
        }
    }

    // Update tray icon if state changed
    if changed {
        if let Some(ref counter) = tray_counter {
            if let Ok(mut count) = counter.lock() {
                *count = count.wrapping_add(1);
            }
        }
    }

    changed
}

/// Helper to compare modes (ignores context/language fields)
fn matches_mode(a: &VoiceMode, b: &VoiceMode) -> bool {
    match (a, b) {
        (VoiceMode::Dictation, VoiceMode::Dictation) => true,
        (VoiceMode::Assistant { .. }, VoiceMode::Assistant { .. }) => true,
        (VoiceMode::Code { .. }, VoiceMode::Code { .. }) => true,
        (VoiceMode::Command, VoiceMode::Command) => true,
        _ => false,
    }
}

/// Helper to print mode change with privacy warning
pub fn print_mode_change(mode: &VoiceMode, base_url: &str) {
    // Detect if backend URL is cloud-based by checking for localhost indicators
    // Local: localhost, 127.0.0.1, 0.0.0.0
    // Cloud: Any other URL (OpenAI, Anthropic, ElevenLabs, etc.)
    let is_cloud = !base_url.contains("localhost")
        && !base_url.contains("127.0.0.1")
        && !base_url.contains("0.0.0.0");

    match mode {
        VoiceMode::Dictation => {
            println!("\n📝 Mode: Dictation\n");
        }
        VoiceMode::Assistant { .. } => {
            println!("\n🤖 Mode: Assistant");
            if is_cloud {
                println!("⚠️  Using cloud AI: {}", base_url);
            }
            println!();
        }
        VoiceMode::Code { .. } => {
            println!("\n💻 Mode: Code");
            if is_cloud {
                println!("⚠️  Using cloud AI: {}", base_url);
            }
            println!();
        }
        VoiceMode::Command => {
            println!("\n⌨️  Mode: Command (Shell)\n");
        }
    }
}
