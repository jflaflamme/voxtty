// Unified cross-platform implementation using tray-icon + muda + enigo
// Works on Linux, macOS, and Windows

use crate::processors::VoiceMode;
use crate::platform::{TrayConfig, TrayHandle};
use anyhow::Result;
use std::sync::{Arc, Mutex};

use muda::{Menu, MenuItem, PredefinedMenuItem, MenuEvent};
use tray_icon::{TrayIcon, TrayIconBuilder, Icon};

/// Cross-platform tray handle using tray-icon
pub struct UnifiedTrayHandle {
    _tray: TrayIcon,
}

impl TrayHandle for UnifiedTrayHandle {
    fn update_mode(&self, _mode: &VoiceMode) {
        // Menu updates handled via MenuEvent receiver
    }
}

/// Start the system tray (works on all platforms)
pub fn start_tray(config: TrayConfig) -> Option<Box<dyn TrayHandle>> {
    // Build menu
    let menu = Menu::new();

    let toggle_item = MenuItem::new("Disable Voice Typing", true, None);
    let toggle_id = toggle_item.id().clone();
    menu.append(&toggle_item).ok()?;

    menu.append(&PredefinedMenuItem::separator()).ok()?;

    let dictation_item = MenuItem::new("● Dictation Mode", true, None);
    let dictation_id = dictation_item.id().clone();
    menu.append(&dictation_item).ok()?;

    let assistant_item = MenuItem::new("○ Assistant Mode", true, None);
    let assistant_id = assistant_item.id().clone();

    let code_item = MenuItem::new("○ Code Mode", true, None);
    let code_id = code_item.id().clone();

    if config.assistant_enabled {
        menu.append(&assistant_item).ok()?;
        menu.append(&code_item).ok()?;
    }

    menu.append(&PredefinedMenuItem::separator()).ok()?;

    let quit_item = MenuItem::new("Quit", true, None);
    let quit_id = quit_item.id().clone();
    menu.append(&quit_item).ok()?;

    // Create icon
    let icon = create_default_icon()?;

    // Build tray
    let tray = TrayIconBuilder::new()
        .with_menu(Box::new(menu))
        .with_tooltip("VoiceTypr")
        .with_icon(icon)
        .build()
        .ok()?;

    let enabled = config.enabled.clone();
    let current_mode = config.current_mode.clone();
    let base_url = config.base_url.clone();

    // Spawn event handler thread
    std::thread::spawn(move || {
        loop {
            if let Ok(event) = MenuEvent::receiver().recv() {
                if event.id == toggle_id {
                    let mut en = enabled.lock().unwrap();
                    *en = !*en;
                    println!("\n🎤 Voice typing {}\n", if *en { "ENABLED ✓" } else { "DISABLED ✗" });
                } else if event.id == dictation_id {
                    let mut mode = current_mode.lock().unwrap();
                    *mode = VoiceMode::Dictation;
                    crate::print_mode_change(&VoiceMode::Dictation, &base_url);
                } else if event.id == assistant_id {
                    let new_mode = VoiceMode::Assistant { context: Vec::new() };
                    let mut mode = current_mode.lock().unwrap();
                    *mode = new_mode.clone();
                    crate::print_mode_change(&new_mode, &base_url);
                } else if event.id == code_id {
                    let new_mode = VoiceMode::Code { language: None };
                    let mut mode = current_mode.lock().unwrap();
                    *mode = new_mode.clone();
                    crate::print_mode_change(&new_mode, &base_url);
                } else if event.id == quit_id {
                    std::process::exit(0);
                }
            }
        }
    });

    Some(Box::new(UnifiedTrayHandle { _tray: tray }))
}

/// Icon state for color selection
#[derive(Clone, Copy)]
pub enum IconState {
    Active,
    Paused,
    Disabled,
}

/// Create icon with specified state (32x32 RGBA)
pub fn create_icon(state: IconState) -> Option<Icon> {
    let size = 32usize;
    let mut rgba = vec![0u8; size * size * 4];

    // Choose color based on state
    let (r, g, b) = match state {
        IconState::Active => (76u8, 175, 80),     // Green
        IconState::Paused => (255, 165, 0),       // Orange
        IconState::Disabled => (128, 128, 128),   // Gray
    };

    // Draw microphone shape: circle with stand
    for y in 0..size {
        for x in 0..size {
            let cx = x as i32 - size as i32 / 2;
            let cy = y as i32 - size as i32 / 2;
            let dist = ((cx * cx + cy * cy) as f32).sqrt();

            // Draw microphone shape: filled circle with stand
            let in_circle = dist < 10.0;
            let in_stand = x >= 14 && x <= 17 && y >= 20 && y <= 26;
            let in_base = y >= 26 && y <= 29 && x >= 10 && x <= 21;

            let idx = (y * size + x) * 4;
            if in_circle || in_stand || in_base {
                rgba[idx] = r;       // R
                rgba[idx + 1] = g;   // G
                rgba[idx + 2] = b;   // B
                rgba[idx + 3] = 255; // A
            }
        }
    }

    Icon::from_rgba(rgba, size as u32, size as u32).ok()
}

/// Create a simple default icon (32x32 RGBA) - green for active
fn create_default_icon() -> Option<Icon> {
    create_icon(IconState::Active)
}

/// Type text using enigo (cross-platform)
pub fn type_text_unified(text: &str) -> Result<()> {
    use enigo::{Enigo, KeyboardControllable};

    let mut enigo = Enigo::new();
    enigo.key_sequence(text);

    Ok(())
}
