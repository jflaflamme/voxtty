// macOS-specific platform implementations
use crate::processors::VoiceMode;
use crate::platform::{TrayConfig, TrayHandle};
use anyhow::Result;
use std::sync::{Arc, Mutex};

use muda::{Menu, MenuItem, PredefinedMenuItem, MenuEvent};
use tray_icon::{TrayIcon, TrayIconBuilder, Icon};

/// macOS tray handle
pub struct MacOSTrayHandle {
    _tray: TrayIcon,
    enabled: Arc<Mutex<bool>>,
    current_mode: Arc<Mutex<VoiceMode>>,
    toggle_id: muda::MenuId,
    dictation_id: muda::MenuId,
    assistant_id: muda::MenuId,
    code_id: muda::MenuId,
}

impl TrayHandle for MacOSTrayHandle {
    fn update_mode(&self, _mode: &VoiceMode) {
        // Menu updates handled via MenuEvent
    }
}

/// Start the macOS system tray
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

    // Create icon (use a default icon for now)
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
    let enabled_clone = enabled.clone();
    let mode_clone = current_mode.clone();
    std::thread::spawn(move || {
        loop {
            if let Ok(event) = MenuEvent::receiver().recv() {
                if event.id == toggle_id {
                    let mut en = enabled_clone.lock().unwrap();
                    *en = !*en;
                    println!("\n🎤 Voice typing {}\n", if *en { "ENABLED ✓" } else { "DISABLED ✗" });
                } else if event.id == dictation_id {
                    let mut mode = mode_clone.lock().unwrap();
                    *mode = VoiceMode::Dictation;
                    crate::print_mode_change(&VoiceMode::Dictation, &base_url);
                } else if event.id == assistant_id {
                    let new_mode = VoiceMode::Assistant { context: Vec::new() };
                    let mut mode = mode_clone.lock().unwrap();
                    *mode = new_mode.clone();
                    crate::print_mode_change(&new_mode, &base_url);
                } else if event.id == code_id {
                    let new_mode = VoiceMode::Code { language: None };
                    let mut mode = mode_clone.lock().unwrap();
                    *mode = new_mode.clone();
                    crate::print_mode_change(&new_mode, &base_url);
                } else if event.id == quit_id {
                    std::process::exit(0);
                }
            }
        }
    });

    Some(Box::new(MacOSTrayHandle {
        _tray: tray,
        enabled,
        current_mode,
        toggle_id: muda::MenuId::new("toggle"),
        dictation_id: muda::MenuId::new("dictation"),
        assistant_id: muda::MenuId::new("assistant"),
        code_id: muda::MenuId::new("code"),
    }))
}

/// Create a simple default icon
fn create_default_icon() -> Option<Icon> {
    // Create a simple 32x32 microphone-like icon (RGBA)
    let size = 32;
    let mut rgba = vec![0u8; size * size * 4];

    // Draw a simple circle in the center
    for y in 0..size {
        for x in 0..size {
            let dx = x as i32 - size as i32 / 2;
            let dy = y as i32 - size as i32 / 2;
            let dist = ((dx * dx + dy * dy) as f32).sqrt();

            let idx = (y * size + x) * 4;
            if dist < 10.0 {
                // Inner circle - microphone color
                rgba[idx] = 100;     // R
                rgba[idx + 1] = 100; // G
                rgba[idx + 2] = 100; // B
                rgba[idx + 3] = 255; // A
            } else if dist < 12.0 {
                // Border
                rgba[idx] = 50;      // R
                rgba[idx + 1] = 50;  // G
                rgba[idx + 2] = 50;  // B
                rgba[idx + 3] = 255; // A
            }
        }
    }

    Icon::from_rgba(rgba, size as u32, size as u32).ok()
}

/// Type text using enigo (cross-platform keyboard simulation)
pub fn type_text(text: &str) -> Result<()> {
    use enigo::{Enigo, KeyboardControllable};

    let mut enigo = Enigo::new();
    enigo.key_sequence(text);

    Ok(())
}

/// Get user home directory for config paths
pub fn get_user_socket_path() -> String {
    // macOS doesn't use ydotool socket
    String::new()
}
