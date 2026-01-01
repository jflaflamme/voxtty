// Linux-specific platform implementations
use crate::processors::VoiceMode;
use crate::platform::{TrayConfig, TrayHandle};
use ksni::{Tray, TrayService};
use std::sync::{Arc, Mutex};
use std::thread;

/// Linux system tray using ksni (DBus StatusNotifierItem)
pub struct LinuxTray {
    enabled: Arc<Mutex<bool>>,
    paused: Arc<Mutex<bool>>,
    current_mode: Arc<Mutex<VoiceMode>>,
    assistant_enabled: bool,
    base_url: String,
    update_counter: Arc<Mutex<u32>>,
}

impl Tray for LinuxTray {
    fn id(&self) -> String {
        env!("CARGO_PKG_NAME").into()
    }

    fn title(&self) -> String {
        let enabled = self.enabled.lock().unwrap();
        let paused = self.paused.lock().unwrap();
        let mode = self.current_mode.lock().unwrap();
        if *paused {
            "Voice Typing: PAUSED".into()
        } else if *enabled {
            format!("Voice Typing: ON ({:?})", mode)
        } else {
            "Voice Typing: OFF".into()
        }
    }

    fn icon_pixmap(&self) -> Vec<ksni::Icon> {
        let enabled = *self.enabled.lock().unwrap();
        let paused = *self.paused.lock().unwrap();
        let mode = self.current_mode.lock().unwrap().clone();

        // Get letter and base color for mode
        let (letter, mode_r, mode_g, mode_b) = match &mode {
            VoiceMode::Dictation => ('D', 76u8, 175, 80),        // Green
            VoiceMode::Assistant { .. } => ('A', 33, 150, 243),  // Blue
            VoiceMode::Code { .. } => ('C', 156, 39, 176),       // Purple
        };

        // Override color based on state
        let (r, g, b) = if !enabled {
            (128u8, 128, 128)  // Gray for disabled
        } else if paused {
            (255, 165, 0)      // Orange for paused
        } else {
            (mode_r, mode_g, mode_b)  // Mode color when active
        };

        // Create 22x22 icon with letter
        let size = 22i32;
        let mut data = Vec::with_capacity((size * size * 4) as usize);

        // Simple 5x7 pixel font for D, A, C
        let letter_pixels: [[u8; 5]; 7] = match letter {
            'D' => [
                [1, 1, 1, 0, 0],
                [1, 0, 0, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 1, 0],
                [1, 1, 1, 0, 0],
            ],
            'A' => [
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1],
            ],
            'C' => [
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 1],
                [0, 1, 1, 1, 0],
            ],
            _ => [[0; 5]; 7],
        };

        for y in 0..size {
            for x in 0..size {
                let cx = x - size / 2;
                let cy = y - size / 2;
                let dist = ((cx * cx + cy * cy) as f32).sqrt();

                // Check if inside circle
                let in_circle = dist < 10.0;

                // Check if pixel is part of the letter (centered in circle)
                let letter_x = x - 8;  // Offset to center 5-wide letter
                let letter_y = y - 7;  // Offset to center 7-tall letter
                let in_letter = if letter_x >= 0 && letter_x < 5 && letter_y >= 0 && letter_y < 7 {
                    letter_pixels[letter_y as usize][letter_x as usize] == 1
                } else {
                    false
                };

                let (ar, ag, ab, aa) = if in_circle {
                    if in_letter {
                        (255u8, 255, 255, 255)  // White letter
                    } else {
                        (r, g, b, 255u8)  // Colored background
                    }
                } else {
                    (0, 0, 0, 0)  // Transparent outside
                };

                // ARGB32 network byte order
                data.push(aa);
                data.push(ar);
                data.push(ag);
                data.push(ab);
            }
        }

        vec![ksni::Icon {
            width: size,
            height: size,
            data,
        }]
    }

    fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
        use ksni::menu::*;
        let enabled = *self.enabled.lock().unwrap();
        let current_mode = self.current_mode.lock().unwrap().clone();

        let mut items = vec![
            StandardItem {
                label: if enabled { "Disable Voice Typing".into() } else { "Enable Voice Typing".into() },
                activate: Box::new(|this: &mut LinuxTray| {
                    let mut enabled = this.enabled.lock().unwrap();
                    *enabled = !*enabled;
                    println!("\n🎤 Voice typing {}\n", if *enabled { "ENABLED ✓" } else { "DISABLED ✗" });
                }),
                ..Default::default()
            }.into(),
        ];

        // Add mode selection if assistant is enabled
        if self.assistant_enabled {
            items.push(MenuItem::Separator);

            // Dictation mode
            items.push(StandardItem {
                label: if matches!(current_mode, VoiceMode::Dictation) { "● Dictation Mode".into() } else { "○ Dictation Mode".into() },
                activate: Box::new(|this: &mut LinuxTray| {
                    let mut mode = this.current_mode.lock().unwrap();
                    *mode = VoiceMode::Dictation;
                    drop(mode);
                    let mut count = this.update_counter.lock().unwrap();
                    *count = count.wrapping_add(1);
                    drop(count);
                    crate::print_mode_change(&VoiceMode::Dictation, &this.base_url);
                }),
                ..Default::default()
            }.into());

            // Assistant mode
            items.push(StandardItem {
                label: if matches!(current_mode, VoiceMode::Assistant { .. }) { "● Assistant Mode".into() } else { "○ Assistant Mode".into() },
                activate: Box::new(|this: &mut LinuxTray| {
                    let new_mode = VoiceMode::Assistant { context: Vec::new() };
                    let mut mode = this.current_mode.lock().unwrap();
                    *mode = new_mode.clone();
                    drop(mode);
                    let mut count = this.update_counter.lock().unwrap();
                    *count = count.wrapping_add(1);
                    drop(count);
                    crate::print_mode_change(&new_mode, &this.base_url);
                }),
                ..Default::default()
            }.into());

            // Code mode
            items.push(StandardItem {
                label: if matches!(current_mode, VoiceMode::Code { .. }) { "● Code Mode".into() } else { "○ Code Mode".into() },
                activate: Box::new(|this: &mut LinuxTray| {
                    let new_mode = VoiceMode::Code { language: None };
                    let mut mode = this.current_mode.lock().unwrap();
                    *mode = new_mode.clone();
                    drop(mode);
                    let mut count = this.update_counter.lock().unwrap();
                    *count = count.wrapping_add(1);
                    drop(count);
                    crate::print_mode_change(&new_mode, &this.base_url);
                }),
                ..Default::default()
            }.into());
        }

        items.push(MenuItem::Separator);
        items.push(StandardItem {
            label: "Quit".into(),
            activate: Box::new(|_this: &mut LinuxTray| {
                std::process::exit(0);
            }),
            ..Default::default()
        }.into());

        items
    }
}

/// Linux tray handle wrapper
pub struct LinuxTrayHandle {
    handle: ksni::Handle<LinuxTray>,
    update_counter: Arc<Mutex<u32>>,
}

impl TrayHandle for LinuxTrayHandle {
    fn update_mode(&self, _mode: &VoiceMode) {
        let mut count = self.update_counter.lock().unwrap();
        *count = count.wrapping_add(1);
        drop(count);
        self.handle.update(|_tray| {});
    }
}

/// Start the Linux system tray
pub fn start_tray(config: TrayConfig) -> Option<Box<dyn TrayHandle>> {
    let update_counter = Arc::new(Mutex::new(0u32));
    let update_counter_clone = update_counter.clone();

    let service = TrayService::new(LinuxTray {
        enabled: config.enabled,
        paused: config.paused,
        current_mode: config.current_mode,
        assistant_enabled: config.assistant_enabled,
        base_url: config.base_url,
        update_counter: update_counter_clone,
    });

    let handle = service.handle();

    thread::spawn(move || {
        let _ = service.run();
    });

    Some(Box::new(LinuxTrayHandle {
        handle,
        update_counter,
    }))
}

/// Get user ID for socket path detection
pub fn get_user_socket_path() -> String {
    let uid = unsafe { libc::getuid() };
    format!("/run/user/{}/.ydotool_socket", uid)
}

/// Type text using ydotool (Linux-specific)
pub fn type_text(text: &str, socket_path: &str) -> anyhow::Result<()> {
    use std::process::Command;

    Command::new("ydotool")
        .env("YDOTOOL_SOCKET", socket_path)
        .arg("type")
        .arg(text)
        .spawn()?;

    Ok(())
}
