// Platform abstractions for cross-platform support

// Unified implementation using tray-icon + muda + enigo (all platforms)
mod unified;
pub use unified::{start_tray, type_text_unified, UnifiedTrayHandle};

// Legacy platform-specific modules (kept for reference/fallback)
#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub use linux::{get_user_socket_path, type_text as type_text_ydotool};

#[cfg(target_os = "macos")]
mod macos;

#[cfg(target_os = "windows")]
mod windows;

use crate::processors::VoiceMode;
use std::sync::{Arc, Mutex};

/// Tray handle abstraction for cross-platform tray management
pub trait TrayHandle: Send {
    fn update_mode(&self, mode: &VoiceMode);
}

/// Configuration for system tray
pub struct TrayConfig {
    pub enabled: Arc<Mutex<bool>>,
    pub paused: Arc<Mutex<bool>>,
    pub current_mode: Arc<Mutex<VoiceMode>>,
    pub assistant_enabled: bool,
    pub base_url: String,
}

/// Stub tray handle for platforms without tray or when tray is disabled
pub struct NoopTrayHandle;

impl TrayHandle for NoopTrayHandle {
    fn update_mode(&self, _mode: &VoiceMode) {}
}

// Non-Linux platforms don't have ydotool socket
#[cfg(not(target_os = "linux"))]
pub fn get_user_socket_path() -> String {
    String::new()
}
