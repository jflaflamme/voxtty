// Screen capture for Screen mode: grab the focused window as exact text (when
// it's a terminal we can query) or as a downscaled screenshot for a vision model.
use anyhow::{Context, Result};
use base64::Engine;
use std::process::Command;

/// Captured screen content: exact text, or a base64 data-URI image.
pub enum ScreenCapture {
    Text(String),
    Image(String),
}

/// Max width (px) for the screenshot sent to the vision model. Full-res images
/// blow the vision token budget (the model returns nothing); ~1280 is plenty.
const SCREENSHOT_MAX_WIDTH: u32 = 1280;

/// Capture the focused window. Prefers exact terminal text (kitty), otherwise a
/// downscaled screenshot.
pub fn capture() -> Result<ScreenCapture> {
    let class = focused_window_class().unwrap_or_default();
    if class.to_lowercase().contains("kitty") {
        if let Some(text) = kitty_text() {
            if !text.trim().is_empty() {
                return Ok(ScreenCapture::Text(truncate_text(&text)));
            }
        }
        // Remote control not reachable — fall through to a screenshot.
    }
    screenshot_data_uri().map(ScreenCapture::Image)
}

/// Active window class via hyprctl (best-effort).
fn focused_window_class() -> Option<String> {
    let out = Command::new("hyprctl").args(["activewindow", "-j"]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    v.get("class").and_then(|c| c.as_str()).map(|s| s.to_string())
}

/// Exact terminal text from kitty's remote control. Requires `allow_remote_control`
/// plus a reachable socket: tries $KITTY_SOCKET, then the default unix:/tmp/kitty.sock,
/// then a plain `kitty @` (works when launched inside kitty).
fn kitty_text() -> Option<String> {
    let socket = std::env::var("KITTY_SOCKET").ok();
    let mut attempts: Vec<Vec<String>> = Vec::new();
    if let Some(s) = socket {
        attempts.push(vec!["@".into(), "--to".into(), s, "get-text".into()]);
    }
    attempts.push(vec![
        "@".into(),
        "--to".into(),
        "unix:/tmp/kitty.sock".into(),
        "get-text".into(),
    ]);
    attempts.push(vec!["@".into(), "get-text".into()]);

    for args in attempts {
        if let Ok(out) = Command::new("kitty").args(&args).output() {
            if out.status.success() {
                let text = String::from_utf8_lossy(&out.stdout).to_string();
                if !text.trim().is_empty() {
                    return Some(text);
                }
            }
        }
    }
    None
}

/// Keep the most recent ~8000 chars of terminal text (the relevant tail), so a
/// long scrollback doesn't blow the prompt.
fn truncate_text(text: &str) -> String {
    const MAX: usize = 8000;
    if text.len() <= MAX {
        return text.to_string();
    }
    let start = text.len() - MAX;
    // Snap to a char boundary.
    let start = (start..text.len()).find(|i| text.is_char_boundary(*i)).unwrap_or(text.len());
    format!("…(earlier output trimmed)\n{}", &text[start..])
}

/// Capture the screen with grim, downscale to a JPEG with ImageMagick, and return
/// a base64 `data:` URI suitable for an OpenAI-compatible image_url part.
fn screenshot_data_uri() -> Result<String> {
    let dir = std::env::temp_dir();
    let png = dir.join("voxtty_screen.png");
    let jpg = dir.join("voxtty_screen.jpg");

    let grim = Command::new("grim")
        .arg(&png)
        .output()
        .context("failed to run grim (is it installed?)")?;
    if !grim.status.success() {
        anyhow::bail!("grim failed: {}", String::from_utf8_lossy(&grim.stderr));
    }

    // Downscale + JPEG-compress. `magick` (IM7) with a `convert` fallback.
    let resize = format!("{}x", SCREENSHOT_MAX_WIDTH);
    let im_args = [
        png.to_string_lossy().to_string(),
        "-resize".into(),
        resize,
        "-quality".into(),
        "80".into(),
        jpg.to_string_lossy().to_string(),
    ];
    let ran = Command::new("magick")
        .args(&im_args)
        .output()
        .or_else(|_| Command::new("convert").args(&im_args).output())
        .context("failed to run ImageMagick (magick/convert) to downscale screenshot")?;
    if !ran.status.success() {
        anyhow::bail!("image downscale failed: {}", String::from_utf8_lossy(&ran.stderr));
    }

    let bytes = std::fs::read(&jpg).context("failed to read downscaled screenshot")?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(format!("data:image/jpeg;base64,{}", b64))
}
