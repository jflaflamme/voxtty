// Simplified Terminal User Interface for voxtty - Single Dashboard
use anyhow::Result;
use crossterm::{
    cursor::Show,
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Sparkline, Wrap},
    Frame, Terminal,
};
use std::collections::VecDeque;
use std::io;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::processors::VoiceMode;

/// Connection status for realtime transcription
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConnectionStatus {
    Disconnected,
    Connecting,
    Connected,
}

/// Processing status for conversation mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProcessingStatus {
    Idle,
    Transcribing,
    Thinking,
    #[allow(dead_code)]
    GeneratingResponse, // Reserved for future use - requires processor-level TUI state access
    PlayingAudio,
    Writing,
}

/// A single conversation entry (input, output)
#[derive(Debug, Clone)]
pub struct ConversationEntry {
    pub input: String,          // What the user said
    pub output: String,         // AI response or processed output
    pub conversation_id: usize, // ID to group related exchanges
}

/// Application state shared with main voxtty process
#[derive(Debug, Clone)]
pub struct AppState {
    pub mode: VoiceMode,
    pub backend: String,
    pub is_listening: bool,
    pub is_enabled: bool,
    pub is_paused: bool,
    pub last_input: String, // Raw transcription (what you said) - DEPRECATED, use conversation_history
    pub last_transcription: String, // Processed output (LLM response or raw in dictation) - DEPRECATED, use conversation_history
    pub last_transcription_time: Option<Instant>,
    pub conversation_history: VecDeque<ConversationEntry>, // Scrollable conversation history
    pub audio_level: f32,
    pub audio_history: VecDeque<u64>, // Rolling buffer of audio levels for Sparkline
    pub vad_active: bool,
    pub selected_device: Option<String>,
    pub should_exit: bool,
    pub echo_test_requested: bool,
    pub echo_test_status: String,
    pub output_enabled: bool,
    pub backend_switch_requested: bool,
    pub available_devices: Vec<String>,
    pub device_switch_requested: Option<String>,
    pub error_message: Option<String>,
    pub error_timestamp: Option<Instant>,
    pub bidirectional_enabled: bool,
    pub realtime_status: ConnectionStatus,
    pub partial_transcription: Option<String>,
    pub is_processing: bool, // LLM is thinking (DEPRECATED - use processing_status)
    pub processing_status: ProcessingStatus, // Granular processing status
    pub current_conversation_id: usize, // ID to track conversation boundaries
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            mode: VoiceMode::Dictation,
            backend: "whisper.cpp".to_string(),
            is_listening: false,
            is_enabled: true,
            is_paused: false,
            last_input: String::new(),
            last_transcription: String::new(),
            last_transcription_time: None,
            conversation_history: VecDeque::with_capacity(10), // Keep last 10 interactions
            audio_level: 0.0,
            audio_history: VecDeque::with_capacity(100), // Keep last 100 samples
            vad_active: false,
            selected_device: Some("Default Microphone".to_string()),
            should_exit: false,
            echo_test_requested: false,
            echo_test_status: String::new(),
            output_enabled: false,
            backend_switch_requested: false,
            available_devices: Vec::new(),
            device_switch_requested: None,
            error_message: None,
            error_timestamp: None,
            bidirectional_enabled: false,
            realtime_status: ConnectionStatus::Disconnected,
            partial_transcription: None,
            is_processing: false,
            processing_status: ProcessingStatus::Idle,
            current_conversation_id: 0,
        }
    }
}

/// Simple TUI Dashboard
pub struct TuiApp {
    state: Arc<Mutex<AppState>>,
    should_quit: bool,
    show_help: bool,
    show_device_selector: bool,
    device_list_state: ratatui::widgets::ListState,
    conversation_scroll: usize, // Scroll offset for conversation history (0 = bottom/latest)
}

impl TuiApp {
    pub fn new(state: Arc<Mutex<AppState>>) -> Self {
        Self {
            state,
            should_quit: false,
            show_help: false,
            show_device_selector: false,
            device_list_state: ratatui::widgets::ListState::default(),
            conversation_scroll: 0, // Start at bottom (latest messages)
        }
    }

    pub fn run(&mut self) -> Result<()> {
        // Set up panic hook to restore terminal on crash
        let original_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |panic_info| {
            let _ = cleanup_terminal();
            original_hook(panic_info);
        }));

        // Always restore terminal on function exit (even on panic/error)
        struct TerminalGuard;
        impl Drop for TerminalGuard {
            fn drop(&mut self) {
                let _ = cleanup_terminal();
            }
        }
        let _guard = TerminalGuard;

        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // Clear terminal on startup
        terminal.clear()?;

        self.run_loop(&mut terminal)
    }

    fn run_loop<B: ratatui::backend::Backend>(&mut self, terminal: &mut Terminal<B>) -> Result<()> {
        let mut last_draw = Instant::now();
        let draw_interval = Duration::from_millis(250); // Redraw at most 4 times per second

        loop {
            // Only redraw if enough time has passed
            let now = Instant::now();
            if now.duration_since(last_draw) >= draw_interval {
                terminal.draw(|f| self.render(f))?;
                last_draw = now;
            }

            // Poll for events with short timeout
            if event::poll(Duration::from_millis(50))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        self.handle_key(key.code);
                        // Redraw immediately after key press
                        terminal.draw(|f| self.render(f))?;
                        last_draw = Instant::now();
                    }
                }
            }

            if self.should_quit {
                break;
            }
        }
        Ok(())
    }

    fn handle_key(&mut self, key: KeyCode) {
        if self.show_device_selector {
            self.handle_device_selector_key(key);
            return;
        }

        match key {
            KeyCode::Char('q') | KeyCode::Esc => {
                if self.show_help {
                    self.show_help = false;
                } else {
                    self.should_quit = true;
                    if let Ok(mut state) = self.state.lock() {
                        state.should_exit = true;
                    }
                }
            }
            KeyCode::Char('?') | KeyCode::Char('h') => self.show_help = !self.show_help,
            KeyCode::Char('1') => {
                if let Ok(mut state) = self.state.lock() {
                    state.mode = VoiceMode::Dictation;
                    state.last_input.clear();
                    state.last_transcription.clear();
                }
            }
            KeyCode::Char('2') => {
                if let Ok(mut state) = self.state.lock() {
                    state.mode = VoiceMode::Assistant { context: vec![] };
                    state.last_input.clear();
                    state.last_transcription.clear();
                }
            }
            KeyCode::Char('3') => {
                if let Ok(mut state) = self.state.lock() {
                    state.mode = VoiceMode::Code { language: None };
                    state.last_input.clear();
                    state.last_transcription.clear();
                }
            }
            KeyCode::Char('4') => {
                if let Ok(mut state) = self.state.lock() {
                    state.mode = VoiceMode::Command;
                    state.last_input.clear();
                    state.last_transcription.clear();
                }
            }
            KeyCode::Char('p') | KeyCode::Char(' ') => {
                if let Ok(mut state) = self.state.lock() {
                    state.is_paused = !state.is_paused;
                }
            }
            KeyCode::Char('s') => {
                if let Ok(mut state) = self.state.lock() {
                    state.is_enabled = !state.is_enabled;

                    // Play sound feedback
                    if state.is_enabled {
                        crate::sounds::play_resume();
                    } else {
                        crate::sounds::play_pause();
                    }
                }
            }
            KeyCode::Char('e') => {
                if let Ok(mut state) = self.state.lock() {
                    state.echo_test_requested = true;
                    state.echo_test_status = "Echo test starting...".to_string();
                }
            }
            KeyCode::Char('o') => {
                if let Ok(mut state) = self.state.lock() {
                    state.output_enabled = !state.output_enabled;

                    // Play sound feedback
                    if state.output_enabled {
                        crate::sounds::play_resume();
                    } else {
                        crate::sounds::play_pause();
                    }
                }
            }
            KeyCode::Char('b') => {
                if let Ok(mut state) = self.state.lock() {
                    state.backend_switch_requested = true;
                }
            }
            KeyCode::Char('d') => {
                self.show_device_selector = true;
                // Initialize selection to current device if possible
                let state = self.state.lock().unwrap();
                if let Some(current) = &state.selected_device {
                    if let Some(pos) = state.available_devices.iter().position(|d| d == current) {
                        self.device_list_state.select(Some(pos));
                    } else {
                        self.device_list_state.select(Some(0));
                    }
                } else {
                    self.device_list_state.select(Some(0));
                }
            }
            KeyCode::Up => {
                // Scroll up through conversation history (older messages)
                let state = self.state.lock().unwrap();
                let history_len = state.conversation_history.len();
                if history_len > 0 {
                    self.conversation_scroll = (self.conversation_scroll + 1).min(history_len - 1);
                }
            }
            KeyCode::Down => {
                // Scroll down through conversation history (newer messages)
                if self.conversation_scroll > 0 {
                    self.conversation_scroll -= 1;
                }
            }
            _ => {}
        }
    }

    fn handle_device_selector_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Esc | KeyCode::Char('q') => {
                self.show_device_selector = false;
            }
            KeyCode::Up => {
                let state = self.state.lock().unwrap();
                let i = match self.device_list_state.selected() {
                    Some(i) => {
                        if i == 0 {
                            state.available_devices.len() - 1
                        } else {
                            i - 1
                        }
                    }
                    None => 0,
                };
                self.device_list_state.select(Some(i));
            }
            KeyCode::Down => {
                let state = self.state.lock().unwrap();
                let i = match self.device_list_state.selected() {
                    Some(i) => {
                        if i >= state.available_devices.len() - 1 {
                            0
                        } else {
                            i + 1
                        }
                    }
                    None => 0,
                };
                self.device_list_state.select(Some(i));
            }
            KeyCode::Enter => {
                let mut state = self.state.lock().unwrap();
                let selected_device = if let Some(selected_idx) = self.device_list_state.selected()
                {
                    state.available_devices.get(selected_idx).cloned()
                } else {
                    None
                };

                if let Some(device_name) = selected_device {
                    state.device_switch_requested = Some(device_name.clone());
                    state.selected_device = Some(device_name);
                }
                self.show_device_selector = false;
            }
            _ => {}
        }
    }

    fn render(&mut self, f: &mut Frame) {
        if self.show_help {
            self.render_help(f);
            return;
        }

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(10), // Header (ASCII Art + Status)
                Constraint::Fill(1),    // Main Content (Audio Visualizer) - Fills remaining space
                Constraint::Min(4),     // Last Transcription (dynamic, min 4 lines)
                Constraint::Length(3),  // Controls Row
            ])
            .split(f.area());

        self.render_header(f, chunks[0]);
        self.render_audio_visualizer(f, chunks[1]);
        self.render_last_transcription(f, chunks[2]);
        self.render_controls_row(f, chunks[3]);

        if self.show_device_selector {
            self.render_device_selector(f);
        }

        // Show error message overlay if present (and not too old)
        let error_to_show = {
            let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(ref error_msg) = state.error_message {
                if let Some(timestamp) = state.error_timestamp {
                    // Show error for 10 seconds
                    if timestamp.elapsed().as_secs() < 10 {
                        Some(error_msg.clone())
                    } else {
                        // Mark for clearing
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        };

        if let Some(error_msg) = error_to_show {
            self.render_error_overlay(f, &error_msg);
        } else {
            // Clear old error if it exists
            let mut state = self.state.lock().unwrap();
            if state.error_message.is_some() && state.error_timestamp.is_some() {
                if let Some(timestamp) = state.error_timestamp {
                    if timestamp.elapsed().as_secs() >= 10 {
                        state.error_message = None;
                        state.error_timestamp = None;
                    }
                }
            }
        }
    }

    fn render_device_selector(&mut self, f: &mut Frame) {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let area = centered_rect(60, 50, f.area());

        f.render_widget(Clear, area); // Clear background

        let items: Vec<ListItem> = state
            .available_devices
            .iter()
            .map(|i| {
                let style = if Some(i.clone()) == state.selected_device {
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                ListItem::new(Line::from(vec![
                    if Some(i.clone()) == state.selected_device {
                        Span::raw(" * ")
                    } else {
                        Span::raw("   ")
                    },
                    Span::styled(i, style),
                ]))
            })
            .collect();

        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Select Input Device (Enter to Select, Esc to Cancel) "),
            )
            .highlight_style(
                Style::default()
                    .bg(Color::Blue)
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol(" > ");

        f.render_stateful_widget(list, area, &mut self.device_list_state);
    }

    fn render_error_overlay(&self, f: &mut Frame, error_msg: &str) {
        let area = centered_rect(70, 20, f.area());

        f.render_widget(Clear, area); // Clear background

        // Format error message with word wrapping
        let error_text = vec![
            Line::from(""),
            Line::from(vec![Span::styled(
                "  ❌ ERROR  ",
                Style::default()
                    .fg(Color::White)
                    .bg(Color::Red)
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::from(""),
            Line::from(vec![Span::styled(
                error_msg,
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            )]),
            Line::from(""),
            Line::from(vec![Span::styled(
                "  This message will disappear in 10 seconds  ",
                Style::default().fg(Color::Gray),
            )]),
        ];

        let paragraph = Paragraph::new(error_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Red).add_modifier(Modifier::BOLD))
                    .title(" Error "),
            )
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true });

        f.render_widget(paragraph, area);
    }

    fn render_header(&self, f: &mut Frame, area: Rect) {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());

        let status_text = if !state.is_enabled {
            ("DISABLED", Color::DarkGray)
        } else if state.is_paused {
            ("PAUSED", Color::Red)
        } else if state.is_listening {
            ("LISTENING", Color::Green)
        } else {
            ("IDLE", Color::Gray)
        };

        // ASCII Art Header - "VOXTTY" with mode-based color gradient
        let primary_color = mode_color(&state.mode).fg.unwrap_or(Color::Cyan);

        let mut lines = vec![
            Line::from(vec![Span::styled(
                "██╗   ██╗ ██████╗ ██╗  ██╗████████╗████████╗██╗   ██╗",
                Style::default()
                    .fg(primary_color)
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::from(vec![Span::styled(
                "██║   ██║██╔═══██╗╚██╗██╔╝╚══██╔══╝╚══██╔══╝╚██╗ ██╔╝",
                Style::default()
                    .fg(primary_color)
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::from(vec![Span::styled(
                "██║   ██║██║   ██║ ╚███╔╝    ██║      ██║    ╚████╔╝ ",
                Style::default()
                    .fg(primary_color)
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::from(vec![Span::styled(
                "╚██╗ ██╔╝██║   ██║ ██╔██╗    ██║      ██║     ╚██╔╝  ",
                Style::default()
                    .fg(primary_color)
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::from(vec![Span::styled(
                " ╚████╔╝ ╚██████╔╝██╔╝ ██╗   ██║      ██║      ██║   ",
                Style::default()
                    .fg(primary_color)
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::from(vec![Span::styled(
                "  ╚═══╝   ╚═════╝ ╚═╝  ╚═╝   ╚═╝      ╚═╝      ╚═╝   ",
                Style::default()
                    .fg(primary_color)
                    .add_modifier(Modifier::BOLD),
            )]),
            Line::from(""), // Spacer
        ];

        let mode_span = Span::styled(
            format!(" {} ", mode_name(&state.mode).to_uppercase()),
            Style::default()
                .bg(mode_color(&state.mode).fg.unwrap_or(Color::White))
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        );

        let mut status_spans = vec![
            Span::styled(" STATUS: ", Style::default().fg(Color::Gray)),
            mode_span,
            Span::raw(" │ "),
            Span::styled(
                format!(" {} ", status_text.0),
                Style::default()
                    .bg(status_text.1)
                    .fg(Color::Black)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" │ "),
            Span::styled(&state.backend, Style::default().fg(Color::Yellow)),
        ];

        if let Some(device) = &state.selected_device {
            status_spans.push(Span::raw(" │ 🎤 "));
            status_spans.push(Span::styled(device, Style::default().fg(Color::White)));
        }

        if state.output_enabled {
            status_spans.push(Span::raw(" │ "));
            status_spans.push(Span::styled(
                "TYPE: ON",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ));
        }

        if state.backend.contains("Realtime") {
            status_spans.push(Span::raw(" │ "));
            match state.realtime_status {
                ConnectionStatus::Connected => {
                    status_spans.push(Span::styled(
                        "● CONNECTED",
                        Style::default().fg(Color::Green),
                    ));
                }
                ConnectionStatus::Connecting => {
                    status_spans.push(Span::styled(
                        "◐ CONNECTING",
                        Style::default().fg(Color::Yellow),
                    ));
                }
                ConnectionStatus::Disconnected => {
                    status_spans.push(Span::styled(
                        "◌ DISCONNECTED",
                        Style::default().fg(Color::Red),
                    ));
                }
            }
        }

        // Add version
        let version = env!("CARGO_PKG_VERSION");
        status_spans.push(Span::raw(" │ "));
        status_spans.push(Span::styled(
            format!("v{}", version),
            Style::default().fg(Color::DarkGray),
        ));

        lines.push(Line::from(status_spans));

        let paragraph = Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL))
            .alignment(Alignment::Center);

        f.render_widget(paragraph, area);
    }

    fn render_audio_visualizer(&self, f: &mut Frame, area: Rect) {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());

        let block = Block::default().borders(Borders::ALL).title(format!(
            " Live Audio Monitor (VAD: {}) ",
            if state.vad_active { "ON" } else { "OFF" }
        ));

        let inner_area = block.inner(area);
        f.render_widget(block, area);

        // Audio Sparkline
        let audio_data: Vec<u64> = state.audio_history.iter().copied().collect();
        // Dynamic max value for better visualization scaling
        let max_value = 100;

        let sparkline =
            Sparkline::default()
                .data(&audio_data)
                .max(max_value)
                .style(if state.vad_active {
                    Style::default().fg(Color::Green)
                } else {
                    Style::default().fg(Color::DarkGray)
                });

        f.render_widget(sparkline, inner_area);
    }

    fn render_last_transcription(&self, f: &mut Frame, area: Rect) {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());

        // Show echo test status if active
        if !state.echo_test_status.is_empty() {
            let content = vec![Line::from(Span::styled(
                &state.echo_test_status,
                Style::default().fg(Color::Yellow),
            ))];

            let widget = Paragraph::new(content)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Echo Test Status"),
                )
                .wrap(Wrap { trim: true });

            f.render_widget(widget, area);
            return;
        }

        // Show warning if voice detection is disabled
        if !state.is_enabled {
            let content = vec![
                Line::from(""),
                Line::from(""),
                Line::from(vec![Span::styled(
                    "   ⚠️  VOICE DETECTION DISABLED  ⚠️",
                    Style::default()
                        .fg(Color::Red)
                        .add_modifier(Modifier::BOLD | Modifier::SLOW_BLINK),
                )]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("   Press ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        "'s'",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        " to enable voice detection",
                        Style::default().fg(Color::Gray),
                    ),
                ]),
            ];

            let widget = Paragraph::new(content)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(" Conversation History | Bidirectional "),
                )
                .alignment(Alignment::Center)
                .wrap(Wrap { trim: true });

            f.render_widget(widget, area);
            return;
        }

        // Build conversation history display
        let history_len = state.conversation_history.len();
        let current_conversation_id = state.current_conversation_id;

        let content = if history_len == 0 && state.partial_transcription.is_none() {
            vec![Line::from(Span::styled(
                "No transcriptions yet",
                Style::default()
                    .fg(Color::Gray)
                    .add_modifier(Modifier::ITALIC),
            ))]
        } else {
            let mut lines = Vec::new();

            // Filter to only show current conversation (same conversation_id)
            let current_conversation: Vec<&ConversationEntry> = state
                .conversation_history
                .iter()
                .filter(|entry| entry.conversation_id == current_conversation_id)
                .collect();

            // Show conversation entries from current conversation only
            for (idx, entry) in current_conversation.iter().enumerate() {
                let is_latest_entry = idx == current_conversation.len() - 1;

                // Determine if this is the currently playing TTS message
                let is_playing_tts = is_latest_entry
                    && state.is_processing
                    && matches!(state.processing_status, ProcessingStatus::PlayingAudio)
                    && entry.output.starts_with("🔊 ");

                // Show input if different from output (Assistant/Code/Command modes)
                if !entry.input.is_empty() && entry.input != entry.output {
                    lines.push(Line::from(vec![
                        Span::styled(
                            "  YOU  ",
                            Style::default()
                                .bg(Color::Rgb(33, 150, 243))
                                .fg(Color::Black)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::raw("  "),
                        Span::styled(&entry.input, Style::default().fg(Color::White)),
                    ]));
                    lines.push(Line::from(""));

                    // Show AI response
                    if !entry.output.is_empty() {
                        // Check if this is an interruption message
                        let is_interrupted = entry.output.contains("[Interrupted]");

                        if is_interrupted {
                            // Special styling for interruption - no badge, centered, orange
                            lines.push(Line::from(vec![
                                Span::raw("         "),
                                Span::styled(
                                    &entry.output,
                                    Style::default()
                                        .fg(Color::Rgb(255, 152, 0)) // Orange
                                        .add_modifier(Modifier::ITALIC | Modifier::DIM),
                                ),
                            ]));
                        } else {
                            // Normal AI response
                            // Determine badge text and color based on TTS playback status
                            let (badge_text, badge_color) = if is_playing_tts {
                                ("  AI 🔊 ", Color::Rgb(255, 152, 0)) // Orange while speaking
                            } else {
                                ("  AI   ", Color::Rgb(76, 175, 80)) // Green when done
                            };

                            // Strip 🔊 prefix from display text if present
                            let display_text = if entry.output.starts_with("🔊 ") {
                                entry.output.trim_start_matches("🔊 ").trim()
                            } else {
                                &entry.output
                            };

                            lines.push(Line::from(vec![
                                Span::styled(
                                    badge_text,
                                    Style::default()
                                        .bg(badge_color)
                                        .fg(Color::Black)
                                        .add_modifier(Modifier::BOLD),
                                ),
                                Span::raw("  "),
                                Span::styled(display_text, Style::default().fg(Color::White)),
                            ]));
                        }
                    }
                } else {
                    // Dictation mode - just show the output
                    // Check if this is an interruption message
                    let is_interrupted = entry.output.contains("[Interrupted]");

                    if is_interrupted {
                        // Special styling for interruption
                        lines.push(Line::from(Span::styled(
                            &entry.output,
                            Style::default()
                                .fg(Color::Rgb(255, 152, 0)) // Orange
                                .add_modifier(Modifier::ITALIC | Modifier::DIM),
                        )));
                    } else {
                        // Normal output
                        lines.push(Line::from(Span::styled(
                            &entry.output,
                            Style::default().fg(Color::White),
                        )));
                    }
                }

                lines.push(Line::from("")); // Blank line between entries
            }

            // Show inline processing status as AI badge
            // Show status for all processing states except Idle and Writing
            if state.is_processing
                && !matches!(
                    state.processing_status,
                    ProcessingStatus::Writing | ProcessingStatus::Idle
                )
            {
                // Show blinking cursor
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis();
                let blink = (now / 500) % 2 == 0;

                // Determine status message and color based on processing_status
                let (status_msg, status_color) = match state.processing_status {
                    ProcessingStatus::Transcribing => {
                        ("Transcribing audio", Color::Rgb(33, 150, 243))
                    }
                    ProcessingStatus::Thinking => ("Thinking", Color::Rgb(156, 39, 176)),
                    ProcessingStatus::GeneratingResponse => {
                        ("Generating response", Color::Rgb(76, 175, 80))
                    }
                    ProcessingStatus::PlayingAudio => ("Speaking", Color::Rgb(255, 152, 0)),
                    ProcessingStatus::Writing => ("Writing", Color::Rgb(255, 193, 7)),
                    ProcessingStatus::Idle => ("Processing", Color::Rgb(255, 193, 7)),
                };

                // Show as AI badge line (matches format of AI responses)
                lines.push(Line::from(vec![
                    Span::styled(
                        "  AI   ",
                        Style::default()
                            .bg(status_color)
                            .fg(Color::Black)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw("  "),
                    Span::styled(
                        status_msg,
                        Style::default()
                            .fg(status_color)
                            .add_modifier(Modifier::ITALIC),
                    ),
                    if blink {
                        Span::styled(" █", Style::default().fg(status_color))
                    } else {
                        Span::raw("  ")
                    },
                ]));
            }

            // Show partial transcription if available
            if let Some(partial) = &state.partial_transcription {
                if !partial.is_empty() {
                    lines.push(Line::from(""));
                    lines.push(Line::from(vec![
                        Span::styled(
                            "  YOU  ",
                            Style::default()
                                .bg(Color::DarkGray)
                                .fg(Color::Black)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::raw("  "),
                        Span::styled(
                            partial,
                            Style::default()
                                .fg(Color::Rgb(169, 169, 169)) // Light gray for partial text
                                .add_modifier(Modifier::ITALIC),
                        ),
                    ]));
                }
            }

            lines
        };

        // Build title - removed scroll indicator since we only show current conversation
        let title = if state.bidirectional_enabled {
            " Current Conversation | Bidirectional ".to_string()
        } else {
            " Conversation History ".to_string()
        };

        let widget = Paragraph::new(content)
            .block(Block::default().borders(Borders::ALL).title(title))
            .wrap(Wrap { trim: true });

        f.render_widget(widget, area);
    }

    fn render_controls_row(&self, f: &mut Frame, area: Rect) {
        let keys = vec![
            ("s", "Enable/Disable"),
            ("p/Space", "Pause/Resume"),
            ("1-4", "Mode"),
            ("e", "Echo"),
            ("d", "Device"),
            ("o", "Type"),
            ("b", "Backend"),
            ("?", "Help"),
            ("q", "Quit"),
        ];

        let spans: Vec<Span> = keys
            .iter()
            .flat_map(|(key, desc)| {
                vec![
                    Span::styled(format!(" [{}]", key), Style::default().fg(Color::Cyan)),
                    Span::raw(format!(" {} ", desc)),
                ]
            })
            .collect();

        let paragraph = Paragraph::new(Line::from(spans))
            .block(Block::default().borders(Borders::ALL).title(" Controls "));

        f.render_widget(paragraph, area);
    }

    fn render_help(&self, f: &mut Frame) {
        let version = env!("CARGO_PKG_VERSION");
        let help_text = vec![
            Line::from(Span::styled(
                format!("voxtty v{} - Keyboard Shortcuts", version),
                Style::default().add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "Navigation:",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(vec![
                Span::styled("  q, Esc       ", Style::default().fg(Color::Cyan)),
                Span::raw("Quit (or close help)"),
            ]),
            Line::from(vec![
                Span::styled("  ?, h         ", Style::default().fg(Color::Cyan)),
                Span::raw("Toggle this help"),
            ]),
            Line::from(vec![
                Span::styled("  ↑, ↓         ", Style::default().fg(Color::Cyan)),
                Span::raw("Scroll conversation history"),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                "Mode Selection:",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(vec![
                Span::styled("  1            ", Style::default().fg(Color::Green)),
                Span::raw("Dictation mode - direct transcription"),
            ]),
            Line::from(vec![
                Span::styled("  2            ", Style::default().fg(Color::Blue)),
                Span::raw("Assistant mode - AI-enhanced writing"),
            ]),
            Line::from(vec![
                Span::styled("  3            ", Style::default().fg(Color::Magenta)),
                Span::raw("Code mode - code generation"),
            ]),
            Line::from(vec![
                Span::styled("  4            ", Style::default().fg(Color::Yellow)),
                Span::raw("Command mode - execute shell commands"),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                "Controls:",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(vec![
                Span::styled("  s            ", Style::default().fg(Color::Cyan)),
                Span::raw("Enable/Disable voice detection"),
            ]),
            Line::from(vec![
                Span::styled("  p, Space     ", Style::default().fg(Color::Cyan)),
                Span::raw("Pause/Resume listening"),
            ]),
            Line::from(vec![
                Span::styled("  e            ", Style::default().fg(Color::Cyan)),
                Span::raw("Echo test - record and playback audio"),
            ]),
            Line::from(vec![
                Span::styled("  d            ", Style::default().fg(Color::Cyan)),
                Span::raw("Select audio input device"),
            ]),
            Line::from(vec![
                Span::styled("  o            ", Style::default().fg(Color::Cyan)),
                Span::raw("Toggle text output (type to active app)"),
            ]),
            Line::from(vec![
                Span::styled("  b            ", Style::default().fg(Color::Cyan)),
                Span::raw("Switch backend (OpenAI ↔ Speaches)"),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                "Live Dashboard:",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from("  • Audio visualization shows real-time voice levels"),
            Line::from("  • VAD indicator shows voice activity detection"),
            Line::from("  • Last transcription displays most recent text"),
            Line::from("  • All controls visible at once - no navigation needed"),
        ];

        let help_widget = Paragraph::new(help_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Help [? to close]"),
            )
            .wrap(Wrap { trim: true })
            .alignment(Alignment::Left);

        let area = centered_rect(70, 85, f.area());
        f.render_widget(help_widget, area);
    }
}

fn mode_name(mode: &VoiceMode) -> &str {
    match mode {
        VoiceMode::Dictation => "Dictation",
        VoiceMode::Assistant { .. } => "Assistant",
        VoiceMode::Code { .. } => "Code",
        VoiceMode::Command => "Command",
    }
}

fn mode_color(mode: &VoiceMode) -> Style {
    match mode {
        VoiceMode::Dictation => Style::default().fg(Color::Rgb(76, 175, 80)), // Green
        VoiceMode::Assistant { .. } => Style::default().fg(Color::Rgb(33, 150, 243)), // Blue
        VoiceMode::Code { .. } => Style::default().fg(Color::Rgb(156, 39, 176)), // Purple/Magenta
        VoiceMode::Command => Style::default().fg(Color::Rgb(255, 193, 7)),   // Amber/Gold
    }
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

/// Helper function to restore the terminal to its original state
fn cleanup_terminal() -> Result<()> {
    disable_raw_mode()?;
    execute!(
        io::stdout(),
        LeaveAlternateScreen,
        DisableMouseCapture,
        Show
    )?;
    Ok(())
}
