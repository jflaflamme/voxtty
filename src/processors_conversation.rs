// Bidirectional conversation processor with clarification support
use crate::conversation::{ConversationContext, ConversationState, LlmAnalysisResponse};
use crate::mcp_tools::McpManager;
use crate::tts_client::TtsClient;
use crate::processors::{AudioProcessor, ProcessContext, VoiceMode};
use crate::screen_capture::ScreenCapture;
use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::json;
use std::any::Any;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

type TranscriptionFn = Arc<dyn Fn(&Path) -> Result<String> + Send + Sync>;

/// Built-in tool names handled by the conversation processor directly
const BUILTIN_TOOLS: &[&str] = &["process_command", "speak", "type_text", "switch_mode"];

/// Maximum MCP tool call iterations per conversation turn
const MAX_MCP_ITERATIONS: usize = 5;

/// Strip `<think>...</think>` blocks that reasoning models (Qwen3, LFM2.5
/// Thinking, DeepSeek-R1) emit inline in `content` — the chain-of-thought must
/// never reach TTS. An unterminated `<think>` swallows the rest of the string
/// (the model ran out of tokens mid-thought; there is no answer after it).
pub fn strip_think_blocks(content: &str) -> String {
    let mut out = String::with_capacity(content.len());
    let mut rest = content;
    while let Some(start) = rest.find("<think>") {
        out.push_str(&rest[..start]);
        match rest[start..].find("</think>") {
            Some(end) => rest = &rest[start + end + "</think>".len()..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out.trim().to_string()
}

#[cfg(test)]
mod think_tests {
    use super::strip_think_blocks;

    #[test]
    fn strips_closed_block() {
        assert_eq!(
            strip_think_blocks("<think>reasoning here</think>\n\nសួស្តី"),
            "សួស្តី"
        );
    }

    #[test]
    fn unterminated_block_swallows_rest() {
        assert_eq!(strip_think_blocks("<think>ran out of tokens"), "");
    }

    #[test]
    fn passes_through_plain_text() {
        assert_eq!(strip_think_blocks("  Bonjour  "), "Bonjour");
    }

    #[test]
    fn strips_multiple_blocks() {
        assert_eq!(
            strip_think_blocks("<think>a</think>x<think>b</think>y"),
            "xy"
        );
    }
}

#[cfg(test)]
mod arg_tests {
    use super::{recover_tool_args, text_arg};
    use serde_json::json;

    #[test]
    fn text_arg_canonical_and_aliases() {
        assert_eq!(text_arg(&json!({"text": "hi"})).as_deref(), Some("hi"));
        assert_eq!(text_arg(&json!({"content": "yo"})).as_deref(), Some("yo"));
    }

    #[test]
    fn text_arg_sole_string_fallback() {
        assert_eq!(text_arg(&json!({"weird": "value"})).as_deref(), Some("value"));
        // Ambiguous (two strings) → no guess.
        assert_eq!(text_arg(&json!({"a": "x", "b": "y"})), None);
    }

    #[test]
    fn recovers_text_from_malformed_json() {
        // Unescaped inner quotes break strict JSON; we still salvage the text.
        let v = recover_tool_args(r#"{"text": "say "Hello World" now"}"#);
        assert_eq!(v.get("text").and_then(|x| x.as_str()), Some(r#"say "Hello World" now"#));
    }

    #[test]
    fn recovers_command_field() {
        let v = recover_tool_args(r#"{"command": "ls -la", bad}"#);
        assert_eq!(v.get("command").and_then(|x| x.as_str()), Some("ls -la"));
    }

    #[test]
    fn mcp_confirmation_summarizes() {
        use super::mcp_confirmation;
        assert_eq!(mcp_confirmation("ok"), "Done.");
        assert_eq!(mcp_confirmation("Done."), "Done.");
        assert_eq!(mcp_confirmation("  "), "Done.");
        // Informative / error results are spoken verbatim.
        assert_eq!(mcp_confirmation("Battery: 95%"), "Battery: 95%");
        assert!(mcp_confirmation("Failed (exit 1): nope").starts_with("Failed"));
    }
}

/// Parse a tool call that a small model emitted as JSON text in `content`
/// instead of a structured `tool_calls` array. Accepts the common shapes
/// `{"name": "...", "arguments": {...}}` / `{"name": "...", "parameters": {...}}`,
/// with the args either inline JSON or a JSON-encoded string, optionally
/// wrapped in a markdown code fence.
fn parse_inline_tool_call(content: &str) -> Option<(String, serde_json::Value)> {
    let mut text = content.trim();
    if let Some(stripped) = text.strip_prefix("```") {
        // Drop an optional language tag (```json) and the closing fence.
        let stripped = stripped.strip_prefix("json").unwrap_or(stripped);
        text = stripped.strip_suffix("```").unwrap_or(stripped).trim();
    }

    // Shape 1: a full JSON object — {"name": "...", "arguments"/"parameters": ...}
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(text) {
        if let Some(name) = value.get("name").and_then(|n| n.as_str()) {
            let args = value
                .get("arguments")
                .or_else(|| value.get("parameters"))
                .cloned()
                .unwrap_or(json!({}));
            // Some models double-encode the arguments as a string.
            let args = match args {
                serde_json::Value::String(s) => serde_json::from_str(&s).ok()?,
                other => other,
            };
            return Some((name.to_string(), args));
        }
        // Shape 1b: bare speak args — {"text": "..."} with no tool name.
        if let Some(obj) = value.as_object() {
            if obj.len() == 1 && obj.get("text").map(|t| t.is_string()).unwrap_or(false) {
                return Some(("speak".to_string(), value));
            }
        }
    }

    // Shape 2: `speak {"text": "..."}` — bare tool name then JSON args, possibly
    // prefixed with an emoji or label (e.g. `🗣️ speak {...}`).
    let brace = text.find('{')?;
    let (prefix, rest) = text.split_at(brace);
    // Tool name = last word of the prefix, stripped to identifier chars.
    let name: String = prefix
        .split_whitespace()
        .last()?
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() {
        return None;
    }
    // Parse the first JSON value, tolerating trailing text after the object.
    let args = serde_json::Deserializer::from_str(rest)
        .into_iter::<serde_json::Value>()
        .next()?
        .ok()?;
    if !args.is_object() {
        return None;
    }
    Some((name, args))
}

/// Best-effort extraction of a string value for `key` from malformed JSON args
/// (small models often break JSON when the value contains quotes). Greedy: takes
/// from the first `"` after the key to the last `"`, tolerating inner quotes.
fn loose_string_value(s: &str, key: &str) -> Option<String> {
    let marker = format!("\"{}\"", key);
    let start = s.find(&marker)? + marker.len();
    let rest = s[start..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.rfind('"')?;
    Some(rest[..end].to_string())
}

/// Extract the text payload for `speak`/`type_text`, tolerating small-model
/// quirks: the canonical "text" key, common aliases, or — failing that — the
/// sole string value in the object.
fn text_arg(args: &serde_json::Value) -> Option<String> {
    for key in ["text", "content", "message", "value"] {
        if let Some(s) = args.get(key).and_then(|v| v.as_str()) {
            return Some(s.to_string());
        }
    }
    // Fallback: exactly one string field, whatever its key.
    if let Some(obj) = args.as_object() {
        let strings: Vec<&str> = obj.values().filter_map(|v| v.as_str()).collect();
        if strings.len() == 1 {
            return Some(strings[0].to_string());
        }
    }
    None
}

/// Turn an MCP tool result into a spoken confirmation. Trivial acks ("ok",
/// "done") become "Done."; errors and informative results (status, etc.) are
/// spoken as-is. Used when the model executes a tool but then fails to produce
/// its own spoken summary — the action still happened, so confirm it.
fn mcp_confirmation(result: &str) -> String {
    let r = result.trim();
    if r.is_empty()
        || r.eq_ignore_ascii_case("ok")
        || r.eq_ignore_ascii_case("done")
        || r.eq_ignore_ascii_case("done.")
    {
        "Done.".to_string()
    } else {
        r.to_string()
    }
}

/// Recover tool arguments from an arguments string that failed strict JSON
/// parsing. Salvages the common single-field shapes used by built-in tools.
fn recover_tool_args(s: &str) -> serde_json::Value {
    for key in ["text", "command", "confirmation", "mode"] {
        if let Some(v) = loose_string_value(s, key) {
            return json!({ key: v });
        }
    }
    json!({})
}

/// Gated debug logging — only prints when the processor's debug flag is on.
macro_rules! dbg_log {
    ($self:ident, $($arg:tt)*) => {
        if $self.debug { eprintln!($($arg)*); }
    };
}

/// Optional per-mode LLM model overrides; each falls back to the default model.
#[derive(Default, Clone)]
pub struct ModeModels {
    pub translate: Option<String>,
    pub command: Option<String>,
    pub code: Option<String>,
    pub assistant: Option<String>,
    pub screen: Option<String>,
}

/// Processor that handles bidirectional conversations with clarification
pub struct ConversationProcessor {
    http_client: Client,
    api_base_url: String,
    api_key: String,
    model: String,
    mode_models: ModeModels,
    context: Arc<Mutex<ConversationContext>>,
    tts_client: Option<Arc<TtsClient>>,
    /// Separate TTS client used only for Translate mode (target-language voice).
    translate_tts_client: Option<Arc<TtsClient>>,
    /// Last mode seen, so we can clear history when the mode changes (otherwise
    /// e.g. Khmer translate turns bleed into the next assistant reply).
    last_mode: Mutex<Option<VoiceMode>>,
    transcription_fn: Option<TranscriptionFn>,
    is_tts_speaking: Arc<Mutex<bool>>,
    tts_interrupt: Arc<AtomicBool>,
    mcp_manager: Option<Arc<Mutex<McpManager>>>,
    /// Optional name the assistant refers to itself by (Assistant mode).
    assistant_name: Option<String>,
    /// Gate for verbose [DEBUG] logging (off unless --debug).
    debug: bool,
}

impl ConversationProcessor {
    /// Create a new ConversationProcessor with a pre-configured TTS client
    pub fn with_tts_client(
        api_base_url: String,
        api_key: String,
        model: String,
        tts_client: Arc<TtsClient>,
        is_tts_speaking: Arc<Mutex<bool>>,
        tts_interrupt: Arc<AtomicBool>,
    ) -> Self {
        Self {
            http_client: Client::new(),
            api_base_url,
            api_key,
            model,
            mode_models: ModeModels::default(),
            context: Arc::new(Mutex::new(ConversationContext::new())),
            tts_client: Some(tts_client),
            translate_tts_client: None,
            last_mode: Mutex::new(None),
            transcription_fn: None,
            is_tts_speaking,
            tts_interrupt,
            mcp_manager: None,
            assistant_name: None,
            debug: false,
        }
    }

    /// Enable verbose [DEBUG] logging (wired from --debug).
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    /// Set the assistant's self-referenced name (Assistant mode identity).
    pub fn set_assistant_name(&mut self, name: Option<String>) {
        self.assistant_name = name.filter(|n| !n.trim().is_empty());
    }

    /// Set optional per-mode model overrides.
    pub fn set_mode_models(&mut self, models: ModeModels) {
        self.mode_models = models;
    }

    /// Set the separate TTS client used for Translate-mode (target language).
    pub fn set_translate_tts(&mut self, client: Arc<TtsClient>) {
        self.translate_tts_client = Some(client);
    }

    /// Resolve the model to use for a given mode (override or default).
    fn model_for(&self, mode: &VoiceMode) -> String {
        let over = match mode {
            VoiceMode::Translate => &self.mode_models.translate,
            VoiceMode::Screen => &self.mode_models.screen,
            VoiceMode::Command => &self.mode_models.command,
            VoiceMode::Code { .. } => &self.mode_models.code,
            VoiceMode::Assistant { .. } => &self.mode_models.assistant,
            _ => &None,
        };
        over.clone().unwrap_or_else(|| self.model.clone())
    }

    /// Pick the TTS client for a mode: the target-language client for Translate
    /// (when configured), otherwise the main (English) client.
    fn tts_for(&self, mode: &VoiceMode) -> Option<&Arc<TtsClient>> {
        if matches!(mode, VoiceMode::Translate) {
            if let Some(t) = &self.translate_tts_client {
                return Some(t);
            }
        }
        self.tts_client.as_ref()
    }

    /// Clear conversation history when the mode changes, so prior-mode turns
    /// (e.g. Khmer translations) don't bleed into the new mode's replies.
    fn reset_on_mode_change(&self, mode: &VoiceMode) {
        let mut last = self.last_mode.lock().unwrap();
        // Compare by variant only — Assistant{context}/Code{language} carry inner
        // data that varies between turns within the same logical mode.
        let changed = matches!(last.as_ref(), Some(prev)
            if std::mem::discriminant(prev) != std::mem::discriminant(mode));
        if changed {
            dbg_log!(self, "[DEBUG] Mode changed; clearing conversation history");
            self.context.lock().unwrap().reset();
        }
        *last = Some(mode.clone());
    }

    /// POST a chat-completions request, attaching auth only when a key is
    /// configured. A bare `Authorization: Bearer ` (empty token) makes local
    /// llama.cpp servers (Lemonade) 500.
    async fn post_chat(&self, body: &serde_json::Value) -> Result<reqwest::Response> {
        let mut request = self
            .http_client
            .post(format!("{}/chat/completions", self.api_base_url))
            .header("Content-Type", "application/json")
            .json(body);

        if !self.api_key.is_empty() {
            request = request.header("Authorization", format!("Bearer {}", self.api_key));
        }

        request
            .send()
            .await
            .context("Failed to send analysis request")
    }

    /// Analyze transcription and determine if clarification is needed
    async fn analyze_transcription(
        &self,
        transcription: &str,
        context: &ConversationContext,
        mode: &VoiceMode,
    ) -> Result<LlmAnalysisResponse> {
        let mut system_prompt = match mode {
            VoiceMode::Command => include_str!("../prompts/command.md").to_string(),
            VoiceMode::Code { .. } => include_str!("../prompts/code.md").to_string(),
            VoiceMode::Translate => crate::translate_prompt(),
            VoiceMode::Screen => include_str!("../prompts/screen.md").to_string(),
            _ => include_str!("../prompts/assistant.md").to_string(),
        };

        // Translate and Screen are focused, single-purpose tasks: the date/time,
        // user skills, and MCP tool catalog are irrelevant noise. Injecting them
        // bloats the prompt and pushes small models off-task. Keep them lean.
        let is_translate = matches!(mode, VoiceMode::Translate);
        let is_screen = matches!(mode, VoiceMode::Screen);
        let lean = is_translate || is_screen;

        if !lean {
            // Inject current date/time so the LLM always knows
            let now = chrono::Local::now();
            system_prompt.push_str(&format!(
                "\n\n## CURRENT CONTEXT\n\n- **Date**: {}\n- **Time**: {}\n",
                now.format("%A, %B %d, %Y"),
                now.format("%I:%M %p")
            ));

            // Inject the assistant's name so it identifies itself consistently.
            if let Some(ref name) = self.assistant_name {
                system_prompt.push_str(&format!(
                    "\n\n## IDENTITY\n\nYour name is {0}. If the user asks who you are or your name, say you are {0}.\n",
                    name
                ));
            }

            // Inject user skills dropped in ~/.config/voxtty/skills/*.md (hot-reloaded).
            system_prompt.push_str(&crate::skills::skills_prompt_section());
        }

        // Append MCP tool descriptions dynamically from manager
        let mcp_tools_snapshot = self.get_mcp_tools();
        if !lean && !mcp_tools_snapshot.is_empty() {
            system_prompt.push_str("\n\n## EXTERNAL TOOLS (MCP)\n\n");
            system_prompt.push_str(
                "You also have access to external tools provided by MCP servers. \
                 Use these tools when the user's request matches their capabilities. \
                 After calling an external tool, use `speak` to tell the user the result.\n\n",
            );
            for tool in &mcp_tools_snapshot {
                if let (Some(name), Some(desc)) = (
                    tool.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str()),
                    tool.get("function")
                        .and_then(|f| f.get("description"))
                        .and_then(|d| d.as_str()),
                ) {
                    system_prompt.push_str(&format!("- **`{}`**: {}\n", name, desc));
                }
            }
        }

        let mut messages = vec![json!({
            "role": "system",
            "content": system_prompt
        })];

        // Add conversation history
        for msg in context.get_context_for_llm() {
            messages.push(json!({
                "role": "user",
                "content": msg
            }));
        }

        // Add current transcription. In Screen mode, attach the captured screen:
        // exact terminal text inline, or a downscaled screenshot as an image part.
        if is_screen {
            match crate::screen_capture::capture() {
                Ok(ScreenCapture::Text(screen_text)) => {
                    messages.push(json!({
                        "role": "user",
                        "content": format!(
                            "Here is the text currently on my screen:\n\n```\n{}\n```\n\n{}",
                            screen_text, transcription
                        ),
                    }));
                }
                Ok(ScreenCapture::Image(data_uri)) => {
                    messages.push(json!({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": transcription},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }));
                }
                Err(e) => {
                    eprintln!("[WARNING] Screen capture failed: {}", e);
                    return Ok(LlmAnalysisResponse {
                        needs_clarification: false,
                        clarification_question: None,
                        response: Some("🔊 I couldn't capture your screen.".to_string()),
                        confidence: 0.0,
                    });
                }
            }
        } else {
            // Pass the user's words verbatim. Do NOT wrap in "Analyze this
            // request:" — small models then narrate an analysis of the request
            // instead of answering it (and the fallback speaks that aloud).
            messages.push(json!({
                "role": "user",
                "content": transcription
            }));
        }

        // Add tool support - tools depend on mode
        let mut tools = serde_json::json!([]);

        // process_command tool (Command mode only)
        if matches!(mode, VoiceMode::Command) {
            tools.as_array_mut().unwrap().push(json!({
                "type": "function",
                "function": {
                    "name": "process_command",
                    "description": "Process a voice command and convert it to a shell command with safety analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "hearing": {
                                "type": "string",
                                "description": "The exact text that was heard/transcribed"
                            },
                            "understanding": {
                                "type": "string",
                                "description": "Explanation of the user's intent"
                            },
                            "command": {
                                "type": "string",
                                "description": "The shell command to execute (or empty string if unsafe/rejected)"
                            },
                            "risk": {
                                "type": "string",
                                "enum": ["safe", "low", "medium", "high", "destructive"],
                                "description": "Safety risk level of the command"
                            }
                        },
                        "required": ["hearing", "understanding", "command", "risk"]
                    }
                }
            }));
        }

        // speak tool (Assistant, Command, Code, Translate, and Screen modes)
        if matches!(
            mode,
            VoiceMode::Assistant { .. }
                | VoiceMode::Command
                | VoiceMode::Code { .. }
                | VoiceMode::Translate
                | VoiceMode::Screen
        ) {
            tools.as_array_mut().unwrap().push(json!({
                "type": "function",
                "function": {
                    "name": "speak",
                    "description": "Speak a response back to the user via TTS. Use this for clarifications, rejections, answers to questions, or confirmations. Does NOT type text to keyboard.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The text to speak"
                            }
                        },
                        "required": ["text"]
                    }
                }
            }));

            // type_text tool
            tools.as_array_mut().unwrap().push(json!({
                "type": "function",
                "function": {
                    "name": "type_text",
                    "description": "Type text to the keyboard (simulates typing). Use this for dictation, writing emails, code, etc. Does NOT speak via TTS.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The text to type to the keyboard"
                            }
                        },
                        "required": ["text"]
                    }
                }
            }));

            // switch_mode tool
            tools.as_array_mut().unwrap().push(json!({
                "type": "function",
                "function": {
                    "name": "switch_mode",
                    "description": "Switch voice input mode when user requests (e.g., 'switch to dictation', 'code mode').",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mode": {
                                "type": "string",
                                "enum": ["dictation", "assistant", "code", "command", "translate", "screen"],
                                "description": "The mode to switch to"
                            },
                            "confirmation": {
                                "type": "string",
                                "description": "Brief confirmation message to speak"
                            }
                        },
                        "required": ["mode", "confirmation"]
                    }
                }
            }));
        }

        // Append MCP tools dynamically from manager. Translate mode is a pure
        // translation task and must not be tempted to call external tools.
        if !is_translate {
            let mcp_tools_snapshot = self.get_mcp_tools();
            for mcp_tool in &mcp_tools_snapshot {
                tools.as_array_mut().unwrap().push(mcp_tool.clone());
            }
        }

        // NOTE: We always use "auto" rather than "required". llama.cpp-backed
        // servers (e.g. Lemonade) return 500 "Context size has been exceeded" for
        // tool_choice="required" (it forces grammar-constrained decoding that blows
        // the context budget). The system prompts already push strong tool usage.
        // TODO(tool-choice): make this provider-configurable; cloud OpenAI supports "required".
        let tool_choice = "auto";

        dbg_log!(self, "[DEBUG] Total tools for LLM: {} (including {} MCP tools)",
            tools.as_array().map(|a| a.len()).unwrap_or(0),
            mcp_tools_snapshot.len()
        );

        // Translate mode: ask the server to render the chat template with
        // thinking disabled (Qwen3-style reasoning models otherwise burn
        // latency on chain-of-thought before the translation). llama.cpp
        // servers honor this when started with `--jinja`; servers that reject
        // the unknown field get one retry without it below.
        // Qwen3-style reasoning models burn the whole budget on hidden thinking
        // for Translate and (vision) Screen turns; disable it.
        let mut template_kwargs = matches!(mode, VoiceMode::Translate | VoiceMode::Screen);

        // Resolve the model for this mode (per-mode override or default).
        let model = self.model_for(mode);
        if model != self.model {
            dbg_log!(self, "[DEBUG] Using per-mode model for {:?}: {}", mode, model);
        }

        // Small quantized models occasionally return only a <think> block (no
        // answer, no tool call). Re-sample a couple of times before giving up
        // rather than surfacing "I had trouble" on a one-off bad sample.
        let mut think_retries: u8 = 0;
        const MAX_THINK_RETRIES: u8 = 2;

        // Remember the last MCP tool result so that if the model executes an
        // action (e.g. switch workspace) but then fails to speak a summary, we
        // still confirm success instead of falsely reporting failure.
        let mut last_mcp_result: Option<String> = None;

        // MCP tool call loop: iterate until we get a built-in tool call or content
        for iteration in 0..=MAX_MCP_ITERATIONS {
            let use_tool_choice = if iteration == 0 { tool_choice } else { "auto" };

            let mut request_body = json!({
                "model": model,
                "messages": messages,
                "temperature": 0.3,
                "tools": tools,
                "tool_choice": use_tool_choice
            });
            if template_kwargs {
                request_body["chat_template_kwargs"] = json!({"enable_thinking": false});
            }

            let mut response = self.post_chat(&request_body).await?;

            if !response.status().is_success() && template_kwargs {
                dbg_log!(self, "[DEBUG] Request with chat_template_kwargs failed ({}); retrying without it",
                    response.status()
                );
                template_kwargs = false;
                request_body
                    .as_object_mut()
                    .unwrap()
                    .remove("chat_template_kwargs");
                response = self.post_chat(&request_body).await?;
            }

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_default();
                anyhow::bail!("API request failed: {} - {}", status, error_text);
            }

            let response_json: serde_json::Value = response.json().await?;

            dbg_log!(self, "[DEBUG] LLM Response (iteration {}): {}",
                iteration,
                serde_json::to_string_pretty(&response_json).unwrap_or_default()
            );

            // Check for tool calls
            if let Some(tool_calls_arr) =
                response_json["choices"][0]["message"]["tool_calls"].as_array()
            {
                dbg_log!(self, "[DEBUG] Found {} tool calls", tool_calls_arr.len());
                if let Some(tool_call) = tool_calls_arr.first() {
                    if let Some(tool_name) = tool_call["function"]["name"].as_str() {
                        let tool_id = tool_call["id"].as_str().unwrap_or("call_0");
                        dbg_log!(self, "[DEBUG] Tool name: {}", tool_name);

                        {
                            // `arguments` is usually a JSON-encoded string, but some
                            // servers return it as an object. Handle both, and
                            // recover from malformed JSON (small models break it on
                            // embedded quotes) instead of silently dropping the args.
                            let raw = &tool_call["function"]["arguments"];
                            let (args, args_str): (serde_json::Value, String) =
                                if let Some(s) = raw.as_str() {
                                    match serde_json::from_str::<serde_json::Value>(s) {
                                        Ok(v) => (v, s.to_string()),
                                        Err(e) => {
                                            let recovered = recover_tool_args(s);
                                            eprintln!(
                                                "[WARNING] Tool '{}' arguments were not valid JSON ({}); recovered {} — raw: {}",
                                                tool_name, e, recovered, s
                                            );
                                            (recovered, s.to_string())
                                        }
                                    }
                                } else if raw.is_object() {
                                    (raw.clone(), raw.to_string())
                                } else {
                                    (json!({}), "{}".to_string())
                                };
                            dbg_log!(self, "[DEBUG] Tool args: {}", args_str);

                            // Handle built-in tools
                            if BUILTIN_TOOLS.contains(&tool_name) {
                                return self.handle_builtin_tool(tool_name, &args, mode);
                            }

                            // Handle MCP tools
                            if let Some(ref mcp_mgr) = self.mcp_manager {
                                if iteration >= MAX_MCP_ITERATIONS {
                                    dbg_log!(self, "[DEBUG] Max MCP iterations ({}) reached",
                                        MAX_MCP_ITERATIONS
                                    );
                                    break;
                                }

                                let tool_result =
                                    match mcp_mgr.lock().unwrap().call_tool(tool_name, args) {
                                        Ok(result) => result,
                                        Err(e) => {
                                            format!("Error calling tool '{}': {}", tool_name, e)
                                        }
                                    };

                                dbg_log!(self, "[DEBUG] MCP tool '{}' result: {}",
                                    tool_name,
                                    &tool_result[..tool_result.len().min(200)]
                                );
                                last_mcp_result = Some(tool_result.clone());

                                // Add assistant message with tool call
                                messages.push(json!({
                                    "role": "assistant",
                                    "content": serde_json::Value::Null,
                                    "tool_calls": [{
                                        "id": tool_id,
                                        "type": "function",
                                        "function": {
                                            "name": tool_name,
                                            "arguments": args_str
                                        }
                                    }]
                                }));

                                // Add tool result message
                                messages.push(json!({
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "content": tool_result
                                }));

                                continue; // Loop back to LLM with tool result
                            }
                        }
                    }
                }
            } else {
                dbg_log!(self, "[DEBUG] No tool calls found in response");
            }

            // Fallback to content
            if let Some(content) = response_json["choices"][0]["message"]["content"].as_str() {
                dbg_log!(self, "[DEBUG] Using content: {}", content);

                // Reasoning models emit chain-of-thought inline; never speak it.
                let content = strip_think_blocks(content);
                let content = content.as_str();

                // Small models often emit the tool call as JSON text in content
                // (e.g. {"name":"speak","arguments":{...}}) instead of a structured
                // tool_calls array. Recover it instead of speaking raw JSON aloud.
                if let Some((tool_name, tool_args)) = parse_inline_tool_call(content) {
                    if BUILTIN_TOOLS.contains(&tool_name.as_str()) {
                        dbg_log!(self, "[DEBUG] Recovered inline tool call from content: {}",
                            tool_name
                        );
                        return self.handle_builtin_tool(&tool_name, &tool_args, mode);
                    }
                }

                // Command mode needs a structured `process_command` tool call; if
                // the model answered in prose there's no command to run.
                if matches!(mode, VoiceMode::Command) {
                    eprintln!("[WARNING] Command mode: LLM returned content instead of a tool");
                    return Ok(LlmAnalysisResponse {
                        needs_clarification: false,
                        clarification_question: None,
                        response: Some("🔊 I had trouble understanding that command. Please try again with more specific details.".to_string()),
                        confidence: 0.0,
                    });
                }

                // Assistant/Code modes: small local models often answer in plain
                // content instead of calling the `speak` tool. Don't discard a
                // usable answer — speak it directly (🔊 prefix routes it to TTS).
                if matches!(
                    mode,
                    VoiceMode::Assistant { .. } | VoiceMode::Code { .. } | VoiceMode::Translate | VoiceMode::Screen
                ) {
                    if content.is_empty() {
                        // The model spent its whole reply on chain-of-thought.
                        // Re-sample a couple of times before giving up.
                        if think_retries < MAX_THINK_RETRIES {
                            think_retries += 1;
                            eprintln!(
                                "[WARNING] LLM returned only <think> blocks; retrying ({}/{})",
                                think_retries, MAX_THINK_RETRIES
                            );
                            continue;
                        }
                        // If an MCP tool already ran, the action succeeded — the
                        // model just couldn't summarize it. Confirm success.
                        if let Some(ref result) = last_mcp_result {
                            eprintln!("[WARNING] No spoken summary after tool call; confirming the action instead");
                            return Ok(LlmAnalysisResponse {
                                needs_clarification: false,
                                clarification_question: None,
                                response: Some(format!("🔊 {}", mcp_confirmation(result))),
                                confidence: 1.0,
                            });
                        }
                        eprintln!("[WARNING] LLM returned only <think> blocks after retries; nothing to speak");
                        return Ok(LlmAnalysisResponse {
                            needs_clarification: false,
                            clarification_question: None,
                            response: Some("🔊 I had trouble with that. Please try again.".to_string()),
                            confidence: 0.0,
                        });
                    }
                    eprintln!("[WARNING] LLM returned content instead of a tool; speaking it directly");
                    return Ok(LlmAnalysisResponse {
                        needs_clarification: false,
                        clarification_question: None,
                        response: Some(format!("🔊 {}", content)),
                        confidence: 0.8,
                    });
                }

                return Ok(LlmAnalysisResponse {
                    needs_clarification: false,
                    clarification_question: None,
                    response: Some(content.to_string()),
                    confidence: 1.0,
                });
            }

            dbg_log!(self, "[DEBUG] No content or tool call found!");
            anyhow::bail!("No content or tool call in LLM response");
        }

        // Exceeded max iterations. If a tool ran, confirm it rather than
        // reporting a generic limit error.
        if let Some(ref result) = last_mcp_result {
            return Ok(LlmAnalysisResponse {
                needs_clarification: false,
                clarification_question: None,
                response: Some(format!("🔊 {}", mcp_confirmation(result))),
                confidence: 1.0,
            });
        }
        Ok(LlmAnalysisResponse {
            needs_clarification: false,
            clarification_question: None,
            response: Some(
                "🔊 I ran into a limit processing your request. Please try again.".to_string(),
            ),
            confidence: 0.0,
        })
    }

    /// Handle a built-in tool call and return the appropriate LlmAnalysisResponse
    fn handle_builtin_tool(
        &self,
        tool_name: &str,
        args: &serde_json::Value,
        _mode: &VoiceMode,
    ) -> Result<LlmAnalysisResponse> {
        match tool_name {
            "process_command" => {
                if let Some(command) = args["command"].as_str() {
                    dbg_log!(self, "[DEBUG] Extracted command: {}", command);
                    Ok(LlmAnalysisResponse {
                        needs_clarification: false,
                        clarification_question: None,
                        response: Some(command.to_string()),
                        confidence: 1.0,
                    })
                } else {
                    Ok(LlmAnalysisResponse {
                        needs_clarification: false,
                        clarification_question: None,
                        response: None,
                        confidence: 0.0,
                    })
                }
            }
            "speak" => {
                if let Some(speak_text) = text_arg(args) {
                    dbg_log!(self, "[DEBUG] Extracted speak text: {}", speak_text);
                    Ok(LlmAnalysisResponse {
                        needs_clarification: false,
                        clarification_question: None,
                        response: Some(format!("🔊 {}", speak_text)),
                        confidence: 1.0,
                    })
                } else {
                    eprintln!("[WARNING] speak tool call had no usable text — args: {}", args);
                    Ok(LlmAnalysisResponse {
                        needs_clarification: false,
                        clarification_question: None,
                        response: None,
                        confidence: 0.0,
                    })
                }
            }
            "type_text" => {
                if let Some(type_text) = text_arg(args) {
                    dbg_log!(self, "[DEBUG] Extracted type text: {}", type_text);
                    Ok(LlmAnalysisResponse {
                        needs_clarification: false,
                        clarification_question: None,
                        response: Some(type_text.to_string()),
                        confidence: 1.0,
                    })
                } else {
                    eprintln!("[WARNING] type_text tool call had no usable text — args: {}", args);
                    Ok(LlmAnalysisResponse {
                        needs_clarification: false,
                        clarification_question: None,
                        response: None,
                        confidence: 0.0,
                    })
                }
            }
            "switch_mode" => {
                if let (Some(mode_str), Some(confirmation)) =
                    (args["mode"].as_str(), args["confirmation"].as_str())
                {
                    dbg_log!(self, "[DEBUG] Mode switch request: {} with confirmation: {}",
                        mode_str, confirmation
                    );
                    Ok(LlmAnalysisResponse {
                        needs_clarification: true,
                        clarification_question: Some(confirmation.to_string()),
                        response: None,
                        confidence: 1.0,
                    })
                } else {
                    Ok(LlmAnalysisResponse {
                        needs_clarification: false,
                        clarification_question: None,
                        response: None,
                        confidence: 0.0,
                    })
                }
            }
            _ => {
                eprintln!("[WARNING] Unknown built-in tool: {}", tool_name);
                Ok(LlmAnalysisResponse {
                    needs_clarification: false,
                    clarification_question: None,
                    response: None,
                    confidence: 0.0,
                })
            }
        }
    }

    /// Handle clarification by speaking the question and waiting for response
    async fn handle_clarification(&self, question: &str) -> Result<String> {
        eprintln!("🤔 Clarification needed: {}", question);

        if let Some(tts) = &self.tts_client {
            eprintln!("🔊 Speaking question...");
            let tts_clone = Arc::clone(tts);
            let question_owned = question.to_string();
            let is_speaking_clone = Arc::clone(&self.is_tts_speaking);

            // Spawn TTS in background thread to prevent blocking
            *is_speaking_clone.lock().unwrap() = true;
            dbg_log!(self, "[DEBUG] Set is_tts_speaking = true (clarification)");
            let interrupt_clone = Arc::clone(&self.tts_interrupt);
            let debug = self.debug;
            std::thread::spawn(move || {
                if debug { eprintln!("[DEBUG TTS Thread] Starting clarification TTS playback"); }
                if let Err(e) =
                    tts_clone.speak_blocking(&question_owned, Some(interrupt_clone))
                {
                    eprintln!("❌ TTS Error: {}", e);
                }
                *is_speaking_clone.lock().unwrap() = false;
                if debug { eprintln!("[DEBUG TTS Thread] Set is_tts_speaking = false (clarification)"); }
                eprintln!("✅ Question spoken, waiting for your response...");
            });
        } else {
            eprintln!("⚠️  No TTS client configured - cannot speak question");
        }

        // Wait for user response (this will come from the next VAD trigger)
        // For now, return a placeholder - the actual implementation will
        // integrate with the main audio loop
        Ok(String::new())
    }
}

impl AudioProcessor for ConversationProcessor {
    fn process(&self, audio_path: &Path, proc_context: &ProcessContext) -> Result<String> {
        // First, transcribe the audio
        let transcription = if let Some(transcribe_fn) = &self.transcription_fn {
            transcribe_fn(audio_path)?
        } else {
            anyhow::bail!("No transcription function configured");
        };

        dbg_log!(self, "[DEBUG ConversationProcessor] Transcription: {}",
            transcription
        );

        if transcription.trim().is_empty() {
            dbg_log!(self, "[DEBUG ConversationProcessor] Empty transcription, returning");
            return Ok(String::new());
        }

        let mode = &proc_context.mode;

        // Use tokio runtime for async operations
        let rt = tokio::runtime::Runtime::new()?;

        rt.block_on(async {
            // Get current state (lock scope minimized)
            let current_state = {
                let ctx = self.context.lock().unwrap();
                ctx.state.clone()
            };

            // A finished turn (ReadyToExecute/Completed) must start fresh on the
            // NEXT utterance rather than swallowing it to reset state. Normalize
            // terminal states to Idle (clearing the previous turn) so the new
            // utterance is processed as a new conversation below.
            let current_state = match current_state {
                ConversationState::ReadyToExecute | ConversationState::Completed => {
                    self.context.lock().unwrap().reset();
                    ConversationState::Idle
                }
                other => other,
            };

            match current_state {
                ConversationState::Idle | ConversationState::Processing => {
                    // New conversation or continuation - update state
                    {
                        let mut ctx = self.context.lock().unwrap();
                        ctx.add_user_message(transcription.clone());
                        ctx.state = ConversationState::Processing;
                    }

                    // Get context snapshot for analysis (separate lock scope)
                    let ctx_snapshot = self.context.lock().unwrap().clone();

                    // Analyze if we need clarification (no lock held during await)
                    dbg_log!(self, "[DEBUG ConversationProcessor] Calling analyze_transcription");
                    let analysis = self
                        .analyze_transcription(&transcription, &ctx_snapshot, mode)
                        .await?;
                    dbg_log!(self, "[DEBUG ConversationProcessor] Analysis complete: needs_clarification={}",
                        analysis.needs_clarification
                    );

                    // Check if clarification is needed
                    let (needs_clarification, can_clarify) = {
                        let ctx = self.context.lock().unwrap();
                        (analysis.needs_clarification, ctx.can_ask_clarification())
                    };

                    if needs_clarification && can_clarify {
                        if let Some(question) = analysis.clarification_question {
                            // Update context state
                            {
                                let mut ctx = self.context.lock().unwrap();
                                ctx.add_assistant_message(question.clone());
                                ctx.state = ConversationState::WaitingForClarification;
                                ctx.clarification_count += 1;
                            }

                            // Speak the question (no lock held)
                            self.handle_clarification(&question).await?;

                            // Return formatted output for display (not typing)
                            Ok(format!("🔊 {}", question))
                        } else {
                            Ok(String::new())
                        }
                    } else {
                        // No clarification needed or limit reached - execute
                        {
                            let mut ctx = self.context.lock().unwrap();
                            ctx.state = ConversationState::ReadyToExecute;
                        }

                        if let Some(response) = analysis.response {
                            {
                                let mut ctx = self.context.lock().unwrap();
                                ctx.add_assistant_message(response.clone());
                            }

                            // Check if this is a TTS response (starts with 🔊)
                            if response.starts_with("🔊 ") {
                                let speak_text = response.trim_start_matches("🔊 ").trim();

                                // Speak via TTS in background
                                if let Some(tts) = &self.tts_client {
                                    eprintln!("🔊 Speaking final answer...");
                                    let tts_clone = Arc::clone(tts);
                                    let speak_text_owned = speak_text.to_string();
                                    let is_speaking_clone = Arc::clone(&self.is_tts_speaking);

                                    // Spawn TTS in background thread
                                    *is_speaking_clone.lock().unwrap() = true;
                                    dbg_log!(self, "[DEBUG] Set is_tts_speaking = true");
                                    let interrupt_clone = Arc::clone(&self.tts_interrupt);
                                    let debug = self.debug;
                                    std::thread::spawn(move || {
                                        if debug { eprintln!("[DEBUG TTS Thread] Starting TTS playback"); }
                                        if let Err(e) = tts_clone.speak_blocking(&speak_text_owned, Some(interrupt_clone)) {
                                            eprintln!("❌ TTS Error: {}", e);
                                        }
                                        *is_speaking_clone.lock().unwrap() = false;
                                        if debug { eprintln!("[DEBUG TTS Thread] Set is_tts_speaking = false"); }
                                        eprintln!("✅ Answer spoken");
                                    });
                                }

                                // Return the formatted text for display (don't type it)
                                Ok(response)
                            } else {
                                // Regular text response - return for typing
                                Ok(response)
                            }
                        } else {
                            // LLM didn't use tools properly - return error message
                            eprintln!("[WARNING] No response from LLM, returning error");
                            Ok("🔊 I'm not sure what you want me to do. Please be more specific, or say 'type as is' to type your words exactly as you said them.".to_string())
                        }
                    }
                }

                ConversationState::WaitingForClarification => {
                    // User is answering a clarification question - update state
                    {
                        let mut ctx = self.context.lock().unwrap();
                        ctx.add_user_message(transcription.clone());
                        ctx.state = ConversationState::Processing;
                    }

                    // Get context snapshot for re-analysis
                    let ctx_snapshot = self.context.lock().unwrap().clone();

                    // Re-analyze with the new context (no lock held during await)
                    let analysis = self
                        .analyze_transcription(&transcription, &ctx_snapshot, mode)
                        .await?;

                    // Check if more clarification is needed
                    let (needs_clarification, can_clarify) = {
                        let ctx = self.context.lock().unwrap();
                        (analysis.needs_clarification, ctx.can_ask_clarification())
                    };

                    if needs_clarification && can_clarify {
                        if let Some(question) = analysis.clarification_question {
                            {
                                let mut ctx = self.context.lock().unwrap();
                                ctx.add_assistant_message(question.clone());
                                ctx.clarification_count += 1;
                            }

                            self.handle_clarification(&question).await?;
                            Ok(format!("🔊 {}", question))
                        } else {
                            Ok(String::new())
                        }
                    } else {
                        {
                            let mut ctx = self.context.lock().unwrap();
                            ctx.state = ConversationState::Completed;
                        }

                        if let Some(response) = analysis.response {
                            {
                                let mut ctx = self.context.lock().unwrap();
                                ctx.add_assistant_message(response.clone());
                            }

                            // Check if this is a TTS response (starts with 🔊)
                            if response.starts_with("🔊 ") {
                                let speak_text = response.trim_start_matches("🔊 ").trim();

                                // Speak via TTS in background
                                if let Some(tts) = &self.tts_client {
                                    eprintln!("🔊 Speaking final answer...");
                                    let tts_clone = Arc::clone(tts);
                                    let speak_text_owned = speak_text.to_string();
                                    let is_speaking_clone = Arc::clone(&self.is_tts_speaking);

                                    // Spawn TTS in background thread
                                    *is_speaking_clone.lock().unwrap() = true;
                                    dbg_log!(self, "[DEBUG] Set is_tts_speaking = true");
                                    let interrupt_clone = Arc::clone(&self.tts_interrupt);
                                    let debug = self.debug;
                                    std::thread::spawn(move || {
                                        if debug { eprintln!("[DEBUG TTS Thread] Starting TTS playback"); }
                                        if let Err(e) = tts_clone.speak_blocking(&speak_text_owned, Some(interrupt_clone)) {
                                            eprintln!("❌ TTS Error: {}", e);
                                        }
                                        *is_speaking_clone.lock().unwrap() = false;
                                        if debug { eprintln!("[DEBUG TTS Thread] Set is_tts_speaking = false"); }
                                        eprintln!("✅ Answer spoken");
                                    });
                                }

                                // Return the formatted text for display (don't type it)
                                Ok(response)
                            } else {
                                // Regular text response - return for typing
                                Ok(response)
                            }
                        } else {
                            // LLM didn't use tools properly - return error message
                            eprintln!("[WARNING] No response from LLM, returning error");
                            Ok("🔊 I'm not sure what you want me to do. Please be more specific, or say 'type as is' to type your words exactly as you said them.".to_string())
                        }
                    }
                }

                ConversationState::ReadyToExecute | ConversationState::Completed => {
                    // Reset for new conversation
                    let mut ctx = self.context.lock().unwrap();
                    ctx.reset();
                    Ok(String::new())
                }
            }
        })
    }

    fn name(&self) -> &str {
        "conversation"
    }

    fn supports_mode(&self, mode: &VoiceMode) -> bool {
        matches!(
            mode,
            VoiceMode::Assistant { .. }
                | VoiceMode::Command
                | VoiceMode::Code { .. }
                | VoiceMode::Translate
                | VoiceMode::Screen
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Set the transcription function for the processor
impl ConversationProcessor {
    #[allow(dead_code)]
    pub fn set_transcription_fn<F>(&mut self, f: F)
    where
        F: Fn(&Path) -> Result<String> + Send + Sync + 'static,
    {
        self.transcription_fn = Some(Arc::new(f));
    }

    /// Set MCP manager for external tool support (tools are read dynamically)
    pub fn set_mcp(&mut self, manager: Arc<Mutex<McpManager>>) {
        self.mcp_manager = Some(manager);
    }

    /// Get MCP tools dynamically from manager (supports background init)
    fn get_mcp_tools(&self) -> Vec<serde_json::Value> {
        self.mcp_manager
            .as_ref()
            .map(|m| m.lock().unwrap().to_openai_tools())
            .unwrap_or_default()
    }

    /// Get the conversation context (for debugging/monitoring)
    #[allow(dead_code)]
    pub fn get_context(&self) -> Arc<Mutex<ConversationContext>> {
        Arc::clone(&self.context)
    }

    /// Reset the conversation
    #[allow(dead_code)]
    pub fn reset_conversation(&self) {
        let mut ctx = self.context.lock().unwrap();
        ctx.reset();
    }

    /// Process text directly (for realtime mode)
    pub fn process_text(&self, text: &str, mode: &VoiceMode, _debug: bool) -> Result<String> {
        dbg_log!(self, "[DEBUG ConversationProcessor::process_text] Input: {}",
            text
        );

        if text.trim().is_empty() {
            return Ok(String::new());
        }

        // Clear history if the mode changed since the last turn (before we add
        // this turn), so prior-mode language/context doesn't bleed over.
        self.reset_on_mode_change(mode);

        // Use tokio runtime for async operations
        let rt = tokio::runtime::Runtime::new()?;

        rt.block_on(async {
            // Update context state (minimized lock scope)
            {
                let mut ctx = self.context.lock().unwrap();
                ctx.add_user_message(text.to_string());
                ctx.state = ConversationState::Processing;
            }

            // Get context snapshot for analysis (separate lock scope)
            let ctx_snapshot = self.context.lock().unwrap().clone();

            // Analyze the text (no lock held during await)
            let analysis = self
                .analyze_transcription(text, &ctx_snapshot, mode)
                .await?;

            // Check if clarification is needed
            let (needs_clarification, can_clarify) = {
                let ctx = self.context.lock().unwrap();
                (analysis.needs_clarification, ctx.can_ask_clarification())
            };

            if needs_clarification && can_clarify {
                if let Some(question) = analysis.clarification_question {
                    {
                        let mut ctx = self.context.lock().unwrap();
                        ctx.add_assistant_message(question.clone());
                        ctx.state = ConversationState::WaitingForClarification;
                        ctx.clarification_count += 1;
                    }

                    // Speak the question (no lock held)
                    self.handle_clarification(&question).await?;

                    // Return formatted output for display
                    Ok(format!("🔊 {}", question))
                } else {
                    Ok(String::new())
                }
            } else {
                {
                    let mut ctx = self.context.lock().unwrap();
                    ctx.state = ConversationState::ReadyToExecute;
                }

                if let Some(response) = analysis.response {
                    {
                        let mut ctx = self.context.lock().unwrap();
                        ctx.add_assistant_message(response.clone());
                    }

                    // Check if this is a TTS response (starts with 🔊)
                    if response.starts_with("🔊 ") {
                        let speak_text = response.trim_start_matches("🔊 ").trim();

                        // Speak via TTS in background, routing Translate mode to
                        // the target-language client when configured.
                        if let Some(tts) = self.tts_for(mode) {
                            eprintln!("🔊 Speaking final answer...");
                            let tts_clone = Arc::clone(tts);
                            let speak_text_owned = speak_text.to_string();
                            let is_speaking_clone = Arc::clone(&self.is_tts_speaking);

                            // Spawn TTS in background thread
                            *is_speaking_clone.lock().unwrap() = true;
                            dbg_log!(self, "[DEBUG process_text] Set is_tts_speaking = true");
                            let interrupt_clone = Arc::clone(&self.tts_interrupt);
                            let debug = self.debug;
                            std::thread::spawn(move || {
                                if debug { eprintln!("[DEBUG process_text Thread] Starting TTS playback"); }
                                if let Err(e) = tts_clone
                                    .speak_blocking(&speak_text_owned, Some(interrupt_clone))
                                {
                                    eprintln!("❌ TTS Error: {}", e);
                                }
                                *is_speaking_clone.lock().unwrap() = false;
                                if debug { eprintln!("[DEBUG process_text Thread] Set is_tts_speaking = false"); }
                                eprintln!("✅ Answer spoken");
                            });
                        }

                        // Return the formatted text for display (don't type it)
                        Ok(response)
                    } else {
                        // Regular text response - return for typing
                        Ok(response)
                    }
                } else {
                    // LLM didn't use tools properly - return error message
                    eprintln!("[WARNING] No response from LLM in process_text, returning error");
                    Ok("🔊 I didn't understand that request. Could you rephrase it?".to_string())
                }
            }
        })
    }
}
