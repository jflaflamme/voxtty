# TODO

## Analyze: hardcoded / duplicated logic (tech-debt audit)

Spotted while wiring the OpenAI-compatible STT + generic TTS work. None are
blocking, but they make the codebase brittle and should be analyzed and
consolidated. Rough priority order:

1. **Two divergent assistant paths (realtime vs batch).** `--realtime` uses
   `process_text` → `AssistantProcessor` (types only, never speaks), while the
   batch path (`processor.process`) handles `speak`/`switch_mode` tool calls and
   triggers TTS. Same logical feature, two implementations with different
   behavior. Spoken replies only work in bidirectional / batch paths.
   → Analyze whether these can be unified behind one processing pipeline.

2. **String-prefix control flow.** Layers signal intent by prefixing output with
   `🔊 ` / `📝` / `💻 $` and downstream code does `response.starts_with("🔊 ")`
   then `trim_start_matches`. Fragile, emoji-as-protocol. Several copies.
   → Replace with a typed enum (e.g. `AssistantOutput::{Speak, Type, Command}`).

3. **Backend selection duplicated.** `Backend` (STT) and `RealtimeProvider` are
   parallel enums; selection is done with two separate if/else+match ladders
   over flags (`--openai`/`--speaches`/...) and `config.backend` strings, in
   multiple places (startup print, connectivity, provider build, TUI switch).
   → Analyze collapsing to one resolved config struct computed once.

4. **Speaches is redundant with the generic OpenAI backend.** Both POST to
   `/v1/audio/transcriptions`; the only real distinct local STT is whisper.cpp.
   → Consider folding `Speaches` into the OpenAI-compatible backend.

5. **Hardcoded model names / endpoints scattered.** e.g. `"whisper-1"`,
   `"kokoro-v1"`, realtime Speaches default model
   `"Systran/faster-distil-whisper-small.en"`, default ports. Some are config
   fields, some are inline literals; not consistently surfaced to config/env.

6. **Duplicated TTS-spawn blocks.** `ConversationProcessor` had 3 near-identical
   thread-spawn + speak blocks (now call `TtsClient::speak_blocking`, but the
   surrounding spawn/flag boilerplate is still copy-pasted). `main.rs` had 3
   identical `spawn_tts` call sites (collapsed via `TtsSettings::from_config`).
   → Factor the spawn/is_tts_speaking/interrupt boilerplate into one helper.

7. **whisper.cpp `response_format` quirk.** The hyphenated `response-format`
   param is silently ignored by the server (returns JSON); we now send the
   underscore form and parse both. Worth a small shared helper instead of the
   inline defensive parse duplicated in `main.rs` and
   `processors_transcription.rs`.

8. **Wake words / mode-switch phrases hardcoded** in `modes.rs`
   (`WakeWordDetector::new`). Not configurable. Substring `contains` matching is
   crude (e.g. "stop listening" inside a longer sentence).

9. **TTS-vs-type decision lives in the LLM tool choice + system prompt.** Whether
   a reply is spoken depends on the model emitting a `speak` tool call, which
   small local models do unreliably. Consider an explicit "speak replies" mode
   (`TTS_SPEAK_REPLIES` / `--speak-replies`) independent of tool calls.
