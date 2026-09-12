# RS-Harness

RS-Harness v0.2.1 is a minimal, model-backed Agent Harness. A session is a
persistent model context: you can send multiple requests to the same session
and every system, user, assistant, or tool message is appended to a complete
JSONL ledger. OpenAI is called through its official Python SDK; there is no
offline or Stub client.

## Quick start

```bash
uv sync
cp .env.example .env
# Put the API key for the provider configured in config/default.toml in .env.
# The default config uses DeepSeek Vision and reads it from OPENAI_API_KEY.
# Start an interactive session directly.
uv run rsh --config config/default.toml --workdir .
# Enter questions at the `rsh>` prompt; use /help, /status, /quit, /exit or /close.
# Resuming with --session-id automatically shows prior user/assistant history.
# Create a persistent conversation and note the returned session id.
uv run rsh create-session --config config/default.toml --workdir .
# List every session recorded under the configured sessions directory.
uv run rsh list-session --config config/default.toml
# Reuse that context for as many requests as needed. If omitted, ask creates a new session automatically.
uv run rsh ask --config config/default.toml --session-id <SESSION_ID> "什么是 Agent？"
# Attach a directly viewable image (JPEG/PNG/WebP/GIF) that the model can see.
uv run rsh ask --config config/default.toml --image workdir/photo.jpg "描述这张图片"
# Use --workdir on ask/create-session to select the tool workspace.
# Close the session explicitly when the conversation is complete.
uv run rsh close-session --config config/default.toml --session-id <SESSION_ID>
```

Set `model`, `base_url`, and `api_key_env` in a Config file (see
`config/example.toml`). The default config uses `deepseek-v4-flash-vision-exp`
at `https://api.deepseek.com` and reads the provider key from
`OPENAI_API_KEY`. Use `--json` to print the session result. When a request
invokes tools, the hook notice lines are written to stdout before the JSON
payload.
The system prompt is configurable through `system_prompt_file` under
`[runtime]`; it must contain `{workdir}` and is rendered with the session's
fixed directory when the session is created. File creation, modification, and
editing are restricted to that directory, and a session cannot be resumed with
a different workdir. File-tool paths stay inside that directory unless a
permission callback explicitly approves a single workspace escape.

Each session writes the following files under `sessions/<session_id>/`:

* `session.json` — current session metadata and lifecycle state, including the
  absolute tool workspace and the persisted `todo_write` task list.
* `messages.jsonl` — append-only, complete message ledger. Each record carries
  a sequence number, timestamp, and the provider message payload.
* `events.jsonl` — append-only lifecycle audit for UserPromptSubmit, PreToolUse,
  PostToolUse, and Stop Hooks. Arguments and outputs are redacted and truncated
  before they are written.

The loop exposes seven tools to the model: `bash`, `read_file`,
`write_file`, `edit_file`, `glob`, `read_image`, and `todo_write`. File-tool paths and glob matches stay
inside the session workspace by default; tool failures are returned to the model as
observations so it can recover. Tool observations are redacted against the
configured API key before they reach the ledger, the model, or any log.

`todo_write` stores a planning list in the session. It accepts at most 20 items;
each item has non-empty `content` and a `status` of `pending`, `in_progress`, or
`completed`, with at most one item in progress. The rendered list is shown through
the configured console/hook channel and is not added as a separate ledger message.
During one `ask` interaction, three consecutive tool-use rounds without a
`todo_write` update cause the next model request to receive
`<reminder>Update your todos.</reminder>`; the counter resets after an update and
at the start of the next interaction.

When a user asks to inspect a local image without giving a path, the model can
use `glob` to find it and then call `read_image`. The tool result is recorded as
text and the image is attached to the next model request as an `image_url`
content block; the ledger stores only an `image_ref` record.

Images enter as standard vision input: pass a local JPEG/PNG/WebP/GIF file
(≤ 32 MB) or an http(s) URL via `--image`. To inspect a local image without
`--image`, let the model use `glob` and `read_image`; query text is not scanned
for image paths. The model receives the image as an `image_url` content block;
the ledger keeps only an `image_ref` record (path, mime, size, sha256) and the
bytes are re-read from disk on every request, so image data never persists in
the trace. Remote-sensing rasters (GeoTIFF and similar) stay
opaque: the harness never decodes pixels, and the message states that their
analysis requires MCP tools.

`ask` appends one user interaction and the model's message(s), including any
tool calls and tool results, then leaves the
session open. Reusing `--session-id` restores the full prior context. In the
interactive client, the restored session's user and assistant text is shown
before the prompt; system messages and tool details remain hidden. When the
option is omitted, a new session is created for that invocation; use `--json`
to obtain its generated `session_id`. Empty query and image input exits
quietly; a closed session rejects further asks. A session with an incomplete
historical tool call must be replaced rather than replaying the old action.

The loop provides `bash`, `read_file`, `write_file`, `edit_file`, `glob`,
`read_image`, and `todo_write` for workspace tasks, plus vision input for web-format images.
Remote-sensing semantics stay behind the MCP boundary. Use `--workdir` to select the workspace for a session;
it may be placed before or after a subcommand, and the subcommand value takes
precedence.

Tool execution is silent inside the loop; the CLI injects a console callback
that prints hook notices to stdout — the working directory at prompt submit,
each tool invocation with a short argument preview before execution, and a
tool-call summary at stop — colored on a terminal, plain text when piped,
matching the reference interactive client.

Hooks run at four lifecycle points. `UserPromptSubmit` records the incoming
request, `PreToolUse` runs `check_permission` and previews the invocation,
and `PostToolUse` normalizes errors and flags outputs over 100,000 characters;
`Stop` records a summary for every interaction termination. Risky commands and
file paths outside the session workspace require a `permission_callback`; without
one they are denied. Hard-deny Bash patterns cannot be approved. An approved
workspace escape applies only to that single file-tool call; the default
`safe_path` fence remains active.

Library callers can register additional callbacks with `HookRegistry` and pass
it to `run(..., hooks=registry)`. The CLI supplies an interactive approval
callback; API callers can provide their own callback or rely on the safe default.

| Event | Timing | Default behavior |
| --- | --- | --- |
| `UserPromptSubmit` | Before the user message is appended | Records request context |
| `PreToolUse` | After argument decoding, before dispatch | Checks permission, previews invocation |
| `PostToolUse` | After execution or skipped result | Normalizes errors and records large-output metadata |
| `Stop` | Once on every interaction termination | Records status and tool-call count |

这是一个好项目
