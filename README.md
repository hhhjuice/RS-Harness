# RS-Harness

RS-Harness v0.1.0 is a minimal, model-backed Agent Harness. A session is a
persistent model context: you can send multiple requests to the same session
and every system, user, assistant, or tool message is appended to a complete
JSONL ledger. OpenAI is called through its official Python SDK; there is no
offline or Stub client.

## Quick start

```bash
uv sync
cp .env.example .env
# Put your OpenAI API key in .env.
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

Set `model` and `api_key_env = "OPENAI_API_KEY"` in a Config file (see
`config/example.toml`). Use `--json` to print the session result. When a request
invokes tools, the tool log lines are written to stdout before the JSON payload.

Each session writes the following files under `sessions/<session_id>/`:

* `session.json` — current session metadata and lifecycle state, including the
  absolute tool workspace.
* `messages.jsonl` — append-only, complete message ledger. Each record carries
  a sequence number, timestamp, and the provider message payload.

The loop exposes six workspace tools to the model: `bash`, `read_file`,
`write_file`, `edit_file`, `glob`, and `read_image`. File-tool paths and glob matches stay
inside the session workspace; tool failures are returned to the model as
observations so it can recover. Tool observations are redacted against the
configured API key before they reach the ledger, the model, or any log.

When a user asks to inspect a local image without giving a path, the model can
use `glob` to find it and then call `read_image`. The tool result is recorded as
text and the image is attached to the next model request as an `image_url`
content block; the ledger stores only an `image_ref` record.

Images enter as standard vision input: pass a local JPEG/PNG/WebP/GIF file
(≤ 20 MB) or an http(s) URL via `--image`, or simply paste a local image path
into the query text. The model receives the image as an `image_url` content
block; the ledger keeps only an `image_ref` record (path, mime, size, sha256)
and the bytes are re-read from disk on every request, so image data never
persists in the trace. Remote-sensing rasters (GeoTIFF and similar) stay
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

The v0.1.0 loop provides `bash`, `read_file`, `write_file`, `edit_file`, `glob`,
and `read_image` for general workspace tasks, plus vision input for web-format images.
MCP, Hooks, memory, context management, streaming, and remote-sensing raster
processing remain out of scope; remote-sensing semantics stay
behind the MCP boundary. Use `--workdir` to select the workspace for a session;
it may be placed before or after a subcommand, and the subcommand value takes
precedence.

Tool execution is silent inside the loop; the CLI injects a log callback that
prints the tool name and the first 200 characters of its result to stdout —
colored on a terminal, plain text when piped — matching the reference
interactive client.
