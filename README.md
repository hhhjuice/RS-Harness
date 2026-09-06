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
uv run rsh --config config/default.toml
# Enter questions at the `rsh>` prompt; use /help, /status, /quit or /close.
# Create a persistent conversation and note the returned session id.
uv run rsh create-session --config config/default.toml
# Reuse that context for as many requests as needed. If omitted, ask creates a new session automatically.
uv run rsh ask --config config/default.toml --session-id <SESSION_ID> "什么是 Agent？"
# Close the session explicitly when the conversation is complete.
uv run rsh close-session --config config/default.toml --session-id <SESSION_ID>
```

Set `model` and `api_key_env = "OPENAI_API_KEY"` in a Config file (see
`config/example.toml`). Use `--json` to print the complete session result.

Each session writes the following files under `sessions/<session_id>/`:

* `session.json` — current session metadata and lifecycle state.
* `messages.jsonl` — append-only, complete message ledger. Each record carries
  a sequence number, timestamp, and the provider message payload.

`ask` appends one user interaction and the model's message(s), then leaves the
session open. Reusing `--session-id` restores the full prior context. When the
option is omitted, a new session is created for that invocation; use `--json`
to obtain its generated `session_id`. Empty query and image input exits
quietly; a closed session rejects further asks.

The v0.1.0 loop deliberately does not implement MCP, tools, Hooks, memory,
context management, streaming, or image-byte processing. Remote-sensing
semantics remain behind the future MCP boundary.
