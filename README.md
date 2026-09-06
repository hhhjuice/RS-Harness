# RS-Harness

RS-Harness v0.1.0 is a minimal, model-backed Agent Harness. It implements one
small Agent Loop for basic question answering and records each invocation as an
independent session. Anthropic and OpenAI are called through their official
Python SDKs; there is no offline or Stub client.

## Quick start

```bash
uv sync
cp .env.example .env
# Put the selected provider's API key in .env.
uv run rs-harness ask --config config/default.toml "什么是 NDVI？"
```

To use OpenAI, select `provider = "openai"`, set `model` and
`api_key_env = "OPENAI_API_KEY"` in a Config file (see
`config/example.toml`). Use `--json` to print the complete session result.

Every non-empty invocation writes `sessions/<session_id>/session.json`,
`result.json` and `events.jsonl`. Empty query and image input exits quietly.

The v0.1.0 loop deliberately does not implement MCP, tools, Hooks, memory,
context management, streaming, or image-byte processing. Remote-sensing
semantics remain behind the future MCP boundary.
