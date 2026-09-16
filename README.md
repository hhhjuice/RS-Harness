# RS-Harness

RS-Harness v0.2.2 is a minimal, model-backed Agent Harness. A session is a
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
# List root sessions recorded under the configured sessions directory.
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
  absolute tool workspace, the persisted `todo_write` task list, and a child's
  name and `parent_session_id`.
* `messages.jsonl` — append-only, complete message ledger. Each record carries
  a sequence number, timestamp, and the provider message payload.
* `events.jsonl` — append-only lifecycle audit for UserPromptSubmit, PreToolUse,
  PostToolUse, and Stop Hooks. Arguments and outputs are redacted and truncated
  before they are written.

Delegation via `task` creates a named, persistent child session in the same
sessions directory, linked to the caller by `parent_session_id`. A new child
starts with its own system prompt and the delegation prompt. It shares the
parent's workdir, permission callback, and hooks, and keeps its own complete
ledgers and todo list. The parent receives the child's identity, status, and
final text; intermediate messages and tool results stay in the child ledger.

| Operation | Model tool call |
| --- | --- |
| Start in background | `task(name="影像元数据检查员", prompt="检查目录中的影像元数据")` |
| Continue | `task(session_id="<CHILD_ID>", prompt="根据刚才的结果检查坐标系")` |
| Inspect / wait | `task(action="status", session_id="<CHILD_ID>")` / `task(action="wait", session_id="<CHILD_ID>", timeout_ms=5000)` |
| Close | `task(action="close", session_id="<CHILD_ID>")` |

`action` defaults to `run`, and `run` starts a background execution by default. Pass `background=false` when a synchronous result is required. Creation requires a name describing the child's
identity and responsibility: 1–64 characters after trimming, without newlines
or control characters. Names are immutable and may repeat; always use the
returned `session_id` to address a child. Continuation and closure do not accept
`name`. Running requires a non-empty `prompt` of at most 8000 characters;
closure does not accept `prompt` and does not call the model.

After creating or validating a child, the tool observation begins with:

```text
session_id: <CHILD_ID>
name: 影像元数据检查员
status: running
execution_status: queued
session_status: running

<This interaction's final answer or error>
```

Save and reuse that ID for follow-ups, including after resuming the parent
session in a new process. The child retains its original system message,
history, name, and todos; prompt templates are read only when creating it.
Only the current parent's direct children with the same workdir can be
addressed. Invalid or unknown IDs return errors rather than creating children.
Children cannot delegate further and are not listed by `list-session`. A parent response can launch several children in one model turn; each launch returns without waiting for the worker.

Finishing an interaction leaves the child open. A timeout, empty response, or
exhausted turn budget produces `status: failed` while the session remains
`running`, so the parent can send another prompt with a fresh interaction
budget. Incomplete historical tool calls prevent continuation and are never
replayed. Closing a child is idempotent; closed or failed sessions, including
old one-shot children, cannot be resumed. Old sessions without a name still
load and display as `未命名子代理` alongside their ID.

Closing a parent through `close_session`, `/close`, or `close-session` also
closes its direct children. Existing child terminal states are preserved;
other children take the parent's terminal state. Cleanup continues if one
child fails to close and reports the failures; calling close again retries
cleanup even when the parent is already closed. `/quit` and `/exit` preserve
both parent and child sessions for later use.

Background workers are process-local daemon threads. If the process exits, queued or running executions become `interrupted` when inspected later; the child session remains available for a new prompt. At most `runtime.max_background_subagents` workers run at once (default 4), with additional jobs queued FIFO.

The loop exposes eight tools to the model: `bash`, `read_file`,
`write_file`, `edit_file`, `glob`, `read_image`, `todo_write`, and `task`. File-tool paths and glob matches stay
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
`read_image`, and `todo_write` for workspace tasks, plus `task` for delegating
focused exploration or self-contained subtasks to a persistent subagent,
plus vision input for web-format images.
Remote-sensing semantics stay behind the MCP boundary. Use `--workdir` to select the workspace for a session;
it may be placed before or after a subcommand, and the subcommand value takes
precedence.

Tool execution is silent inside the loop; the CLI injects a console callback
that prints hook notices to stdout — the working directory at prompt submit,
each tool invocation with a short argument preview before execution, and a
tool-call summary at stop — colored on a terminal, plain text when piped,
matching the reference interactive client.
Subagent notices use `[sub:<name>|<session_id>]` to identify their source;
names are redacted before persistence or display, and display delimiters in
names are escaped.

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
