import asyncio
import json
from pathlib import Path
from uuid import uuid4

import pytest
from typer.testing import CliRunner

from rs_harness.cli import app
from rs_harness.config import Config, ConfigError
from rs_harness.sessions.catalog import SessionCatalog
from tests.images import write_test_image


def test_create_and_list_session(config, tmp_path):
    runner = CliRunner()
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[model]",
                'model = "test-model"',
                'api_key = "test-key"',
                "",
                "[runtime]",
                f'sessions_dir = "{config.runtime.sessions_dir}"',
                f'system_prompt_file = "{config.runtime.system_prompt_file}"',
                'checkpoint_file = "checkpoints.sqlite3"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    created = runner.invoke(app, ["create-session", "--config", str(config_path), "--json"])
    assert created.exit_code == 0, created.output
    assert json.loads(created.output)["session_id"]
    assert "thread_id" not in json.loads(created.output)
    listed = runner.invoke(app, ["list-session", "--config", str(config_path), "--json"])
    assert listed.exit_code == 0, listed.output
    assert "session_id" in listed.output
    assert all("thread_id" not in meta for meta in json.loads(listed.output))


@pytest.mark.parametrize("text", ["", "\n", "/quit\n", "/exit\n"])
def test_empty_or_exit_pipe_input_leaves_no_session(config, tmp_path, text):
    result = CliRunner().invoke(app, ["--config", str(tmp_path / "config.toml")], input=text)
    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.output
    assert "Warning:" not in result.output
    assert SessionCatalog.list(config) == []


def test_help_shows_reasoning_options(tmp_path, config):
    result = CliRunner().invoke(
        app, ["--config", str(tmp_path / "config.toml")], input="/help\n/quit\n",
    )

    assert result.exit_code == 0, result.output
    assert "/reasoning [none|minimal|low|medium|high|xhigh|max|default]" in result.output


def test_reasoning_command_persists_and_resets_without_a_session(config, tmp_path):
    path = tmp_path / "config.toml"
    runner = CliRunner()

    set_result = runner.invoke(app, ["--config", str(path)], input="/help\n/reasoning\n/reasoning high\n/quit\n")
    assert set_result.exit_code == 0, set_result.output
    assert "/reasoning  show or set reasoning effort" in set_result.output
    assert "Reasoning effort: default" in set_result.output
    assert "Reasoning effort: high" in set_result.output
    assert Config.load(path).model.reasoning_effort == "high"
    assert SessionCatalog.list(config) == []

    reset_result = runner.invoke(app, ["--config", str(path)], input="/reasoning\n/reasoning default\n/quit\n")
    assert reset_result.exit_code == 0, reset_result.output
    assert "Reasoning effort: high" in reset_result.output
    assert Config.load(path).model.reasoning_effort is None
    assert "reasoning_effort" not in path.read_text(encoding="utf-8")


def test_reasoning_command_uses_relative_config_path(config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["--config", "config.toml"], input="/reasoning high\n/quit\n")

    assert result.exit_code == 0, result.output
    assert Config.load(tmp_path / "config.toml").model.reasoning_effort == "high"


@pytest.mark.parametrize("invalid", ["LOW", "auto", "high extra", "default extra"])
def test_reasoning_command_rejects_invalid_values(config, tmp_path, invalid):
    path = tmp_path / "config.toml"
    before = path.read_bytes()

    result = CliRunner().invoke(app, ["--config", str(path)], input=f"/reasoning {invalid}\n/quit\n")

    assert result.exit_code == 0, result.output
    assert "Usage: /reasoning" in result.output
    assert path.read_bytes() == before
    assert SessionCatalog.list(config) == []


def test_reasoning_command_applies_to_next_question(config, tmp_path, monkeypatch):
    from rs_harness import application, cli

    seen = []

    async def ask_with_capture(current_config, session_id, text, **kwargs):
        seen.append(current_config.model.reasoning_effort)
        return application.InteractionResult(
            session_id=session_id, interaction_id="test-interaction",
            status="succeeded", session_status="running", answer="done",
        )

    monkeypatch.setattr(cli, "ask", ask_with_capture)
    path = tmp_path / "config.toml"
    result = CliRunner().invoke(
        app, ["--config", str(path)],
        input="/reasoning high\nfirst question\n/reasoning none\nsecond question\n/quit\n",
    )

    assert result.exit_code == 0, result.output
    assert seen == ["high", "none"]
    assert Config.load(path).model.reasoning_effort == "none"


def test_reasoning_save_failure_keeps_current_setting(config, tmp_path, monkeypatch):
    from rs_harness import cli

    path = tmp_path / "config.toml"
    before = path.read_bytes()

    def fail_save(*args):
        raise ConfigError("Unable to save model.reasoning_effort")

    monkeypatch.setattr(cli, "save_reasoning_effort", fail_save)
    result = CliRunner().invoke(app, ["--config", str(path)], input="/reasoning high\n/reasoning\n/quit\n")

    assert result.exit_code == 0, result.output
    assert "Error: Unable to save model.reasoning_effort" in result.output
    assert "Reasoning effort: default" in result.output
    assert path.read_bytes() == before
    assert SessionCatalog.list(config) == []


@pytest.mark.parametrize("extra", [[], [""], ["   "]])
def test_empty_ask_creates_no_session(config, tmp_path, extra):
    result = CliRunner().invoke(app, ["ask", "--config", str(tmp_path / "config.toml"), *extra])
    assert result.exit_code == 0
    assert SessionCatalog.list(config) == []


def test_root_workdir_applies_to_ask(config, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from rs_harness import application, cli

    workdir = tmp_path / "requested-workdir"
    created = []

    def create_session(config, *, workdir=None):
        created.append(workdir)
        return SimpleNamespace(session_id="session-id")

    async def ask(config, session_id, message, **kwargs):
        return application.InteractionResult(
            session_id=session_id, interaction_id="interaction-id",
            status="succeeded", session_status="running", answer="done",
        )

    monkeypatch.setattr(cli, "create_session", create_session)
    monkeypatch.setattr(cli, "ask", ask)
    result = CliRunner().invoke(app, [
        "--workdir", str(workdir), "ask", "hello",
        "--config", str(tmp_path / "config.toml"),
    ])

    assert result.exit_code == 0, result.output
    assert created == [str(workdir)]


def test_explicit_child_default_config_overrides_root_config(config, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from rs_harness import cli

    root_config = tmp_path / "custom.toml"
    root_config.write_text("", encoding="utf-8")
    child_config = tmp_path / "config" / "default.toml"
    child_config.parent.mkdir()
    child_config.write_text("", encoding="utf-8")
    loaded = []

    def load(path):
        loaded.append(path)
        return config

    def create_session(config, *, workdir=None):
        return SimpleNamespace(session_id="session-id", path=tmp_path / "session")

    monkeypatch.setattr(cli, "_load", load)
    monkeypatch.setattr(cli, "create_session", create_session)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, [
        "--config", str(root_config), "create-session",
        "--config", "config/default.toml", "--json",
    ])

    assert result.exit_code == 0, result.output
    assert loaded == [Path("config/default.toml")]


def test_invalid_image_json_creates_no_session(config, tmp_path):
    result = CliRunner().invoke(app, ["ask", "--config", str(tmp_path / "config.toml"), "--image", "missing.png", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"]["type"] == "InputError"
    assert result.stderr == ""
    assert SessionCatalog.list(config) == []


def test_print_action_escapes_bash_control_characters(capsys):
    from rs_harness import cli

    cli._print_action({"name": "bash", "args": {"command": "printf '\x1b]52;c;secret\x07\n'"}})

    output = capsys.readouterr().err
    assert "\x1b" not in output
    assert "\\u001b" in output
    assert "\\u0007" in output
    assert "\\n" in output


def test_cli_progress_previews_tools_without_sensitive_values(capsys):
    from rs_harness.progress import CliProgress

    async def run():
        progress = CliProgress()
        run_id = uuid4()
        await progress.on_tool_start(
            {"name": "bash"}, "", run_id=run_id,
            inputs={"command": "printf '\x1b'; API_KEY=secret-value"},
        )
        await progress.on_tool_end("secret output", run_id=run_id)
        write_run = uuid4()
        await progress.on_tool_start(
            {"name": "write_file"}, "", run_id=write_run,
            inputs={"path": "note.txt", "content": "private content"},
        )
        await progress.on_tool_end("done", run_id=write_run)
        long_run = uuid4()
        await progress.on_tool_start(
            {"name": "bash"}, "", run_id=long_run,
            inputs={"command": "x" * 1000},
        )
        await progress.on_tool_end("done", run_id=long_run)

    asyncio.run(run())
    output = capsys.readouterr().err
    assert "[Tool] Running bash (" in output
    assert "secret-value" not in output
    assert "\\u001b" in output
    assert "secret output" not in output
    assert "private content" not in output
    assert output.count('"path":"note.txt"') == 1
    assert output.count("15 chars") == 1
    assert output.count("…") == 1
    assert max(map(len, output.splitlines())) < 200
    assert "\x1b" not in output


@pytest.mark.parametrize("command", [
    "OPENAI_API_KEY=credential1 python job.py",
    "AWS_SECRET_ACCESS_KEY=credential1 python job.py",
    "app_access_token=credential1 python job.py",
    "client --api-key=credential1",
    "API_KEY='credential1 credential2' python job.py",
    'API_KEY="credential1 credential2" python job.py',
    r"API_KEY=credential1\ credential2 python job.py",
    r'API_KEY="credential1\" credential2" python job.py',
    "API_KEY='credential1'\" credential2\" python job.py",
    "API_KEY=credential1; PASSWORD=credential2 python job.py",
    "curl -H 'Authorization: Bearer credential1 credential2' https://example.com",
    'curl --header "authorization: Basic credential1" https://example.com',
    "curl --header='Authorization: Bearer credential1' https://example.com",
    "curl -H'Authorization: Bearer credential1' https://example.com",
    r"curl -H Authorization:\ Bearer\ credential1 https://example.com",
])
@pytest.mark.parametrize("outcome", ["success", "error_status", "exception"])
async def test_tool_notices_reuse_redacted_operation(capsys, command, outcome):
    from langchain_core.messages import ToolMessage
    from rs_harness.progress import CliProgress

    progress = CliProgress()
    run_id = uuid4()
    inputs = {"command": command}
    await progress.on_tool_start({"name": "bash"}, "", run_id=run_id, inputs=inputs)
    assert inputs == {"command": command}
    inputs["command"] = "changed after start"
    if outcome == "success":
        await progress.on_tool_end("private result", run_id=run_id)
    elif outcome == "error_status":
        await progress.on_tool_end(ToolMessage(content="failed", tool_call_id="call", status="error"), run_id=run_id)
    else:
        await progress.on_tool_error(RuntimeError("failed"), run_id=run_id)

    output = capsys.readouterr().err
    assert len(output.splitlines()) == 1
    end = output.splitlines()[0]
    summary = end.split(") ", 1)[1]
    assert "[redacted]" in summary
    assert "credential1" not in output
    assert "credential2" not in output
    assert "changed after start" not in output
    assert "private result" not in output
    if outcome == "success":
        assert len(summary) <= 160
        assert "Running bash (" in end
        assert end.endswith(") " + summary)
    else:
        kind = "ToolError" if outcome == "error_status" else "RuntimeError"
        suffix = f" ({kind}): failed"
        summary = summary.removesuffix(suffix)
        assert len(summary) <= 160
        assert "Failed bash (" in end
        assert end.endswith(") " + summary + suffix)


@pytest.mark.parametrize("command", [
    "API_KEY='credential1 credential2",
    'API_KEY="credential1 credential2',
    "API_KEY=credential1\\",
    "API_KEY=$(printf 'credential1 credential2') python job.py",
    "API_KEY=`printf 'credential1 credential2'` python job.py",
    "bash -c 'export API_KEY=credential1; python job.py'",
    'curl -H "Authorization: Bearer credential1',
])
def test_ambiguous_secret_boundary_hides_command(command):
    from rs_harness.progress import _arguments

    assert json.loads(_arguments({"command": command})) == {"command": "[redacted command]"}


@pytest.mark.parametrize("command", [
    "ls -la /Users/hgz/WorkSpace/RS-Harness/ 2>&1 | head -40",
    "ls -la /Users/hgz/WorkSpace/ 2>&1 | head -40",
    "printf '%s\\n' 'hello world' && ls *.txt",
])
async def test_tool_notices_preserve_non_sensitive_operation(capsys, command):
    from rs_harness.progress import CliProgress

    progress = CliProgress()
    run_id = uuid4()
    await progress.on_tool_start({"name": "bash"}, "", run_id=run_id, inputs={"command": command})
    await progress.on_tool_end("done", run_id=run_id)

    output = capsys.readouterr().err
    assert len(output.splitlines()) == 1
    end = output.splitlines()[0]
    summary = json.dumps({"command": command}, ensure_ascii=False, separators=(",", ":"))
    assert end.startswith("[Tool] Running bash (")
    assert end.endswith(") " + summary)


@pytest.mark.parametrize("as_json", [False, True])
def test_ask_progress_stays_on_stderr(config, tmp_path, monkeypatch, as_json):
    from langchain_core.messages import AIMessage
    from rs_harness import application, cli
    from tests.models import ScriptedModel

    model = ScriptedModel([
        AIMessage(content="private reasoning", tool_calls=[{
            "name": "write_file", "args": {"path": "note.txt", "content": "private file body"},
            "id": "write", "type": "tool_call",
        }]),
        AIMessage(content="done"),
    ])

    async def ask_with_model(config, session_id, text, **kwargs):
        return await application.ask(config, session_id, text, model=model, **kwargs)

    monkeypatch.setattr(cli, "ask", ask_with_model)
    arguments = ["ask", "write", "--config", str(tmp_path / "config.toml"), "--workdir", str(tmp_path)]
    result = CliRunner().invoke(app, arguments + (["--json"] if as_json else []))

    assert result.exit_code == 0, result.output
    assert (tmp_path / "note.txt").read_text() == "private file body"
    if as_json:
        assert json.loads(result.stdout)["answer"] == "done"
        assert result.stderr == ""
    else:
        assert result.stdout == "done\n"
        assert "[Tool] Running write_file (" in result.stderr
        assert "[Middleware]" not in result.stderr
        assert "[Model] Call 2" in result.stderr
        assert "note.txt" in result.stderr
    assert "private reasoning" not in result.stderr
    assert "private file body" not in result.stderr


def test_ask_collects_approval_and_resumes(config, tmp_path, monkeypatch):
    from rs_harness import application, cli

    session = application.create_session(config, workdir=str(tmp_path))
    interrupt = {
        "id": "interrupt-1",
        "action_requests": [{"name": "bash", "args": {"command": "rm remove.txt"}}],
        "review_configs": [{"allowed_decisions": ["approve", "reject"]}],
    }

    class FakePrompt:
        async def prompt_async(self, message):
            cli.typer.echo(message, err=True)
            return "yes"

    async def ask_with_model(config, session_id, text, **kwargs):
        return application.InteractionResult(
            session_id=session_id, interaction_id="interaction-1", status="interrupted",
            session_status="running", interrupts=[interrupt],
        )

    async def resume_with_model(config, session_id, decisions, **kwargs):
        assert decisions == {"interrupt-1": {"decisions": [{"type": "approve"}]}}
        return application.InteractionResult(
            session_id=session_id, interaction_id="interaction-1", status="succeeded",
            session_status="running", answer="done",
        )

    monkeypatch.setattr(cli, "resume", resume_with_model)
    monkeypatch.setattr(cli, "ask", ask_with_model)
    monkeypatch.setattr(cli, "_can_collect_approvals", lambda: True)
    monkeypatch.setattr(cli, "PromptSession", lambda **kwargs: FakePrompt())
    result = CliRunner().invoke(app, [
        "ask", "remove", "--config", str(tmp_path / "config.toml"), "--session-id", session.session_id,
    ])

    assert result.exit_code == 0, result.output
    assert result.stdout == "done\n"
    assert "Execute? [yes/no]>" in result.stderr
    assert "rm remove.txt" in result.stderr


def test_ask_collects_each_action_and_new_interrupts(config, tmp_path, monkeypatch):
    from rs_harness import application, cli

    def remove(name):
        return {"name": "bash", "args": {"command": f"rm {name}"}, "id": name, "type": "tool_call"}

    first_interrupt = {
        "id": "interrupt-1",
        "action_requests": [remove("one"), remove("two")],
        "review_configs": [{"allowed_decisions": ["approve", "reject"]}] * 2,
    }
    second_interrupt = {
        "id": "interrupt-2",
        "action_requests": [remove("three")],
        "review_configs": [{"allowed_decisions": ["approve", "reject"]}],
    }
    submitted = []

    async def ask_with_model(config, session_id, text, **kwargs):
        return application.InteractionResult(
            session_id=session_id, interaction_id="interaction-1", status="interrupted",
            session_status="running", interrupts=[first_interrupt],
        )

    async def resume_with_model(config, session_id, decisions, **kwargs):
        submitted.append(decisions)
        interrupted = len(submitted) == 1
        return application.InteractionResult(
            session_id=session_id, interaction_id="interaction-1", status="interrupted" if interrupted else "succeeded",
            session_status="running", interrupts=[second_interrupt] if interrupted else [],
            answer=None if interrupted else "done",
        )

    monkeypatch.setattr(cli, "ask", ask_with_model)
    monkeypatch.setattr(cli, "resume", resume_with_model)
    session = application.create_session(config, workdir=str(tmp_path))
    choices = iter(("yes", "no", "no"))

    class FakePrompt:
        async def prompt_async(self, message):
            return next(choices)

    monkeypatch.setattr(cli, "_can_collect_approvals", lambda: True)
    monkeypatch.setattr(cli, "PromptSession", lambda **kwargs: FakePrompt())

    result = CliRunner().invoke(app, [
        "ask", "remove", "--config", str(tmp_path / "config.toml"), "--session-id", session.session_id,
    ])

    assert result.exit_code == 0, result.output
    assert result.stderr.count("Waiting for approval (2 items)") == 1
    assert result.stderr.count("Waiting for approval (1 item)") == 1
    assert submitted == [
        {"interrupt-1": {"decisions": [{"type": "approve"}, {"type": "reject"}]}},
        {"interrupt-2": {"decisions": [{"type": "reject"}]}},
    ]


def test_cli_image_only_and_resume_keep_workdir(config, tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage
    from rs_harness import application, cli
    from tests.models import ScriptedModel

    workdir = tmp_path / "workspace"
    workdir.mkdir()
    write_test_image(workdir / "sample.png")
    model = ScriptedModel([AIMessage(content="seen"), AIMessage(content="remembered")])

    async def ask_with_model(loaded_config, session_id, text, **kwargs):
        return await application.ask(loaded_config, session_id, text, model=model, **kwargs)

    monkeypatch.setattr(cli, "ask", ask_with_model)
    monkeypatch.chdir(tmp_path)
    options = ["--config", str(tmp_path / "config.toml"), "--json"]
    result = CliRunner().invoke(app, ["ask", *options, "--workdir", str(workdir), "--image", "sample.png", "--image", "https://example.com/sample.png"])
    assert result.exit_code == 0, result.output
    session_id = json.loads(result.stdout)["session_id"]
    blocks = model.inputs[0][-1].content
    assert [block["type"] for block in blocks] == ["image", "image"]
    continued = CliRunner().invoke(app, ["ask", "follow up", *options, "--session-id", session_id])
    assert continued.exit_code == 0, continued.output
    assert model.inputs[1][1].content == blocks
    failed = CliRunner().invoke(app, ["ask", *options, "--session-id", session_id, "--image", "missing.png"])
    assert failed.exit_code == 1
    assert model.call_count == 2
    changed_workdir = CliRunner().invoke(app, ["ask", "hello", *options, "--session-id", session_id, "--workdir", str(tmp_path)])
    assert changed_workdir.exit_code == 1
    assert "session_workdir_mismatch" in json.loads(changed_workdir.stdout)["error"]["message"]


@pytest.mark.parametrize("already_pending", [False, True])
def test_pipe_never_consumes_next_line_as_approval(config, tmp_path, monkeypatch, already_pending):
    from langchain_core.messages import AIMessage
    from rs_harness import application, cli
    from tests.models import ScriptedModel

    session = application.create_session(config, workdir=str(tmp_path))
    (tmp_path / "file.txt").write_text("keep")
    model = ScriptedModel([AIMessage(content="", tool_calls=[{
        "name": "bash", "args": {"command": "rm file.txt"}, "id": "delete", "type": "tool_call",
    }])])
    async def ask_with_model(config, session_id, text, **kwargs):
        return await application.ask(config, session_id, text, model=model, **kwargs)
    monkeypatch.setattr(cli, "ask", ask_with_model)
    if already_pending:
        asyncio.run(ask_with_model(config, session.session_id, "run"))
    result = CliRunner().invoke(app, [
        "--config", str(tmp_path / "config.toml"), "--session-id", session.session_id,
    ], input="run\nyes\n/close\n")
    assert result.exit_code == 2, result.output
    assert "Action 1:" not in result.output
    assert "rm file.txt" in result.stderr
    assert result.stderr.count("Waiting for approval (1 item)") == 1
    assert "Decisions:" not in result.stderr
    assert (tmp_path / "file.txt").read_text() == "keep"
    assert model.call_count == 1
    assert SessionCatalog.open(config, session.session_id).meta.status == "running"


def test_ask_json_reports_pending_gate(config, tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage
    from rs_harness import application, cli
    from tests.models import ScriptedModel

    model = ScriptedModel([AIMessage(content="", tool_calls=[{
        "name": "bash", "args": {"command": "rm file.txt"}, "id": "delete", "type": "tool_call",
    }])])
    async def ask_with_model(config, session_id, text, **kwargs):
        return await application.ask(config, session_id, text, model=model, **kwargs)
    monkeypatch.setattr(cli, "ask", ask_with_model)
    options = ["--config", str(tmp_path / "config.toml"), "--json"]
    first = CliRunner().invoke(app, ["ask", "remove", *options, "--workdir", str(tmp_path)])
    assert first.exit_code == 2, first.output
    pending = json.loads(first.stdout)
    second = CliRunner().invoke(app, ["ask", "new", *options, "--session-id", pending["session_id"]])
    assert second.exit_code == 1, second.output
    result = json.loads(second.stdout)
    assert result["error"]["type"] == "session_pending_approval"
    assert result["interrupts"] == pending["interrupts"]
    assert result["interaction_id"] == pending["interaction_id"]
    assert first.stderr == second.stderr == ""


def test_ask_json_can_collect_approval_and_pause_again(config, tmp_path, monkeypatch):
    from rs_harness import application, cli

    session = application.create_session(config, workdir=str(tmp_path))
    def interrupt(identifier, filename):
        return {
            "id": identifier,
            "action_requests": [{"name": "bash", "args": {"command": f"rm {filename}"}}],
            "review_configs": [{"allowed_decisions": ["approve", "reject"]}],
        }

    prompts = iter(("no", KeyboardInterrupt()))

    class FakePrompt:
        async def prompt_async(self, message):
            prompt = next(prompts)
            if isinstance(prompt, BaseException):
                raise prompt
            cli.typer.echo(message, err=True)
            return prompt

    async def ask_with_model(config, session_id, text, **kwargs):
        return application.InteractionResult(
            session_id=session_id, interaction_id="interaction-1", status="interrupted",
            session_status="running", interrupts=[interrupt("interrupt-1", "one.txt")],
        )

    async def resume_with_model(config, session_id, decisions, **kwargs):
        return application.InteractionResult(
            session_id=session_id, interaction_id="interaction-1", status="interrupted",
            session_status="running", interrupts=[interrupt("interrupt-2", "two.txt")],
        )

    monkeypatch.setattr(cli, "resume", resume_with_model)
    monkeypatch.setattr(cli, "ask", ask_with_model)
    monkeypatch.setattr(cli, "_can_collect_approvals", lambda: True)
    monkeypatch.setattr(cli, "PromptSession", lambda **kwargs: FakePrompt())
    result = CliRunner().invoke(app, [
        "ask", "run", "--config", str(tmp_path / "config.toml"), "--session-id", session.session_id, "--json",
    ])
    assert result.exit_code == 2, result.output
    assert "Execute? [yes/no]>" in result.stderr
    pending = json.loads(result.stdout)
    assert pending["interrupts"][0]["action_requests"][0]["args"] == {"command": "rm two.txt"}
