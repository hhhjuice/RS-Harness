import fcntl
import json
import os
import pty
import select
import struct
import subprocess
import sys
import termios
import time

import pytest

from rs_harness.sessions.catalog import SessionCatalog


@pytest.fixture
def terminal_cli(config, tmp_path, request):
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 40, 0, 0))
    settings = getattr(request, "param", None)
    script = (settings["script"] if isinstance(settings, dict) else settings) or """
import json
from rs_harness import cli
from rs_harness.application import InteractionResult

async def record_input(config, session_id, text, **kwargs):
    print("RSH_INPUT=" + json.dumps(text, ensure_ascii=False), flush=True)
    return InteractionResult(
        session_id=session_id, interaction_id="test", status="succeeded",
        answer="received", session_status="running",
    )

cli.ask = record_input
cli.main()
"""
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", script, "--config", str(tmp_path / "config.toml"), "--workdir", str(tmp_path)],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env={**os.environ, "TERM": "xterm-256color", "PROMPT_TOOLKIT_NO_CPR": "1"},
    )
    os.close(slave)
    pending = b""

    def read_until(marker):
        nonlocal pending
        deadline = time.monotonic() + 10
        while marker not in pending:
            if time.monotonic() >= deadline:
                pytest.fail(f"Terminal timed out waiting for {marker!r}: {pending!r}")
            ready, _, _ = select.select([master], [], [], 0.1)
            if ready:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    chunk = b""
                if not chunk:
                    pytest.fail(f"Terminal closed while waiting for {marker!r}: {pending!r}")
                pending += chunk
        end = pending.index(marker) + len(marker)
        received, pending = pending[:end], pending[end:]
        return received

    try:
        read_until(settings["marker"] if isinstance(settings, dict) else b"rsh>")
        yield master, process, read_until
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        os.close(master)


def submit(terminal_cli, keys):
    master, _, read_until = terminal_cli
    os.write(master, (keys + "\r").encode("utf-8"))
    read_until(b"RSH_INPUT=")
    text = json.loads(read_until(b"\n").decode("utf-8"))
    read_until(b"rsh>")
    return text


def test_terminal_prompt_is_blue(terminal_cli):
    master, _, read_until = terminal_cli
    os.write(master, b"blue prompt\r")
    read_until(b"RSH_INPUT=")
    read_until(b"\n")
    displayed = read_until(b"rsh>")
    assert b"\x1b[0;34m" in displayed


@pytest.mark.parametrize(("keys", "expected"), [
    ("中文测试\x1b[D\x1b[D\x7f国", "中国测试"),
    ("中文测试\x1b[D\x1b[D\x08国", "中国测试"),
    ("中文测试\x1b[D\x1b[D\x1b[3~验", "中文验试"),
    ("A中文B\x1b[D\x1b[D\x1b[C\x7f字", "A中字B"),
    ("中文\x1b[H开\x1b[F尾", "开中文尾"),
    ("按\x1b[C\x1b[C\x1b[A\x1b[D\x1b[D\x1b[B\x1b[C\x1b[D", "按"),
    ("中文" * 30 + "AB\x1b[D\x1b[D\x7f试", "中文" * 29 + "中试AB"),
    ("\x1b[200~第一行中文\n第二行中文\x1b[201~", "第一行中文\n第二行中文"),
], ids=["backspace", "control-h", "delete", "mixed-width", "home-end", "arrows", "wrapped-line", "paste"])
def test_terminal_edits_unicode_without_submitting_escape_sequences(terminal_cli, keys, expected):
    assert submit(terminal_cli, keys) == expected


def test_terminal_history_preserves_chinese_and_current_draft(terminal_cli):
    assert submit(terminal_cli, "上一条中文") == "上一条中文"
    assert submit(terminal_cli, "\x1b[A\x1b[D\x7f字") == "上一条字文"
    assert submit(terminal_cli, "新问题\x1b[A\x1b[B") == "新问题"


@pytest.mark.parametrize("keys", ["\x04", "未提交的问题\x03"], ids=["eof", "interrupt"])
def test_terminal_interrupt_or_eof_preserves_session(terminal_cli, config, keys):
    assert submit(terminal_cli, "已提交的问题") == "已提交的问题"
    master, process, _ = terminal_cli
    os.write(master, keys.encode("utf-8"))
    assert process.wait(timeout=5) == 0
    sessions = SessionCatalog.list(config)
    assert len(sessions) == 1
    assert sessions[0].meta.status == "running"


PERMISSION_SCRIPT = '''
from langchain_core.messages import AIMessage
from rs_harness import cli, application
from tests.models import ScriptedModel

model = ScriptedModel([
    AIMessage(content="", tool_calls=[
        {"name":"bash", "args":{"command":"rm first.txt # \\u001b[31m\\n"}, "id":"first", "type":"tool_call"},
        {"name":"bash", "args":{"command":"rm second.txt"}, "id":"second", "type":"tool_call"},
    ]), AIMessage(content="permission-result"),
])
original_build = application.build_agent
def build(config, *, workdir, checkpointer, model=model):
    return original_build(config, workdir=workdir, checkpointer=checkpointer, model=model or globals()["model"])
application.build_agent = build
cli.main()
'''


def begin_review(terminal_cli, tmp_path):
    for name in ("first.txt", "second.txt"):
        (tmp_path / name).write_text("keep")
    master, _, read_until = terminal_cli
    os.write(master, b"remove\r")
    displayed = read_until("Execute? [yes/no]>".encode())
    assert displayed.count("Waiting for approval (2 items)".encode()) == 1
    assert "Decisions:".encode() not in displayed
    assert b"\x1b[0;33m" in displayed
    assert "Tool: bash".encode() in displayed
    assert "Operation: rm first.txt".encode() in displayed
    assert b'"workdir"' not in displayed
    assert b'"id"' not in displayed
    return master, read_until


@pytest.mark.parametrize("terminal_cli", [PERMISSION_SCRIPT], indirect=True)
def test_terminal_collects_all_decisions_with_no_default_approval(terminal_cli, tmp_path):
    master, read_until = begin_review(terminal_cli, tmp_path)
    os.write(master, b"\r")
    read_until("Enter y/yes or n/no.".encode())
    read_until("Execute? [yes/no]>".encode())
    os.write(master, b"y\r")
    displayed = read_until("Execute? [yes/no]>".encode())
    assert "Decisions:".encode() not in displayed
    assert "Waiting for approval (".encode() not in displayed
    assert (tmp_path / "first.txt").exists()
    os.write(master, b"n\r")
    displayed = read_until(b"permission-result")
    assert displayed.count("Decisions: 1 approved, 1 rejected".encode()) == 1
    assert "failed".encode() not in displayed
    assert "[Model] Call 2".encode() in displayed
    assert "[Model] Call 1".encode() not in displayed
    read_until(b"rsh>")
    assert not (tmp_path / "first.txt").exists()
    assert (tmp_path / "second.txt").read_text() == "keep"
    os.write(master, b"/quit\r")
    assert terminal_cli[1].wait(timeout=5) == 0


@pytest.mark.parametrize("terminal_cli", [PERMISSION_SCRIPT], indirect=True)
def test_terminal_model_count_continues_across_approval_and_resets_per_input(terminal_cli, tmp_path):
    master, read_until = begin_review(terminal_cli, tmp_path)
    os.write(master, b"y\r")
    read_until("Execute? [yes/no]>".encode())
    os.write(master, b"n\r")
    interaction = read_until(b"rsh>")
    assert interaction.count("Decisions: 1 approved, 1 rejected".encode()) == 1
    assert "[Model] Call 1".encode() not in interaction
    assert interaction.index("Decisions:".encode()) < interaction.index("[Model] Call 2".encode())
    os.write(master, b"another\r")
    read_until("[Model] Call 1".encode())
    read_until(b"rsh>")
    os.write(master, b"/quit\r")
    assert terminal_cli[1].wait(timeout=5) == 0


@pytest.mark.parametrize("terminal_cli", [PERMISSION_SCRIPT], indirect=True)
@pytest.mark.parametrize("keys", [b"\x03", b"\x04"])
def test_terminal_partial_approval_is_discarded(terminal_cli, config, tmp_path, keys):
    import asyncio
    from rs_harness.application import get_pending_approvals

    master, read_until = begin_review(terminal_cli, tmp_path)
    os.write(master, b"y\r")
    read_until("Execute? [yes/no]>".encode())
    os.write(master, keys)
    displayed = read_until(b"Use rsh --session-id")
    assert "Decisions:".encode() not in displayed
    assert "failed".encode() not in displayed
    assert terminal_cli[1].wait(timeout=5) == 0
    assert (tmp_path / "first.txt").read_text() == "keep"
    assert (tmp_path / "second.txt").read_text() == "keep"
    session = SessionCatalog.list(config)[0]
    pending = asyncio.run(get_pending_approvals(config, session.session_id))
    assert len(pending.interrupts[0]["action_requests"]) == 2
    assert "approval_submitted" not in session.trajectory_path.read_text()


@pytest.mark.parametrize("terminal_cli", [PERMISSION_SCRIPT], indirect=True)
def test_terminal_treats_slash_commands_as_invalid_approval_input(terminal_cli, config, tmp_path):
    master, read_until = begin_review(terminal_cli, tmp_path)
    for command in (b"/status\r", b"/reasoning high\r", b"/close\r", b"/quit\r"):
        os.write(master, command)
        read_until("Enter y/yes or n/no.".encode())
        read_until("Execute? [yes/no]>".encode())
    os.write(master, b"n\r")
    read_until("Execute? [yes/no]>".encode())
    os.write(master, b"n\r")
    read_until(b"permission-result")
    read_until(b"rsh>")
    assert SessionCatalog.list(config)[0].meta.status == "running"
    assert (tmp_path / "first.txt").read_text() == "keep"
    assert (tmp_path / "second.txt").read_text() == "keep"
    assert "reasoning_effort" not in (tmp_path / "config.toml").read_text(encoding="utf-8")
    os.write(master, b"/quit\r")
    assert terminal_cli[1].wait(timeout=5) == 0


PERMISSION_STARTUP_SCRIPT = PERMISSION_SCRIPT.replace("cli.main()", '''
import asyncio
import sys
from pathlib import Path
from rs_harness.config import Config

config = Config.load(sys.argv[sys.argv.index("--config") + 1])
workdir = Path(sys.argv[sys.argv.index("--workdir") + 1])
for name in ("first.txt", "second.txt"):
    (workdir / name).write_text("keep")
session = application.create_session(config, workdir=str(workdir))
asyncio.run(application.ask(config, session.session_id, "earlier request", model=model))
sys.argv.extend(["--session-id", session.session_id])
cli.main()
''')


@pytest.mark.parametrize("terminal_cli", [{"script": PERMISSION_STARTUP_SCRIPT, "marker": "Execute? [yes/no]>".encode()}], indirect=True, ids=["pending-startup"])
def test_terminal_opens_existing_approval_before_accepting_text(terminal_cli, config, tmp_path):
    master, process, read_until = terminal_cli
    os.write(master, b"this must not become a new question\r")
    read_until("Enter y/yes or n/no.".encode())
    read_until("Execute? [yes/no]>".encode())
    os.write(master, b"y\r")
    read_until("Execute? [yes/no]>".encode())
    os.write(master, b"n\r")
    read_until(b"permission-result")
    read_until(b"rsh>")
    os.write(master, b"/quit\r")
    assert process.wait(timeout=5) == 0
    session = SessionCatalog.list(config)[0]
    records = [json.loads(line) for line in session.trajectory_path.read_text().splitlines()]
    assert sum(record["event"] == "user_input" for record in records) == 1
    assert not (tmp_path / "first.txt").exists()
    assert (tmp_path / "second.txt").exists()
