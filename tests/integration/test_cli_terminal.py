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
def terminal_cli(config, tmp_path):
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 40, 0, 0))
    script = """
import json
from rs_harness import cli
from rs_harness.application import InteractionResult

async def record_input(config, session_id, text):
    print("RSH_INPUT=" + json.dumps(text, ensure_ascii=False), flush=True)
    return InteractionResult(
        session_id=session_id, interaction_id="test", status="succeeded",
        answer="received", session_status="running",
    )

cli.ask = record_input
cli.main()
"""
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", script, "--config", str(tmp_path / "config.toml")],
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
        read_until(b"rsh>")
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
