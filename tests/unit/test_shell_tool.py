import asyncio
import json
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest
from langchain_core.tools import ToolException

from rs_harness.context import AgentContext
from rs_harness.tools.shell import bash


@pytest.fixture
def runtime(tmp_path):
    return SimpleNamespace(context=AgentContext(str(tmp_path), bash_timeout_s=2))


async def test_bash_workdir_stdout_stderr_and_exit(runtime, tmp_path):
    result = await bash.coroutine("pwd; printf '你好'; printf 'error' >&2", runtime)
    assert str(tmp_path) in result["stdout"]
    assert "你好" in result["stdout"]
    assert result["stderr"] == "error"
    assert result["exit_code"] == 0
    assert result["timed_out"] is result["truncated"] is False
    with pytest.raises(ToolException) as failure:
        await bash.coroutine("printf 'failure' >&2; exit 7", runtime)
    assert json.loads(str(failure.value))["exit_code"] == 7


async def test_bash_output_is_bounded_and_both_pipes_are_drained(runtime, tmp_path):
    runtime.context = AgentContext(str(tmp_path), max_tool_output_bytes=101)
    script = "import os; os.write(1, b'a' * 1000000); os.write(2, b'b' * 1000000)"
    result = await bash.coroutine(f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}", runtime)
    assert len(result["stdout"].encode()) + len(result["stderr"].encode()) <= 101
    assert result["truncated"] is True


def assert_stopped(pid):
    process = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    assert process.returncode != 0 or process.stdout.strip().startswith("Z"), process.stdout


@pytest.mark.parametrize("shell_exits", [False, True])
async def test_timeout_stops_descendants_holding_pipes(runtime, tmp_path, shell_exits):
    runtime.context = AgentContext(str(tmp_path), bash_timeout_s=0.2)
    command = "sleep 30 & echo $! > child.pid; " + ("exit 0" if shell_exits else "wait")
    with pytest.raises(ToolException) as failure:
        await asyncio.wait_for(bash.coroutine(command, runtime), 3)
    assert json.loads(str(failure.value))["timed_out"] is True
    assert_stopped(int((tmp_path / "child.pid").read_text()))


async def test_cancellation_cleans_up_and_propagates(runtime, tmp_path):
    task = asyncio.create_task(bash.coroutine("sleep 30 & echo $! > child.pid; wait", runtime))
    async with asyncio.timeout(2):
        while not (tmp_path / "child.pid").exists():
            await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert_stopped(int((tmp_path / "child.pid").read_text()))


async def test_start_failure_is_tool_error(runtime, tmp_path):
    runtime.context = AgentContext(str(tmp_path / "missing"))
    with pytest.raises(ToolException, match="could not start"):
        await bash.coroutine("pwd", runtime)


@pytest.mark.parametrize("command", ["printf '你好'", r"printf '\377'"])
async def test_decoding_never_exceeds_the_byte_budget(runtime, tmp_path, command):
    runtime.context = AgentContext(str(tmp_path), max_tool_output_bytes=1)
    result = await bash.coroutine(command, runtime)
    assert len(result["stdout"].encode()) <= 1
    assert result["truncated"] is True


async def test_timeout_escalates_for_processes_ignoring_sigterm(runtime, tmp_path):
    runtime.context = AgentContext(str(tmp_path), bash_timeout_s=0.2)
    command = "trap '' TERM; echo $$ > shell.pid; sleep 30 & echo $! > child.pid; wait"
    with pytest.raises(ToolException):
        await asyncio.wait_for(bash.coroutine(command, runtime), 3)
    assert_stopped(int((tmp_path / "shell.pid").read_text()))
    assert_stopped(int((tmp_path / "child.pid").read_text()))
