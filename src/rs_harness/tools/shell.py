"""Bounded asynchronous Bash execution on POSIX systems."""

import asyncio
import json
import os
import signal

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from ..context import AgentContext
from . import NonBlank


class BashInput(BaseModel):
    command: NonBlank = Field(description="Bash command to run in the session workspace, without interactive input.")


async def _terminate(process: asyncio.subprocess.Process) -> None:
    for termination_signal in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, termination_signal)
        except ProcessLookupError:
            break
        if termination_signal == signal.SIGTERM:
            await asyncio.sleep(0.1)
    await process.wait()


def _decode_output(output: bytearray) -> tuple[str, bool]:
    encoded = output.decode("utf-8", errors="replace").encode("utf-8")
    return encoded[:len(output)].decode("utf-8", errors="ignore"), len(encoded) > len(output)


@tool(args_schema=BashInput)
async def bash(command: str, runtime: ToolRuntime[AgentContext]) -> dict:
    """Execute Bash with a timeout and bounded output. cwd is not an OS sandbox; dependent actions need separate calls."""
    try:
        process = await asyncio.create_subprocess_exec(
            "/bin/bash", "-c", command,
            cwd=runtime.context.workdir,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise ToolException(f"bash could not start: {exc}") from exc

    remaining = runtime.context.max_tool_output_bytes
    truncated = False
    stdout = bytearray()
    stderr = bytearray()

    async def consume(stream, output):
        nonlocal remaining, truncated
        while chunk := await stream.read(8192):
            accepted = chunk[:remaining]
            output.extend(accepted)
            remaining -= len(accepted)
            truncated |= len(accepted) < len(chunk)

    readers = [asyncio.create_task(consume(process.stdout, stdout)), asyncio.create_task(consume(process.stderr, stderr))]
    completion = asyncio.gather(process.wait(), *readers)
    timed_out = False
    try:
        await asyncio.wait_for(asyncio.shield(completion), runtime.context.bash_timeout_s)
    except TimeoutError:
        timed_out = True
        await _terminate(process)
    except asyncio.CancelledError:
        await _terminate(process)
        raise
    finally:
        if not completion.done():
            try:
                await asyncio.wait_for(asyncio.shield(completion), 0.5)
            except TimeoutError:
                completion.cancel()
        await asyncio.gather(completion, return_exceptions=True)

    stdout_text, stdout_truncated = _decode_output(stdout)
    stderr_text, stderr_truncated = _decode_output(stderr)
    result = {
        "exit_code": process.returncode,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "timed_out": timed_out,
        "truncated": truncated or stdout_truncated or stderr_truncated,
    }
    if timed_out or process.returncode != 0:
        raise ToolException(json.dumps(result, ensure_ascii=False))
    return result
