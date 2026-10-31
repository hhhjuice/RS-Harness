"""Process helpers shared by deterministic and explicitly enabled live tests."""

import json
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from rs_harness.config import MCPConfig

SERVER = Path(__file__).with_name("server.py")


def stdio_config(log, *, kind="docs", variant_file=None, tools=None, resources=None, prompts=None, timeout=10):
    args = [str(SERVER), "--kind", kind, "--log", str(log)]
    if variant_file:
        args.extend(["--variant-file", str(variant_file)])
    return MCPConfig.model_validate({"timeout_s": timeout, "servers": {kind: {
        "transport": "stdio", "command": sys.executable,
        "args": args, "cwd": str(SERVER.parent), "tools": tools or {},
        "resources": resources or {}, "prompts": prompts or {},
    }}})


@contextmanager
def http_server(tmp_path, *, kind="docs"):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log = tmp_path / "http-events.jsonl"
    with (tmp_path / "http-stderr.log").open("w+") as stderr:
        process = subprocess.Popen([sys.executable, str(SERVER), "--kind", kind, "--log", str(log), "--port", str(port)],
                                   stdout=subprocess.DEVNULL, stderr=stderr)
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    stderr.seek(0)
                    raise AssertionError(stderr.read())
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                        break
                except OSError:
                    time.sleep(0.05)
            else:
                raise AssertionError("Fixture HTTP server did not start")
            yield f"http://127.0.0.1:{port}/mcp", log
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def run_process(config_path, **request):
    result = subprocess.run([sys.executable, "-m", "tests.fixtures.mcp.runner", str(config_path)], input=json.dumps(request),
                            cwd=SERVER.parents[3], capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)
