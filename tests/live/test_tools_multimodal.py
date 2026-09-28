import json
import subprocess
import sys
from pathlib import Path

import pytest

from rs_harness.config import Config, ConfigError
from tests.images import write_test_image

pytestmark = pytest.mark.live


@pytest.fixture
def live_config_path():
    path = Path(".agents/config.toml").resolve()
    if not path.is_file():
        pytest.skip("Configure .agents/config.toml to run real-model tests")
    try:
        Config.load(path).api_key()
    except ConfigError as exc:
        pytest.skip(exc.message)
    return path


def ask_in_process(config_path, workdir, text, *, images=(), session_id=""):
    script = """
import asyncio
import json
import sys
from pathlib import Path
from rs_harness.application import ask, create_session
from rs_harness.config import Config
from rs_harness.inputs import build_human_message

async def run():
    config = Config.load(sys.argv[1])
    workdir = Path(sys.argv[2])
    config = config.model_copy(update={'runtime': config.runtime.model_copy(update={
        'sessions_dir': workdir / 'sessions', 'checkpoint_file': workdir / 'sessions' / 'checkpoints.sqlite3',
    })})
    message = build_human_message(sys.argv[4], images=json.loads(sys.argv[5]), workdir=workdir, config=config)
    session_id = sys.argv[3] or create_session(config, workdir=str(workdir)).session_id
    result = await ask(config, session_id, message)
    print(result.model_dump_json())

asyncio.run(run())
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(config_path), str(workdir), session_id, text, json.dumps(images)],
        capture_output=True, text=True, timeout=240,
    )
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["status"] == "succeeded", result
    return result


def assert_colors(answer):
    normalized = answer.lower()
    assert "red" in normalized or "红" in normalized, answer
    assert "blue" in normalized or "蓝" in normalized, answer


def test_real_model_five_tools(live_config_path, tmp_path):
    result = ask_in_process(
        live_config_path, tmp_path,
        "Use each of the five tools to complete these steps in order. "
        "First write_file creates note.txt containing exactly 'first'. "
        "Then read_file reads it. Then edit_file replaces 'first' with 'second'. "
        "Then glob finds *.txt. Finally bash runs 'cat note.txt'. "
        "Wait for each result before requesting the next dependent step. Report the final content.",
    )
    assert (tmp_path / "note.txt").read_text() == "second"
    trajectory = tmp_path / "sessions" / result["session_id"] / "trajectory.jsonl"
    events = [json.loads(line) for line in trajectory.read_text().splitlines()]
    assert {event["payload"]["name"] for event in events if event["event"] == "tool_start"} >= {
        "write_file", "read_file", "edit_file", "glob", "bash",
    }
    assert "second" in result["answer"]


@pytest.mark.parametrize("text", ["Describe the colors and their positions in this image.", ""])
def test_real_model_local_image(live_config_path, tmp_path, text):
    image = tmp_path / "colors.png"
    write_test_image(image)
    result = ask_in_process(live_config_path, tmp_path, text, images=[str(image)])
    assert_colors(result["answer"])


def test_real_model_url_image(live_config_path, tmp_path):
    result = ask_in_process(
        live_config_path, tmp_path, "Describe the animal and its color in this image.",
        images=["https://cdn.deepseek.com/platform/favicon.png"],
    )
    normalized = result["answer"].lower()
    assert any(word in normalized for word in ["whale", "鲸"]), result["answer"]
    assert any(word in normalized for word in ["blue", "navy", "蓝"]), result["answer"]


def test_real_model_image_session_across_processes(live_config_path, tmp_path):
    image = tmp_path / "colors.png"
    write_test_image(image)
    first = ask_in_process(live_config_path, tmp_path, "Describe the colors in this image.", images=[str(image)])
    assert_colors(first["answer"])
    image.unlink()
    second = ask_in_process(
        live_config_path, tmp_path, "Repeat the two colors in the earlier image and say which is on the left.",
        session_id=first["session_id"],
    )
    assert_colors(second["answer"])


def test_real_model_read_image_and_recovery(live_config_path, tmp_path):
    image = tmp_path / "sample.png"
    write_test_image(image)
    format_instruction = (
        'Reply only with a JSON object whose keys are "left" and "right" and whose values '
        'are the English color names of the left and right halves. No Markdown.'
    )
    first = ask_in_process(
        live_config_path, tmp_path,
        "Use read_image to inspect sample.png. " + format_instruction,
    )
    assert json.loads(first["answer"]) == {"left": "red", "right": "blue"}
    image.unlink()
    second = ask_in_process(
        live_config_path, tmp_path,
        "Use the earlier image in this conversation without reopening the file. " + format_instruction,
        session_id=first["session_id"],
    )
    assert json.loads(second["answer"]) == {"left": "red", "right": "blue"}
    trajectory = (tmp_path / "sessions" / first["session_id"] / "trajectory.jsonl").read_text()
    events = [json.loads(line) for line in trajectory.splitlines()]
    starts = [event for event in events if event["event"] == "tool_start" and event["payload"]["name"] == "read_image"]
    assert len(starts) == 1
    assert "base64" not in trajectory
