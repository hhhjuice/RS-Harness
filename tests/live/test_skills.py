import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.live.test_tools_multimodal import live_config_path

pytestmark = pytest.mark.live


@pytest.fixture
def skill_library(tmp_path):
    source = Path(__file__).resolve().parents[2] / ".agents/skills/find-skills/SKILL.md"
    target = tmp_path / "skills/find-skills/SKILL.md"
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    return target.parent.parent


def run_skill_scenario(config_path, tmp_path, skills, prompt):
    # Each real-model scenario gets a process, workspace, checkpoint and session.
    script = '''
import asyncio, json, sys
from pathlib import Path
from langchain.agents.middleware import AgentMiddleware
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import ToolMessage
from rs_harness.agents import main, subagent
from rs_harness.application import ask, create_session
from rs_harness.config import Config
from rs_harness.sessions.checkpoints import open_checkpointer

class SkillOnlyProbe(AsyncCallbackHandler):
    def __init__(self):
        self.main_inputs = []
        self.calls = []

    async def on_chat_model_start(self, serialized, messages, *, metadata=None, **kwargs):
        if not (metadata or {}).get("subagent_id"):
            self.main_inputs.append("## What is the Skills CLI?" in str(messages))

    async def on_tool_start(self, serialized, input_str, *, inputs=None, **kwargs):
        name = serialized.get("name")
        self.calls.append(name)

class SkillOnlyGuard(AgentMiddleware):
    async def awrap_tool_call(self, request, handler):
        if request.tool_call["name"] not in {"load_skill", "delegate_task"}:
            raise AssertionError("This live test permits skill loading and delegation only: " + request.tool_call["name"])
        return await handler(request)

production_middleware = main.build_middleware
def guarded_middleware(*args, **kwargs):
    return [*production_middleware(*args, **kwargs), SkillOnlyGuard()]
main.build_middleware = guarded_middleware
subagent.build_middleware = guarded_middleware

async def run():
    config = Config.load(sys.argv[1])
    workspace = Path(sys.argv[2]) / "workspace"
    workspace.mkdir()
    skills = Path(sys.argv[3])
    raw = (skills / "find-skills/SKILL.md").read_text()
    runtime = config.runtime.model_copy(update={
        "skills_dir": skills, "sessions_dir": workspace / "sessions",
        "checkpoint_file": workspace / "sessions/checkpoints.sqlite3",
    })
    child = config.subagents[0].model_copy(update={
        "id": "skill_reader", "name": "Skill Reader",
        "description": "Load requested skills and summarize their procedures.",
        "tools": ("load_skill",), "persistence": "per_thread",
    })
    config = config.model_copy(update={"runtime": runtime, "subagents": (child,)})
    session = create_session(config, workdir=str(workspace))
    probe = SkillOnlyProbe()
    result = await ask(config, session.session_id, sys.argv[4], callbacks=[probe])
    run_config = {"configurable": {"thread_id": session.session_id}}
    async with open_checkpointer(config) as saver:
        root = await saver.aget_tuple(run_config)
        checkpoints = [item async for item in saver.alist(run_config)]
    messages = root.checkpoint["channel_values"].get("messages", []) if root else []
    tools = [message for message in messages if isinstance(message, ToolMessage)]
    child_loads = {
        message.tool_call_id for item in checkpoints if item.config["configurable"]["checkpoint_ns"]
        for message in item.checkpoint["channel_values"].get("messages", [])
        if isinstance(message, ToolMessage) and message.name == "load_skill"
        and message.status == "success" and raw in message.content
    }
    print(json.dumps({
        "result": result.model_dump(mode="json"), "model": config.model.model,
        "main_inputs": probe.main_inputs, "calls": probe.calls,
        "tools": [{"name": item.name, "status": item.status, "full_skill": raw in item.content} for item in tools],
        "child_load_count": len(child_loads),
    }, ensure_ascii=False))

asyncio.run(run())
'''
    process = subprocess.run([sys.executable, "-c", script, str(config_path), str(tmp_path), str(skills), prompt],
                             capture_output=True, text=True, timeout=240)
    assert process.returncode == 0, process.stderr
    data = json.loads(process.stdout)
    assert data["result"]["status"] == "succeeded", data
    assert data["main_inputs"] and not data["main_inputs"][0], data
    return data


def assert_main_loaded(data):
    assert data["tools"] == [{"name": "load_skill", "status": "success", "full_skill": True}], data
    assert data["calls"] == ["load_skill"], data
    assert any(data["main_inputs"][1:]), data


def test_real_model_explicit_load_and_summary(live_config_path, tmp_path, skill_library):
    data = run_skill_scenario(
        live_config_path, tmp_path, skill_library,
        "请先调用 load_skill 加载 find-skills，然后根据说明概述查找和评估技能的流程，"
        "包括入口、搜索命令和质量依据。只使用 load_skill，不委派、不访问网络、不安装、不执行命令。",
    )
    assert_main_loaded(data)
    answer = data["result"]["answer"]
    assert "skills.sh" in answer and "npx skills find" in answer, answer


def test_real_model_selects_skill_by_task_relevance(live_config_path, tmp_path, skill_library):
    data = run_skill_scenario(
        live_config_path, tmp_path, skill_library,
        "我想找一个用于代码审查的 Agent 技能。先说明你会怎样查找和评估，包括搜索入口、"
        "命令示例及质量依据。工具仅可按需使用 load_skill 加载技能说明；不要扫描文件、联网、安装、执行命令或委派。",
    )
    assert_main_loaded(data)
    assert "npx skills find" in data["result"]["answer"], data


def test_real_model_unrelated_question_needs_no_skill(live_config_path, tmp_path, skill_library):
    data = run_skill_scenario(live_config_path, tmp_path, skill_library, "计算 17 + 25，只回答数字。")
    assert data["result"]["answer"].strip() == "42", data
    assert data["calls"] == [] and data["tools"] == [], data
    assert not any(data["main_inputs"]), data


def test_real_model_delegated_loading_stays_in_child_history(live_config_path, tmp_path, skill_library):
    data = run_skill_scenario(
        live_config_path, tmp_path, skill_library,
        "请只调用一次 delegate_task，agent_id 为 skill_reader，让它调用 load_skill 加载 find-skills，"
        "概述评估技能质量的三项依据，不复制完整说明。把禁止联网、安装和执行命令的要求传给它。"
        "主代理自己不加载技能，只汇总子代理结果。",
    )
    assert data["tools"] == [{"name": "delegate_task", "status": "success", "full_skill": False}], data
    assert data["child_load_count"] == 1, data
    assert data["calls"] == ["delegate_task", "load_skill"], data
    assert not any(data["main_inputs"]), data
