import json

import pytest

from rs_harness.config import Config
from tests.fixtures.mcp.server import events
from tests.fixtures.mcp.support import http_server, run_process, stdio_config
from tests.live.test_tools_multimodal import live_config_path

pytestmark = pytest.mark.live


def live_request(path, tmp_path, mcp, **request):
    return run_process(path, live=True, workdir=str(tmp_path), mcp=mcp,
                       runtime={"sessions_dir": str(tmp_path / "sessions"),
                                "checkpoint_file": str(tmp_path / "sessions/checkpoints.sqlite3"),
                                "skills_dir": str(tmp_path / "skills"), "max_model_calls": 12}, **request)


def calls(log):
    return [item["tool"] for item in events(log) if item["event"] == "call"]


def test_docs_search_and_version(live_config_path, tmp_path):
    with http_server(tmp_path) as (url, log):
        mcp = {"servers": {"docs": {"transport": "http", "url": url}}}
        data = live_request(live_config_path, tmp_path, mcp,
                            text="连接 docs，搜索 agent hooks，并告诉我文档 API 版本。请通过 MCP 工具获取这两项信息，只使用 connect_mcp 和发现的 docs 工具。")
        assert data["result"]["status"] == "succeeded", data["result"]
        assert sorted(calls(log)) == ["get.version", "search"]
        assert "docs-api-8" in data["result"]["answer"]
        first = next(item for item in data["tool_sets"] if not item["child"])["names"]
        assert "connect_mcp" in first and not any(name.startswith("mcp__") for name in first)
        assert any("mcp__docs__search" in item["names"] for item in data["tool_sets"])


def test_resource_and_prompt_discovery_driven_calls(live_config_path, tmp_path):
    log = tmp_path / "docs.jsonl"
    mcp = stdio_config(log).model_dump(mode="json")
    data = live_request(live_config_path, tmp_path, mcp, text=(
        "连接 docs，从连接后的实际目录中选择 JSON 任务目录资源和带 topic 参数的阅读指导提示词，各调用一次。"
        "topic 使用 resources。不要调用其他 MCP 工具。最后说明读取到的 schema_version 和提示词内容。"
    ))
    assert data["result"]["status"] == "succeeded", data["result"]
    tools = {item["name"] for item in data["messages"] if item["type"] == "tool" and item["status"] == "success"}
    assert {"connect_mcp", "read_mcp_resource", "get_mcp_prompt"} <= tools
    assert sum(item["event"] == "read_resource" for item in events(log)) >= 1
    assert sum(item["event"] == "get_prompt" for item in events(log)) >= 1
    assert calls(log) == []
    assert "fixture" in data["result"]["answer"] and "resources guidance" in data["result"]["answer"]


def test_deploy_status_without_trigger(live_config_path, tmp_path):
    log = tmp_path / "deploy.jsonl"
    mcp = stdio_config(log, kind="deploy", tools={"trigger": "confirm"}).model_dump(mode="json")
    data = live_request(live_config_path, tmp_path, mcp,
                        text="连接 deploy，只查看 web 服务状态，不要触发部署，也不要调用其他工具。")
    assert data["result"]["status"] == "succeeded", data["result"]
    assert calls(log) == ["status"]


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_deploy_approval_in_new_process(live_config_path, tmp_path, decision):
    log = tmp_path / "deploy.jsonl"
    mcp = stdio_config(log, kind="deploy", tools={"trigger": "confirm"}).model_dump(mode="json")
    first = live_request(live_config_path, tmp_path, mcp,
                         text="连接 deploy 并调用 trigger 为 web 触发一次测试部署。不查状态。如果人工拒绝，请结束任务，不要再次尝试。")
    pending = first["result"]
    assert pending["status"] == "interrupted", pending
    assert not calls(log)
    assert [item["name"] for item in pending["interrupts"][0]["action_requests"]] == ["mcp__deploy__trigger"]
    decisions = {pending["interrupts"][0]["id"]: {"decisions": [{"type": decision}]}}
    result = live_request(live_config_path, tmp_path, mcp, action="resume", session_id=pending["session_id"], decisions=decisions)
    assert result["result"]["status"] == "succeeded", result["result"]
    assert calls(log) == (["trigger"] if decision == "approve" else [])
    assert result["result"]["interaction_id"] == pending["interaction_id"]


@pytest.mark.parametrize("confirm", [False, True])
def test_default_allow_and_explicit_confirm(live_config_path, tmp_path, confirm):
    log = tmp_path / "deploy.jsonl"
    mcp = stdio_config(log, kind="deploy", tools={"unlisted": "confirm"} if confirm else {}).model_dump(mode="json")
    data = live_request(live_config_path, tmp_path, mcp,
                        text="连接 deploy，然后调用 unlisted 工具一次。只使用 connect_mcp 和这个工具。")
    assert data["result"]["status"] == ("interrupted" if confirm else "succeeded"), data["result"]
    if confirm:
        assert [item["name"] for item in data["result"]["interrupts"][0]["action_requests"]] == ["mcp__deploy__unlisted"]
    else:
        assert data["result"]["interrupts"] == []
    assert calls(log) == ([] if confirm else ["unlisted"])


def test_skill_base_mcp_and_delegation_compose(live_config_path, tmp_path):
    log = tmp_path / "docs.jsonl"
    mcp = stdio_config(log).model_dump(mode="json")
    skill = tmp_path / "skills/fixture-skill/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: fixture-skill\ndescription: Read a fixture note and summarize it.\n---\nUse read_file on notes.txt and summarize its text. Do not run shell commands.\n")
    (tmp_path / "notes.txt").write_text("MCP fixture composition evidence: eight.")
    live_config = Config.load(live_config_path)
    if not live_config.subagents:
        pytest.skip("live config has no subagents")
    child_id = live_config.subagents[0].id
    data = live_request(live_config_path, tmp_path, mcp, text=(
        "请按顺序完成：主代理调用 load_skill 加载 fixture-skill，调用 read_file 读取 notes.txt；"
        "连接 docs 后搜索 agent hooks；然后委派一次给 " + child_id + "，让它只读取 notes.txt 并总结内容，不连接或使用 MCP。"
        "最后汇总两项信息。不要执行命令或修改文件。"
    ))
    assert data["result"]["status"] == "succeeded", data["result"]
    tools = {item["name"] for item in data["messages"] if item["type"] == "tool" and item["status"] == "success"}
    assert {"load_skill", "read_file", "connect_mcp", "mcp__docs__search", "delegate_task"} <= tools
    child_sets = [item["names"] for item in data["tool_sets"] if item["child"]]
    forbidden = {"connect_mcp", "read_mcp_resource", "get_mcp_prompt"}
    assert child_sets and all(not any(name in forbidden or name.startswith("mcp__") for name in names) for names in child_sets)
    assert calls(log) == ["search"]


def test_mcp_image_reaches_model(live_config_path, tmp_path):
    log = tmp_path / "all.jsonl"
    mcp = stdio_config(log, kind="all").model_dump(mode="json")
    data = live_request(live_config_path, tmp_path, mcp, text=(
        "连接 all，只调用 picture 获取图片。根据实际图片判断左半边和右半边的颜色。"
        '最后仅返回 JSON：{"left": "英文颜色", "right": "英文颜色"}，不用 Markdown。'
    ))
    assert data["result"]["status"] == "succeeded", data["result"]
    assert json.loads(data["result"]["answer"]) == {"left": "red", "right": "blue"}
    assert calls(log) == ["picture"]
