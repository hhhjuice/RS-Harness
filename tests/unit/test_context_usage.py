import json
from io import StringIO

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from rich.console import Console

from rs_harness.config import ModelConfig
from rs_harness.context_display import print_context_usage
from rs_harness.context_usage import (
    CATEGORY_LABELS, CONTEXT_SOURCE, estimate_categories, model_key, resolve_context_window, summarize_usage,
)


def test_categories_are_exclusive_and_add_up():
    skill_prompt = "Skills catalog: read the selected skill before using it."
    system = SystemMessage(content=[{"type": "text", "text": "System instructions."},
                                    {"type": "text", "text": skill_prompt}])
    messages = [
        HumanMessage(content="用户问题"),
        AIMessage(content="answer", tool_calls=[{"name": "load_skill", "args": {"name": "demo"}, "id": "one"}]),
        ToolMessage(content="Skill instructions " * 30, name="load_skill", tool_call_id="one"),
        ToolMessage(content="Skill instructions " * 30, name="load_skill", tool_call_id="two"),
        ToolMessage(content="Unknown skill", name="load_skill", status="error", tool_call_id="three"),
        ToolMessage(content="Tool output", name="bash", tool_call_id="four"),
    ]
    counts = estimate_categories(system, messages, [], skills_prompt=skill_prompt)
    assert sum(counts.values()) == count_tokens_approximately([system, *messages])
    assert counts["user"] == count_tokens_approximately(messages[:1])
    assert counts["assistant"] == count_tokens_approximately(messages[1:2])
    assert counts["tool_results"] == count_tokens_approximately(messages[-2:])
    assert counts["skills"] > count_tokens_approximately(messages[2:4])
    assert counts["images"] == counts["tool_schemas"] == 0


def test_images_ignore_base64_size_and_tool_artifacts():
    def count(size):
        message = HumanMessage(content=[{"type": "text", "text": "Look"},
                                         {"type": "image", "base64": "x" * size, "mime_type": "image/png"},
                                         {"type": "image_url", "image_url": {"url": "https://invalid.example/image"}}])
        tool = ToolMessage(content="OK", artifact={"data": "x" * size}, tool_call_id="a")
        return estimate_categories(None, [message, tool], [])
    assert count(100) == count(100_000)
    assert count(100)["images"] == 170
    assert count(100)["user"] > 0


def test_block_content_counts_separate_tool_calls_once():
    call = {"name": "bash", "args": {"command": "echo " + "detail " * 100}, "id": "a", "type": "tool_call"}
    text_only = AIMessage(content=[{"type": "text", "text": "Running"}])
    with_call = text_only.model_copy(update={"tool_calls": [call]})
    counts = estimate_categories(None, [with_call], [])
    assert counts["assistant"] > estimate_categories(None, [text_only], [])["assistant"] + 100
    embedded = AIMessage(content=[call], tool_calls=[call])
    assert estimate_categories(None, [embedded], [])["assistant"] == count_tokens_approximately([embedded])


@pytest.mark.parametrize("block_type", ["function_call", "custom_tool_call"])
def test_responses_tool_calls_are_counted_once(block_type):
    args = {"__arg1": "instructions " * 1000}
    block = {"type": block_type, "id": "fc_item", "call_id": "call_one", "name": "work"}
    block.update({"arguments": json.dumps(args)} if block_type == "function_call" else {"input": args["__arg1"]})
    message = AIMessage(content=[block], tool_calls=[{"name": "work", "args": args, "id": "call_one"}])
    original = message.model_dump()

    counts = estimate_categories(None, [message], [])

    assert counts["assistant"] == count_tokens_approximately([message])
    assert sum(counts.values()) == counts["assistant"]
    assert message.model_dump() == original


@pytest.mark.parametrize("block_type", ["function_call", "custom_tool_call"])
def test_responses_blocks_do_not_hide_unembedded_calls(block_type):
    args = {"__arg1": "embedded instructions " * 100}
    block = {"type": block_type, "id": "fc_item", "call_id": "call_one", "name": "work"}
    block.update({"arguments": json.dumps(args)} if block_type == "function_call" else {"input": args["__arg1"]})
    embedded = {"name": "work", "args": args, "id": "call_one"}
    # Item IDs are not call IDs. Even this matching string must remain a separate call.
    separate = {"name": "work", "args": {"__arg1": "separate instructions " * 100}, "id": "fc_item"}
    mixed = AIMessage(content=[block], tool_calls=[embedded, separate])
    only_separate = AIMessage(content=[block], tool_calls=[separate])

    mixed_count = estimate_categories(None, [mixed], [])["assistant"]

    assert mixed_count == estimate_categories(None, [only_separate], [])["assistant"]
    assert mixed_count > count_tokens_approximately([mixed])


def test_framework_messages_keep_their_source():
    messages = [HumanMessage(content="Continue", additional_kwargs={CONTEXT_SOURCE: "system"}),
                HumanMessage(content="Image tool caption", additional_kwargs={CONTEXT_SOURCE: "tool_results"})]
    counts = estimate_categories(None, messages, [])
    assert counts["user"] == 0
    assert counts["system"] == count_tokens_approximately(messages[:1])
    assert counts["tool_results"] == count_tokens_approximately(messages[1:])


def test_calibration_and_capacity(config):
    config = config.model_copy(update={"model": config.model.model_copy(update={"context_window": 100})})
    counts = dict.fromkeys(CATEGORY_LABELS, 10)
    calibration = {"model_key": model_key(config), "estimated_input_tokens": 70, "input_tokens": 140}
    usage = summarize_usage(config, "session", counts, calibration)
    assert usage.estimation_method == "calibrated"
    assert usage.used_tokens == sum(item.tokens for item in usage.categories) == 140
    assert usage.usage_percent == 140
    assert usage.remaining_tokens == 0
    assert all(item.percent == 20 for item in usage.categories)
    other = config.model_copy(update={"model": config.model.model_copy(update={"base_url": "http://other/v1"})})
    assert summarize_usage(other, "session", counts, calibration).estimation_method == "approximate"
    assert summarize_usage(config, "session", counts, {**calibration, "input_tokens": 0}).used_tokens == 70


def test_capacity_uses_profile_or_override_not_output_limit(config):
    assert resolve_context_window(config) is None
    known = config.model_copy(update={"model": config.model.model_copy(update={"model": "gpt-4o-mini", "max_tokens": 123})})
    assert resolve_context_window(known) == 128000
    explicit = known.model_copy(update={"model": known.model.model_copy(update={"context_window": 64000})})
    assert resolve_context_window(explicit) == 64000


@pytest.mark.parametrize("value", [0, -1, True, "128000", 1.5])
def test_context_window_requires_positive_integer(value):
    with pytest.raises(ValueError):
        ModelConfig(model="test", context_window=value)


def sample_usage(config, *, window=128000, used=32000):
    config = config.model_copy(update={"model": config.model.model_copy(update={"context_window": window})})
    counts = dict.fromkeys(CATEGORY_LABELS, 0)
    counts.update(system=used // 8, user=used // 8, assistant=used // 4, tool_results=used // 2)
    return summarize_usage(config, "session", counts)


@pytest.mark.parametrize("width", [100, 76, 55, 40, 32])
def test_panel_fits_terminal_and_redirects_as_plain_text(config, width):
    stream = StringIO()
    console = Console(file=stream, width=width, force_terminal=False)
    print_context_usage(sample_usage(config), console=console)
    output = stream.getvalue()
    assert "32,000" in output and "128,000" in output and "96,000" in output and "25.0%" in output
    assert "≈" in output and "\x1b" not in output
    assert "Set model.context_window" not in output
    assert all(len(line) <= min(width, 76) for line in output.splitlines())


@pytest.mark.parametrize("used,color", [(32000, "34"), (104000, "33"), (128000, "31"), (160000, "31")])
def test_panel_usage_colors_and_overflow(config, used, color):
    stream = StringIO()
    console = Console(file=stream, width=76, force_terminal=True, color_system="standard", no_color=False)
    print_context_usage(sample_usage(config, used=used), console=console)
    assert f"\x1b[{color}m" in stream.getvalue()
    if used > 128000:
        assert "125.0%" in stream.getvalue() and "Free 0" in stream.getvalue()


def test_unknown_capacity_and_no_color(config, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    stream = StringIO()
    console = Console(file=stream, width=76, force_terminal=True, color_system="standard")
    print_context_usage(sample_usage(config, window=None), console=console)
    output = stream.getvalue()
    assert "Set model.context_window" in output and "Free —" in output
    # Rich may retain bold/dim under NO_COLOR; it must not emit color codes.
    assert not any(f"[{code}m" in output for code in ["31", "33", "34", "90", "1;34"])


def test_model_name_is_literal_terminal_text(config):
    usage = sample_usage(config).model_copy(update={"model": "[red]bad\x1b[2J\nname"})
    stream = StringIO()
    print_context_usage(usage, console=Console(file=stream, width=76, force_terminal=False))
    assert "[red]bad\\x1b[2J\\nname" in stream.getvalue()
    assert "\x1b" not in stream.getvalue()
