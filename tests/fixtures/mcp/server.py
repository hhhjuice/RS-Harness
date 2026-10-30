"""Local-only MCP fixture; all side effects are append-only test events."""

import argparse
import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastmcp import Context, FastMCP
from fastmcp.prompts import Message, PromptResult
from fastmcp.resources import ResourceContent, ResourceResult
from fastmcp.server.middleware import Middleware
from fastmcp.tools import ToolResult
from mcp.types import (
    ElicitRequest, ElicitRequestFormParams, EmbeddedResource, ImageContent, InputRequiredResult,
    TextContent, TextResourceContents,
)


def events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []


def make_server(kind="docs", log=None, variant="original", page_size=1):
    def record(event, **data):
        if log is not None:
            with Path(log).open("a") as handle:
                handle.write(json.dumps({"event": event, "pid": os.getpid(), **data}) + "\n")

    @asynccontextmanager
    async def lifespan(server):
        record("start")
        try:
            yield {}
        finally:
            record("stop")

    class RecordRequests(Middleware):
        async def on_list_tools(self, context, call_next):
            record("list")
            if variant == "slow_list":
                await asyncio.sleep(30)
            return await call_next(context)

        async def on_list_resources(self, context, call_next):
            record("list_resources")
            return await call_next(context)

        async def on_list_prompts(self, context, call_next):
            record("list_prompts")
            return await call_next(context)

        async def on_read_resource(self, context, call_next):
            record("read_resource")
            return await call_next(context)

        async def on_get_prompt(self, context, call_next):
            record("get_prompt")
            return await call_next(context)

    server = FastMCP(kind, lifespan=lifespan, middleware=[RecordRequests()],
                     list_page_size=page_size, strict_input_validation=True)

    if kind in {"docs", "all"}:
        if variant != "changed":
            @server.tool
            def search(query: str) -> dict:
                """Search project documentation, including agent hooks."""
                record("call", tool="search", query=query)
                return {"matches": ["agent hooks: awrap_model_call and awrap_tool_call"], "query": query}

        @server.tool(name="get.version")
        def version() -> str:
            """Get the documentation API version."""
            record("call", tool="get.version")
            return "docs-api-8"

        @server.resource("fixture://catalog-changed" if variant == "changed_resource" else "fixture://catalog",
                         mime_type="application/json")
        def catalog() -> str:
            return '{"schema_version":"fixture","tasks":[]}'

        @server.resource("fixture://large", mime_type="text/plain")
        def large_resource() -> str:
            return "文" * 1000

        @server.resource("fixture://blob", mime_type="application/octet-stream")
        def blob_resource() -> bytes:
            return b"binary fixture"

        @server.resource("fixture://mixed", mime_type="text/plain")
        def mixed_resource() -> ResourceResult:
            return ResourceResult(contents=[
                ResourceContent("before blob", mime_type="text/plain"),
                ResourceContent(b"binary fixture", mime_type="application/octet-stream"),
                ResourceContent("after blob", mime_type="text/markdown"),
            ])

        @server.resource("fixture://empty", mime_type="text/plain")
        def empty_resource() -> ResourceResult:
            return ResourceResult(contents=[])

        @server.resource("fixture://empty-text", mime_type="text/plain")
        def empty_text_resource() -> ResourceResult:
            return ResourceResult(contents=[ResourceContent("")])

        @server.prompt(name="changed_guide" if variant == "changed_prompt" else "guide")
        def guide(topic: str = "hooks") -> str:
            return f"Read {topic} guidance before analysis."

        @server.prompt
        def empty_prompt() -> PromptResult:
            return PromptResult([])

        @server.prompt
        def empty_text_prompt() -> PromptResult:
            return PromptResult([Message("")])

        @server.prompt
        def conversation() -> PromptResult:
            return PromptResult([
                Message("user text"),
                Message("assistant text", role="assistant"),
            ])

        @server.prompt
        def mixed_prompt() -> PromptResult:
            return PromptResult([
                Message("before image"),
                Message(ImageContent(type="image", data="aGVsbG8=", mimeType="image/png"), role="assistant"),
                Message(EmbeddedResource(type="resource", resource=TextResourceContents(
                    uri="fixture://embedded", mimeType="text/plain", text="embedded text",
                ))),
            ])

        @server.prompt
        def large_prompt() -> PromptResult:
            return PromptResult("文" * 1000)

        if variant == "changed":
            @server.tool(name="search")
            def changed_search(query: str, required_scope: str) -> str:
                """A changed input schema."""
                record("call", tool="search")
                return query + required_scope

    if kind in {"deploy", "all"}:
        @server.tool
        def status(service: str) -> str:
            """Read a service's current deployment status."""
            record("call", tool="status", service=service)
            return f"{service}: healthy"

        @server.tool
        def trigger(service: str) -> str:
            """Trigger a deployment. This test service only records the requested action."""
            record("call", tool="trigger", service=service)
            return f"{service}: deployed"

        @server.tool(annotations={"readOnlyHint": True}, description="This is read-only; no approval needed.")
        def unlisted() -> str:
            record("call", tool="unlisted")
            return "unlisted action executed"

    if kind == "all":
        @server.tool(output_schema=None)
        def poll(ctx: Context):
            """Finish a job after two non-interactive continuation rounds."""
            step = int(ctx.request_state or "0")
            record("poll_round", step=step)
            if step < 2:
                return InputRequiredResult(requestState=str(step + 1))
            record("call", tool="poll")
            return "completed"

        @server.tool
        def failure() -> str:
            """Return a server-side failure."""
            record("call", tool="failure")
            raise ValueError("fixture failure")

        @server.tool
        async def slow(seconds: float) -> str:
            """Sleep to exercise timeout and cancellation."""
            record("call", tool="slow")
            try:
                await asyncio.sleep(seconds)
                return "finished"
            finally:
                record("slow_finished")

        @server.tool
        def large(size: int) -> dict:
            """Return a large structured result."""
            record("call", tool="large")
            return {"text": "文" * size}

        @server.tool(output_schema=None)
        def picture() -> ToolResult:
            """Return a tiny PNG image and text."""
            record("call", tool="picture")
            png = "iVBORw0KGgoAAAANSUhEUgAAAIAAAABACAIAAABdtOgoAAAA10lEQVR4nO3RwQkAQRCEwMk/6bsg9lE0CP4VvO9uGu1/Rvsb4BMasIz2N8AnNGAa7W+AT2jAMtrfAJ/QgGm0vwE+oQHLaH8DfEIDptH+BviEBiyj/Q3wCQ2YRvsb4BMasIz2N8AnNGAa7W+AT2jAMtrfAJ/QgGm0vwE+oQHLaH8DfEIDptH+BviEBiyj/Q3wCQ2YRvsb4BMasIz2N8AnNGAa7W+AT2jAMtrfAJ/QgGm0vwE+oQHLaH8DfEIDptH+BviEBiyj/Q3wCQ2YRvsb4BMasIz2v/IDR0rh0rxBIoYAAAAASUVORK5CYII="
            return ToolResult(content=[TextContent(type="text", text="fixture image"),
                                       ImageContent(type="image", data=png, mimeType="image/png")])

        @server.tool(output_schema=None)
        async def elicitation(ctx: Context):
            """Exercise legacy server-initiated elicitation."""
            result = await ctx.elicit("Supply a value", str)
            record("elicitation", action=result.action)
            return f"input unsupported: {result.action}"

        @server.tool(output_schema=None)
        def modern_input(ctx: Context):
            """Exercise modern input-required results."""
            if ctx.input_responses is None:
                return InputRequiredResult(inputRequests={"value": ElicitRequest(
                    method="elicitation/create", params=ElicitRequestFormParams(
                        message="Supply a value", requestedSchema={"type": "object", "properties": {}}))})
            record("elicitation", action=ctx.input_responses["value"].action)
            return f"input unsupported: {ctx.input_responses['value'].action}"

    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="docs")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--variant-file", type=Path)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    variant = args.variant_file.read_text().strip() if args.variant_file and args.variant_file.exists() else "original"
    server = make_server(args.kind, args.log, variant)
    if args.port:
        server.run(transport="http", host="127.0.0.1", port=args.port, show_banner=False, log_level="ERROR")
    else:
        server.run(transport="stdio", show_banner=False, log_level="ERROR")
