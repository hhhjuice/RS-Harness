"""Typer CLI for local text and image sessions."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from typing import Annotated, Sequence

import typer
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.tools import ToolException
from prompt_toolkit import PromptSession
from prompt_toolkit.output.color_depth import ColorDepth
from prompt_toolkit.output.defaults import create_output
from prompt_toolkit.styles import Style
from rich.markup import escape

from . import __version__
from .application import ApprovalStatus, InteractionResult, ask, create_session, get_configured_mcp, get_context_usage, get_pending_approvals, resume
from .config import REASONING_EFFORTS, Config, ConfigError, save_reasoning_effort
from .context_display import print_context_usage
from .inputs import InputError, build_human_message
from .mcp import MCPRecoveryError
from .progress import CliProgress
from .sessions.catalog import Session, SessionBusy, SessionCatalog, SessionError
from .skills import SkillLoader

RSH_PROMPT = [("class:rsh-prompt", "rsh> ")]
RSH_PROMPT_STYLE = Style.from_dict({"rsh-prompt": "ansiblue"})
APPROVAL_PROMPT = [("class:approval-prompt", "Execute? [yes/no]> ")]
APPROVAL_PROMPT_STYLE = Style.from_dict({"approval-prompt": "ansiyellow"})
REASONING_USAGE = "/reasoning [none|minimal|low|medium|high|xhigh|max|default]"
INTERACTIVE_HELP = (
    "/help       show this help\n"
    "/status     show session status\n"
    "/context    show context usage\n"
    "/skill      show installed skills\n"
    "/mcp        show configured MCP servers and tools\n"
    "/reasoning  show or set reasoning effort\n"
    f"            {REASONING_USAGE}\n"
    "/close      close session\n"
    "/quit       leave session running\n"
    "/exit       leave session running"
)
CLI_USAGE_HELP = (
    "Images: rsh ask --image PATH_OR_URL [TEXT]; repeat --image for multiple images.\n"
    "JSON output: rsh COMMAND --json (ask, create-session, list-session, close-session).\n"
    "Run rsh COMMAND --help for all command options."
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="Start an interactive session, or run a command.",
    epilog=escape(CLI_USAGE_HELP),
)


def _load(path: Path) -> Config:
    return Config.load(path)


def _print_skills(config: Config) -> None:
    skills = SkillLoader(config.runtime.skills_dir)
    typer.echo(f"Skills directory: {config.runtime.skills_dir}")
    typer.echo(f"Installed skills ({len(skills.skills)}):\n{skills.catalog()}" if skills.skills else "No skills installed.")


def _terminal_text(value: str) -> str:
    """Render remote control characters literally instead of executing them in a terminal."""
    return "".join(char if char.isprintable() else char.encode("unicode_escape").decode("ascii") for char in value)


async def _print_mcp(config: Config) -> None:
    servers = await get_configured_mcp(config)
    if not servers:
        typer.echo("No MCP servers configured.")
        return
    typer.echo(f"Configured MCP servers ({len(servers)}):")
    for server in servers:
        typer.echo(f"- {server['name']} ({server['transport']}), tools: {len(server['tools'])}")
        for tool in server["tools"]:
            name = _terminal_text(json.dumps(tool["raw_name"], ensure_ascii=False)[1:-1])
            description = _terminal_text(" ".join(tool["description"].split()))
            if len(description) > 120:
                description = description[:117] + "..."
            summary = f": {description}" if description else ""
            typer.echo(f"  - {name} [{tool['permission']}]{summary}")


def _print_result(result: InteractionResult, as_json: bool, progress: CliProgress | None = None) -> None:
    payload = result.model_dump(mode="json")
    if as_json:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    elif result.status == "succeeded":
        typer.echo(result.answer or "")
    elif result.interrupts:
        _print_approvals(result)
    elif progress is not None and progress.has_reported_error(result.error):
        typer.echo("This run failed.", err=True)
    else:
        typer.echo(f"Error: {payload.get('error')}", err=True)


def _print_action(action: dict) -> None:
    name = str(action.get("name", "unknown"))
    arguments = action.get("args") or {}
    if name == "bash" and isinstance(arguments, dict):
        command = arguments.get("command", "")
        if isinstance(command, str):
            operation = json.dumps(command, ensure_ascii=False)[1:-1]
        else:
            operation = json.dumps(command, ensure_ascii=False, separators=(",", ":"))
    else:
        operation = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    typer.echo(f"Tool: {name}", err=True)
    typer.echo(f"Operation: {operation}", err=True)


def _print_approvals(pending: ApprovalStatus | InteractionResult) -> None:
    _print_approval_header(pending)
    for interrupt in pending.interrupts:
        for action in interrupt.get("action_requests", []):
            _print_action(action)


def _print_approval_header(pending: ApprovalStatus | InteractionResult) -> None:
    count = sum(len(item.get("action_requests", [])) for item in pending.interrupts)
    if count:
        items = "item" if count == 1 else "items"
        typer.echo(f"[Approval] Waiting for approval ({count} {items})", err=True)


def _exit_code(result: InteractionResult) -> int:
    return {"succeeded": 0, "failed": 1, "interrupted": 2}[result.status]


def _print_error(error: Exception, as_json: bool, *, code: str | None = None) -> None:
    if as_json:
        typer.echo(json.dumps({"status": "failed", "error": {"type": code or type(error).__name__, "message": str(error)}}, ensure_ascii=False))
    else:
        typer.echo(f"Error: {error}", err=True)


DEFAULT_CONFIG = Path(".agents/config.toml")
DEFAULT_WORKDIR = Path(__file__).resolve().parents[2] / "workdir"


def _create_session(config: Config, *, workdir: Path | None) -> Session:
    if workdir is None:
        workdir = DEFAULT_WORKDIR
        try:
            workdir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SessionError(f"Unable to create working directory {workdir}: {exc}") from exc
    return create_session(config, workdir=str(workdir))


def _root_option(ctx: typer.Context, name: str, value):
    source = ctx.get_parameter_source(name)
    if source is not None and source.name == "COMMANDLINE":
        return value
    return ctx.obj.get(name, value)


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    config: Annotated[Path, typer.Option("--config", dir_okay=False)] = DEFAULT_CONFIG,
    session_id: Annotated[str | None, typer.Option("--session-id")] = None,
    workdir: Annotated[Path | None, typer.Option("--workdir", file_okay=False)] = None,
) -> None:
    ctx.obj = {"config": config, "session_id": session_id, "workdir": workdir}
    if ctx.invoked_subcommand is not None:
        return
    asyncio.run(_interactive(config, session_id=session_id, workdir=workdir))


@app.command("ask")
def ask_command(
    ctx: typer.Context,
    text: Annotated[str, typer.Argument()] = "",
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = DEFAULT_CONFIG,
    session_id: Annotated[str | None, typer.Option("--session-id")] = None,
    workdir: Annotated[Path | None, typer.Option("--workdir", file_okay=False)] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
    images: Annotated[list[str] | None, typer.Option("--image", help="Local image path or HTTP(S) URL; repeat for multiple images.")] = None,
) -> None:
    """Send text and/or images (--image); use --json for JSON output."""
    if not text.strip() and not images:
        raise typer.Exit(code=0)
    try:
        config = _root_option(ctx, "config", config)
        session_id = _root_option(ctx, "session_id", session_id)
        workdir = _root_option(ctx, "workdir", workdir)
        loaded = _load(config)
        session = SessionCatalog.open(loaded, session_id) if session_id else None
        input_workdir = session.meta.workdir if session else (workdir or DEFAULT_WORKDIR)
        if session and workdir and workdir.expanduser().resolve() != Path(session.meta.workdir):
            raise InputError("session_workdir_mismatch: an existing session keeps its original workdir")
        message = build_human_message(text, images=images or (), workdir=input_workdir, config=loaded)
        if session_id is None:
            session = _create_session(loaded, workdir=workdir)
            session_id = session.session_id
        progress = None if as_json else CliProgress()
        options = {} if progress is None else {"callbacks": [progress]}
        result = asyncio.run(_ask_with_inline_approvals(loaded, session_id, message, **options))
        _print_result(result, as_json, progress)
        if result.status != "succeeded":
            raise typer.Exit(code=_exit_code(result))
    except (ConfigError, SessionError, SessionBusy, InputError) as exc:
        if as_json:
            typer.echo(json.dumps({"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)}}, ensure_ascii=False))
        else:
            typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("create-session")
def create_session_command(
    ctx: typer.Context,
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = DEFAULT_CONFIG,
    workdir: Annotated[Path | None, typer.Option("--workdir", file_okay=False)] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Create a session; use --json for session details."""
    try:
        config = _root_option(ctx, "config", config)
        workdir = _root_option(ctx, "workdir", workdir)
        session = _create_session(_load(config), workdir=workdir)
    except (ConfigError, SessionError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    payload = {"session_id": session.session_id, "session_path": str(session.path)}
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2) if as_json else session.session_id)


@app.command("list-session")
def list_session_command(
    ctx: typer.Context,
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = DEFAULT_CONFIG,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List sessions; use --json for session details."""
    try:
        config = _root_option(ctx, "config", config)
        sessions = SessionCatalog.list(_load(config))
    except ConfigError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    payload = [session.meta.model_dump(mode="json") for session in sessions]
    if as_json:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        typer.echo("\n".join(item["session_id"] for item in payload))


@app.command("close-session")
def close_session_command(
    ctx: typer.Context,
    session_id: Annotated[str | None, typer.Option("--session-id")] = None,
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = DEFAULT_CONFIG,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Close the selected session; use --json for session details."""
    try:
        session_id = _root_option(ctx, "session_id", session_id)
        if session_id is None:
            raise typer.BadParameter("Missing option '--session-id'.")
        config = _root_option(ctx, "config", config)
        session = SessionCatalog.close(_load(config), session_id)
    except (ConfigError, SessionError, SessionBusy) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    payload = session.meta.model_dump(mode="json")
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2) if as_json else session.session_id)


@app.command("list-skill")
def list_skill_command(
    ctx: typer.Context,
    config: Annotated[Path, typer.Option("--config", dir_okay=False)] = DEFAULT_CONFIG,
) -> None:
    """List installed skills from the configured skills directory."""
    try:
        _print_skills(_load(_root_option(ctx, "config", config)))
    except ConfigError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("list-mcp")
def list_mcp_command(
    ctx: typer.Context,
    config: Annotated[Path, typer.Option("--config", dir_okay=False)] = DEFAULT_CONFIG,
) -> None:
    """Show MCP servers and tools from the current configuration."""
    try:
        config = _root_option(ctx, "config", config)
        asyncio.run(_print_mcp(_load(config)))
    except (ConfigError, ToolException) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


async def _interactive(config_path: Path, *, session_id: str | None, workdir: Path | None) -> None:
    try:
        config_path = config_path.expanduser().resolve()
        config = _load(config_path)
        session = SessionCatalog.open(config, session_id) if session_id else None
    except (ConfigError, SessionError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"rsh {__version__}; enter /help for commands.")
    if session is not None:
        typer.echo(f"session {session.session_id}")

    prompt = PromptSession(
        RSH_PROMPT,
        style=RSH_PROMPT_STYLE,
        color_depth=ColorDepth.DEPTH_4_BIT,
    ) if sys.stdin.isatty() and sys.stdout.isatty() else None
    pending = await get_pending_approvals(config, session.session_id) if session else None
    # progress 实例的生命周期 = 本进程内一次用户输入的交互链（ask + 其引发的各轮审批 resume），
    # 审批恢复复用错误去重状态；各次模型输出在结束或中断时收尾。
    progress: CliProgress | None = None
    while True:
        if pending is not None and pending.interrupts:
            if prompt is None or pending.session_status == "closed":
                _print_approvals(pending)
                if prompt is None:
                    raise typer.Exit(code=2)
                break
            decisions = await _collect_approvals(pending)
            if decisions is None:
                break
            if progress is None:
                progress = CliProgress()
            result = await resume(config, session.session_id, decisions, callbacks=[progress])
            if result.status != "interrupted":
                _print_result(result, False, progress)
                progress = None
            pending = await get_pending_approvals(config, session.session_id)
            continue
        try:
            if prompt is not None:
                text = await prompt.prompt_async()
            else:
                text = typer.prompt("rsh", prompt_suffix="> ", default="", show_default=False)
        except (EOFError, KeyboardInterrupt, typer.Abort):
            typer.echo()
            break
        command = text.strip()
        if not command:
            continue
        if command in {"/quit", "/exit"}:
            break
        if command == "/help":
            typer.echo(INTERACTIVE_HELP)
            continue
        if command.split(maxsplit=1)[0] == "/context":
            if command != "/context":
                typer.echo("Usage: /context", err=True)
            elif session is None:
                typer.echo("No session")
            else:
                try:
                    print_context_usage(await get_context_usage(config, session.session_id))
                except (ConfigError, SessionError, MCPRecoveryError, OSError, sqlite3.Error) as exc:
                    typer.echo(f"Error: {exc}", err=True)
            continue
        if command.split(maxsplit=1)[0] == "/skill":
            if command != "/skill":
                typer.echo("Usage: /skill", err=True)
                continue
            try:
                _print_skills(config)
            except ConfigError as exc:
                typer.echo(f"Error: {exc}", err=True)
            continue
        if command.split(maxsplit=1)[0] == "/mcp":
            if command != "/mcp":
                typer.echo("Usage: /mcp", err=True)
                continue
            try:
                await _print_mcp(config)
            except (ConfigError, ToolException) as exc:
                typer.echo(f"Error: {exc}", err=True)
            continue
        if command.split(maxsplit=1)[0] == "/reasoning":
            parts = command.split()
            if len(parts) == 1:
                typer.echo(f"Reasoning effort: {config.model.reasoning_effort or 'default'}\n{REASONING_USAGE}")
                continue
            if len(parts) != 2 or parts[1] not in (*REASONING_EFFORTS, "default"):
                typer.echo(f"Usage: {REASONING_USAGE}", err=True)
                continue
            effort = None if parts[1] == "default" else parts[1]
            try:
                save_reasoning_effort(config_path, effort)
            except ConfigError as exc:
                typer.echo(f"Error: {exc}", err=True)
                continue
            config = config.model_copy(update={"model": config.model.model_copy(update={"reasoning_effort": effort})})
            typer.echo(f"Reasoning effort: {effort or 'default'}")
            continue
        if command == "/status":
            if session is None:
                typer.echo("No session")
            else:
                session.reload()
                pending = await get_pending_approvals(config, session.session_id)
                typer.echo(json.dumps({**session.meta.model_dump(mode="json"), **pending.model_dump(mode="json")}, ensure_ascii=False, indent=2))
            continue
        if command == "/close":
            if session is None:
                typer.echo("No session")
            else:
                try:
                    session = SessionCatalog.close(config, session.session_id)
                    typer.echo(f"Closed session {session.session_id}")
                except (SessionError, SessionBusy) as exc:
                    typer.echo(f"Error: {exc}", err=True)
            break

        try:
            if session is None:
                session = _create_session(config, workdir=workdir)
            progress = CliProgress()
            result = await ask(config, session.session_id, text, callbacks=[progress])
            if result.interrupts:
                pending = _approval_status_from_result(result)
            else:
                _print_result(result, False, progress)
                progress = None
        except (ConfigError, SessionError, SessionBusy) as exc:
            progress = None
            typer.echo(f"Error: {exc}", err=True)

    if session is not None and session.meta.status == "running":
        typer.echo(f"Use rsh --session-id {session.session_id} to resume this session.", err=True)


def _approval_status_from_result(result: InteractionResult) -> ApprovalStatus:
    return ApprovalStatus(**{key: getattr(result, key) for key in ApprovalStatus.model_fields})


def _can_collect_approvals() -> bool:
    return sys.stdin.isatty() and sys.stderr.isatty()


async def _ask_with_inline_approvals(
    config: Config,
    session_id: str,
    message: HumanMessage,
    *,
    callbacks: Sequence[BaseCallbackHandler] = (),
) -> InteractionResult:
    result = await ask(config, session_id, message, callbacks=callbacks)
    while result.interrupts and _can_collect_approvals():
        decisions = await _collect_approvals(_approval_status_from_result(result))
        if decisions is None:
            break
        result = await resume(config, session_id, decisions, callbacks=callbacks)
    return result


async def _collect_approvals(pending: ApprovalStatus) -> dict | None:
    _print_approval_header(pending)
    prompt = PromptSession(
        output=create_output(stdout=sys.stderr),
        style=APPROVAL_PROMPT_STYLE,
        color_depth=ColorDepth.DEPTH_4_BIT,
    )
    decisions_by_interrupt = {}
    try:
        for item in pending.interrupts:
            decisions = []
            for action in item["action_requests"]:
                _print_action(action)
                while True:
                    choice = (await prompt.prompt_async(APPROVAL_PROMPT)).strip()
                    choice = choice.casefold()
                    if choice in {"y", "yes"}:
                        decisions.append({"type": "approve"})
                        break
                    if choice in {"n", "no"}:
                        decisions.append({"type": "reject"})
                        break
                    typer.echo("Enter y/yes or n/no.", err=True)
            decisions_by_interrupt[item["id"]] = {"decisions": decisions}
    except (EOFError, KeyboardInterrupt):
        return None
    return decisions_by_interrupt


def main() -> None:
    app()
