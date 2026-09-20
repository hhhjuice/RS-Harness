"""Typer CLI for local text and image sessions."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.output.color_depth import ColorDepth
from prompt_toolkit.output.defaults import create_output
from prompt_toolkit.styles import Style

from . import __version__
from .application import ApprovalStatus, InteractionResult, ask, create_session, get_pending_approvals, resume
from .config import Config, ConfigError
from .inputs import InputError, build_human_message
from .sessions.catalog import SessionBusy, SessionCatalog, SessionError

app = typer.Typer(add_completion=False, no_args_is_help=False)

RSH_PROMPT = [("class:rsh-prompt", "rsh> ")]
RSH_PROMPT_STYLE = Style.from_dict({"rsh-prompt": "ansiblue"})
APPROVAL_PROMPT = [("class:approval-prompt", "是否执行？[yes/no]> ")]
APPROVAL_PROMPT_STYLE = Style.from_dict({"approval-prompt": "ansiyellow"})


def _load(path: Path) -> Config:
    return Config.load(path)


def _print_result(result: InteractionResult, as_json: bool) -> None:
    payload = result.model_dump(mode="json")
    if as_json:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    elif result.status == "succeeded":
        typer.echo(result.answer or "")
    elif result.interrupts:
        _print_approvals(result)
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
    typer.echo(f"工具: {name}", err=True)
    typer.echo(f"操作: {operation}", err=True)


def _print_approvals(pending: ApprovalStatus | InteractionResult) -> None:
    for interrupt in pending.interrupts:
        for action in interrupt.get("action_requests", []):
            _print_action(action)


def _exit_code(result: InteractionResult) -> int:
    return {"succeeded": 0, "failed": 1, "interrupted": 2}[result.status]


def _print_error(error: Exception, as_json: bool, *, code: str | None = None) -> None:
    if as_json:
        typer.echo(json.dumps({"status": "failed", "error": {"type": code or type(error).__name__, "message": str(error)}}, ensure_ascii=False))
    else:
        typer.echo(f"Error: {error}", err=True)


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    config: Annotated[Path, typer.Option("--config", dir_okay=False)] = Path("config/default.toml"),
    session_id: Annotated[str | None, typer.Option("--session-id")] = None,
    workdir: Annotated[Path | None, typer.Option("--workdir", file_okay=False)] = None,
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    asyncio.run(_interactive(config, session_id=session_id, workdir=workdir))


@app.command("ask")
def ask_command(
    text: Annotated[str, typer.Argument()] = "",
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path("config/default.toml"),
    session_id: Annotated[str | None, typer.Option("--session-id")] = None,
    workdir: Annotated[Path | None, typer.Option("--workdir", file_okay=False)] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
    images: Annotated[list[str] | None, typer.Option("--image", help="Local image path or HTTP(S) URL; repeat for multiple images.")] = None,
) -> None:
    if not text.strip() and not images:
        raise typer.Exit(code=0)
    try:
        loaded = _load(config)
        session = SessionCatalog.open(loaded, session_id) if session_id else None
        input_workdir = session.meta.workdir if session else (workdir or Path.cwd())
        if session and workdir and workdir.expanduser().resolve() != Path(session.meta.workdir):
            raise InputError("session_workdir_mismatch: an existing session keeps its original workdir")
        message = build_human_message(text, images=images or (), workdir=input_workdir, config=loaded)
        if session_id is None:
            session = create_session(loaded, workdir=str(workdir) if workdir else None)
            session_id = session.session_id
        result = asyncio.run(ask(loaded, session_id, message))
        _print_result(result, as_json)
        if result.status != "succeeded":
            raise typer.Exit(code=_exit_code(result))
    except (ConfigError, SessionError, SessionBusy, InputError) as exc:
        if as_json:
            typer.echo(json.dumps({"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)}}, ensure_ascii=False))
        else:
            typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("approvals")
def approvals_command(
    session_id: Annotated[str, typer.Option("--session-id")],
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path("config/default.toml"),
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    try:
        loaded = _load(config)
        pending = asyncio.run(get_pending_approvals(loaded, session_id))
    except Exception as exc:
        _print_error(exc, as_json)
        raise typer.Exit(code=1) from exc
    if as_json:
        typer.echo(pending.model_dump_json(indent=2))
    else:
        _print_approvals(pending)


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


@app.command("resume")
def resume_command(
    session_id: Annotated[str, typer.Option("--session-id")],
    decisions_file: Annotated[Path, typer.Option("--decisions-file", dir_okay=False)],
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path("config/default.toml"),
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    try:
        decisions = json.loads(decisions_file.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, ValueError) as exc:
        _print_error(exc, as_json, code="invalid_approval_decisions")
        raise typer.Exit(code=1) from exc
    try:
        loaded = _load(config)
        session = SessionCatalog.open(loaded, session_id)
        result = asyncio.run(resume(loaded, session_id, decisions))
    except (ConfigError, SessionError) as exc:
        _print_error(exc, as_json)
        raise typer.Exit(code=1) from exc
    _print_result(result, as_json)
    if result.status != "succeeded":
        raise typer.Exit(code=_exit_code(result))


@app.command("create-session")
def create_session_command(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path("config/default.toml"),
    workdir: Annotated[Path | None, typer.Option("--workdir", file_okay=False)] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    try:
        session = create_session(_load(config), workdir=str(workdir) if workdir else None)
    except (ConfigError, SessionError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    payload = {"session_id": session.session_id, "session_path": str(session.path)}
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2) if as_json else session.session_id)


@app.command("list-session")
def list_session_command(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path("config/default.toml"),
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    try:
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
    session_id: Annotated[str, typer.Option("--session-id")],
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path("config/default.toml"),
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    try:
        session = SessionCatalog.close(_load(config), session_id)
    except (ConfigError, SessionError, SessionBusy) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    payload = session.meta.model_dump(mode="json")
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2) if as_json else session.session_id)


async def _interactive(config_path: Path, *, session_id: str | None, workdir: Path | None) -> None:
    try:
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
            result = await resume(config, session.session_id, decisions)
            if result.status != "interrupted":
                _print_result(result, False)
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
            typer.echo("/status  show session status\n/close   close session\n/quit    leave session running\n/exit    leave session running")
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
                session = create_session(config, workdir=str(workdir) if workdir else None)
            result = await ask(config, session.session_id, text)
            if result.interrupts:
                pending = ApprovalStatus(**{key: getattr(result, key) for key in ApprovalStatus.model_fields})
            else:
                _print_result(result, False)
        except (ConfigError, SessionError, SessionBusy) as exc:
            typer.echo(f"Error: {exc}", err=True)

    if session is not None and session.meta.status == "running":
        typer.echo(f"Use rsh --session-id {session.session_id} to resume this session.", err=True)


async def _collect_approvals(pending: ApprovalStatus) -> dict | None:
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
                    typer.echo("请输入 y/yes 或 n/no。", err=True)
            decisions_by_interrupt[item["id"]] = {"decisions": decisions}
    except (EOFError, KeyboardInterrupt):
        return None
    return decisions_by_interrupt


def main() -> None:
    app()
