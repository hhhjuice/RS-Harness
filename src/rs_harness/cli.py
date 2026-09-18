"""Typer CLI for local text sessions."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from prompt_toolkit import PromptSession

from . import __version__
from .application import InteractionResult, ask, create_session
from .config import Config, ConfigError
from .sessions.catalog import SessionBusy, SessionCatalog, SessionError

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _load(path: Path) -> Config:
    return Config.load(path)


def _print_result(result: InteractionResult, as_json: bool) -> None:
    payload = result.model_dump(mode="json")
    if as_json:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    elif result.status == "succeeded":
        typer.echo(result.answer or "")
    else:
        typer.echo(f"Error: {payload.get('error')}", err=True)


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path("config/default.toml"),
    session_id: Annotated[str | None, typer.Option("--session-id")] = None,
    workdir: Annotated[Path | None, typer.Option("--workdir", file_okay=False)] = None,
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    asyncio.run(_interactive(config, session_id=session_id, workdir=workdir))


@app.command("ask")
def ask_command(
    text: str,
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path("config/default.toml"),
    session_id: Annotated[str | None, typer.Option("--session-id")] = None,
    workdir: Annotated[Path | None, typer.Option("--workdir", file_okay=False)] = None,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    if not text.strip():
        raise typer.Exit(code=0)
    try:
        loaded = _load(config)
        if session_id is None:
            session = create_session(loaded, workdir=str(workdir) if workdir else None)
            session_id = session.session_id
        result = asyncio.run(ask(loaded, session_id, text))
        _print_result(result, as_json)
        if result.status != "succeeded":
            raise typer.Exit(code=1)
    except (ConfigError, SessionError, SessionBusy) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


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

    prompt = PromptSession("rsh> ") if sys.stdin.isatty() and sys.stdout.isatty() else None
    while True:
        try:
            if prompt is not None:
                text = await prompt.prompt_async()
            else:
                text = typer.prompt("rsh", prompt_suffix="> ", default="", show_default=False)
        except (EOFError, KeyboardInterrupt, typer.Abort):
            typer.echo()
            return
        command = text.strip()
        if not command:
            continue
        if command in {"/quit", "/exit"}:
            return
        if command == "/help":
            typer.echo("/status  show session status\n/close   close session\n/quit    leave session running\n/exit    leave session running")
            continue
        if command == "/status":
            if session is None:
                typer.echo("No session")
            else:
                typer.echo(json.dumps(session.meta.model_dump(mode="json"), ensure_ascii=False, indent=2))
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
            return

        try:
            if session is None:
                session = create_session(config, workdir=str(workdir) if workdir else None)
                typer.echo(f"session {session.session_id}")
            result = await ask(config, session.session_id, text)
            _print_result(result, False)
        except (ConfigError, SessionError, SessionBusy) as exc:
            typer.echo(f"Error: {exc}", err=True)


def main() -> None:
    app()
