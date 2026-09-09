"""Typer command-line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from .agent_loop import InputRequest, close_session, create_session, run
from .config import Config, ConfigError
from .runtime.session import Session, SessionState


def _configure_readline() -> None:
    """Apply the terminal bindings used by the interactive reference client."""

    try:
        import readline

        readline.parse_and_bind("set bind-tty-special-chars off")
        readline.parse_and_bind("set input-meta on")
        readline.parse_and_bind("set output-meta on")
        readline.parse_and_bind("set convert-meta off")
    except ImportError:
        pass


_configure_readline()


def _ask_impl(
    query: str | None,
    image: str | None,
    session_id: str | None,
    config: Path,
    as_json: bool,
    workdir: Path | str | None = None,
) -> int:
    request = InputRequest(query=query, image=image)
    # Empty input is intentionally silent and must not load Config or a model.
    if not request.has_input():
        return 0

    try:
        loaded_config = Config.load(config)
    except ConfigError as exc:
        _print_error(exc.code, exc.message, as_json=as_json)
        return 1

    if session_id is None:
        try:
            session_id = create_session(
                loaded_config, query=query, image=image, workdir=workdir
            ).state.session_id
        except Exception as exc:
            _print_error("runtime_error", f"Unable to create session: {exc}", as_json=as_json)
            return 1
    result = run(session_id, request, loaded_config, tool_log=_cli_tool_log, workdir=workdir)
    if as_json:
        _echo(result.model_dump_json(indent=2))
        return {"succeeded": 0, "not_supported": 2, "failed": 1}[result.status]
    elif result.status == "succeeded":
        _echo(result.answer or "")
        return 0
    elif result.status == "not_supported":
        _echo("当前版本尚未支持仅影像输入的探索模式。", err=True)
        return 2
    else:
        error = result.errors[0] if result.errors else None
        message = f"{error.code}: {error.message}" if error else "request failed"
        _echo(message, err=True)
        return 1


app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.callback(invoke_without_command=True)
def _root_callback(
    ctx: typer.Context,
    config: Annotated[Path, typer.Option("--config", exists=False)] = Path("config/default.toml"),
    session_id: Annotated[str | None, typer.Option("--session-id")] = None,
    workdir: Annotated[Path | None, typer.Option("--workdir")] = None,
) -> None:
    """RS-Harness command group."""
    ctx.ensure_object(dict)
    ctx.obj["workdir"] = workdir
    if ctx.invoked_subcommand is None:
        raise typer.Exit(_interactive_impl(config, session_id, workdir))


@app.command()
def ask(
    ctx: typer.Context,
    query: Annotated[str | None, typer.Argument(help="Natural-language query.")] = None,
    image: Annotated[
        str | None, typer.Option("--image", help="Opaque image path, URI or object ID.")
    ] = None,
    session_id: Annotated[str | None, typer.Option("--session-id", help="Persistent session id; omitted to create one.")] = None,
    workdir: Annotated[Path | None, typer.Option("--workdir", help="Workspace directory for tools.")] = None,
    config: Annotated[Path, typer.Option("--config", exists=False, help="TOML Config path.")] = Path(
        "config/default.toml"
    ),
    as_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Print the SessionResult as JSON; tool logs may precede it.",
        ),
    ] = False,
) -> None:
    root_workdir = (ctx.obj or {}).get("workdir")
    effective_workdir = workdir if workdir is not None else root_workdir
    raise typer.Exit(_ask_impl(query, image, session_id, config, as_json, effective_workdir))


@app.command("create-session")
def create_session_command(
    ctx: typer.Context,
    config: Annotated[Path, typer.Option("--config", exists=False)] = Path("config/default.toml"),
    workdir: Annotated[Path | None, typer.Option("--workdir", help="Workspace directory for tools.")] = None,
) -> None:
    root_workdir = (ctx.obj or {}).get("workdir")
    effective_workdir = workdir if workdir is not None else root_workdir
    try:
        session = create_session(Config.load(config), workdir=effective_workdir)
    except ConfigError as exc:
        _print_error(exc.code, exc.message, as_json=False)
        raise typer.Exit(1) from exc
    except Exception as exc:
        _print_error("runtime_error", str(exc), as_json=False)
        raise typer.Exit(1) from exc
    _echo(json.dumps({"session_id": session.state.session_id, "session_path": str(session.path)}, ensure_ascii=False))


@app.command("close-session")
def close_session_command(
    session_id: Annotated[str, typer.Option("--session-id")],
    config: Annotated[Path, typer.Option("--config", exists=False)] = Path("config/default.toml"),
) -> None:
    _echo(close_session(Config.load(config), session_id).model_dump_json(indent=2))


@app.command("list-session")
def list_session_command(
    config: Annotated[Path, typer.Option("--config", exists=False)] = Path("config/default.toml"),
) -> None:
    """Print every session state under the configured sessions directory."""
    raise typer.Exit(_list_sessions_impl(config))


def _print_error(code: str, message: str, *, as_json: bool) -> None:
    if as_json:
        _echo(
            json.dumps(
                {"status": "failed", "errors": [{"code": code, "message": message}]},
                ensure_ascii=False,
            )
        )
    else:
        _echo(f"{code}: {message}", err=True)


def _session_summary(state: SessionState) -> dict[str, object]:
    # Keep the listing scannable: full answers and error details stay in
    # session.json and /status.
    return {
        "session_id": state.session_id,
        "status": state.status,
        "mode": state.mode,
        "query": state.query,
        "workdir": state.workdir,
        "turn": state.turn,
        "message_count": state.message_count,
        "error_count": len(state.errors),
        "started_at": state.started_at,
        "finished_at": state.finished_at,
    }


def _list_sessions_impl(config_path: Path) -> int:
    try:
        config = Config.load(config_path)
        sessions = Session.list(config)
    except ConfigError as exc:
        _print_error(exc.code, exc.message, as_json=False)
        return 1
    except Exception as exc:
        _print_error("runtime_error", str(exc), as_json=False)
        return 1
    _echo(
        json.dumps(
            [_session_summary(state) for state in sessions], ensure_ascii=False, indent=2
        )
    )
    return 0


def _echo(message: str, *, err: bool = False) -> None:
    print(message, file=sys.stderr if err else sys.stdout)


def _format_session_history(messages: list[dict[str, object]]) -> str:
    """Render user and assistant text from a session ledger for the CLI."""

    entries: list[str] = []
    for message in messages:
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        content = message.get("content")
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text_blocks = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
                and part.get("type", "text") == "text"
                and isinstance(part.get("text"), str)
            ]
            text = "\n".join(text_blocks).strip()
        else:
            text = ""
        if text:
            entries.append(f"{role}> {text}")
    return "\n".join(entries)


def _cli_tool_log(name: str, preview: str) -> None:
    # Tool logs share stdout with the answer/JSON payload; keep escape codes
    # out of pipes and redirected output.
    if sys.stdout.isatty():
        print(f"\033[33m> {name}\033[0m")
    else:
        print(f"> {name}")
    print(preview)


def _interactive_impl(
    config_path: Path,
    session_id: str | None,
    workdir: Path | str | None = None,
) -> int:
    resumed = bool(session_id)
    try:
        config = Config.load(config_path)
        if session_id:
            session = Session.open(config, session_id)
        else:
            kwargs = {} if workdir is None else {"workdir": workdir}
            session = create_session(config, **kwargs)
    except Exception as exc:
        _print_error("runtime_error", str(exc), as_json=False)
        return 1

    session_id = session.state.session_id
    _echo(f"rsh session {session_id}")
    if resumed:
        history = _format_session_history(session.messages())
        if history:
            _echo(f"Session history:\n{history}\n")
    _echo("Type /help for commands.\n")
    while True:
        try:
            text = input("\033[34mrsh>\033[0m ")
        except (EOFError, KeyboardInterrupt):
            _echo(f"\nSession remains running: {session_id}")
            _echo(f"Resume with: uv run rsh --session-id {session_id}")
            return 0
        text = text.strip()
        if not text:
            continue
        if text == "/help":
            _echo(
                "/help  show commands\n"
                "/status  show session status\n"
                "/close  close session\n"
                "/quit  leave session running\n"
                "/exit  leave session running"
            )
            continue
        if text == "/status":
            session = Session.open(config, session_id)
            _echo(json.dumps(session.state.model_dump(mode="json"), ensure_ascii=False, indent=2))
            continue
        if text in ("/quit", "/exit"):
            _echo(f"Session remains running: {session_id}")
            _echo(f"Resume with: uv run rsh --session-id {session_id}")
            return 0
        if text == "/close":
            _echo(session.close().model_dump_json(indent=2))
            return 0
        if text.startswith("/"):
            _echo(f"Unknown command: {text}", err=True)
            continue

        result = run(session_id, InputRequest(query=text), config, tool_log=_cli_tool_log, workdir=workdir)
        if result.status == "succeeded":
            _echo(result.answer or "")
        elif result.status == "not_supported":
            _echo("当前版本尚未支持该输入。", err=True)
        else:
            error = result.errors[0] if result.errors else None
            _echo(f"{error.code}: {error.message}" if error else "request failed", err=True)


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
