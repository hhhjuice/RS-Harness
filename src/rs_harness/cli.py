"""Typer command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Annotated

try:  # Typer is the documented CLI stack; keep imports usable with old Click.
    import typer
except Exception:  # pragma: no cover - only exercised by an incompatible env
    typer = None  # type: ignore[assignment]

from .agent_loop import InputRequest, close_session, create_session, run
from .config import Config, ConfigError
from .runtime.session import Session


def _ask_impl(query: str | None, image: str | None, session_id: str | None, config: Path, as_json: bool) -> int:
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
            session_id = create_session(loaded_config, query=query, image=image).state.session_id
        except Exception as exc:
            _print_error("runtime_error", f"Unable to create session: {exc}", as_json=as_json)
            return 1
    result = run(session_id, request, loaded_config)
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


if typer is not None:
    app = typer.Typer(add_completion=False, no_args_is_help=False)

    @app.callback(invoke_without_command=True)
    def _root_callback(
        ctx: typer.Context,
        config: Annotated[Path, typer.Option("--config", exists=False)] = Path("config/default.toml"),
        session_id: Annotated[str | None, typer.Option("--session-id")] = None,
    ) -> None:
        """RS-Harness command group."""
        if ctx.invoked_subcommand is None:
            raise typer.Exit(_interactive_impl(config, session_id))

    @app.command()
    def ask(
        query: Annotated[str | None, typer.Argument(help="Natural-language query.")] = None,
        image: Annotated[
            str | None, typer.Option("--image", help="Opaque image path, URI or object ID.")
        ] = None,
        session_id: Annotated[str | None, typer.Option("--session-id", help="Persistent session id; omitted to create one.")] = None,
        config: Annotated[Path, typer.Option("--config", exists=False, help="TOML Config path.")] = Path(
            "config/default.toml"
        ),
        as_json: Annotated[
            bool, typer.Option("--json", help="Print the complete SessionResult as JSON.")
        ] = False,
    ) -> None:
        raise typer.Exit(_ask_impl(query, image, session_id, config, as_json))

    @app.command("create-session")
    def create_session_command(
        config: Annotated[Path, typer.Option("--config", exists=False)] = Path("config/default.toml"),
    ) -> None:
        session = create_session(Config.load(config))
        _echo(json.dumps({"session_id": session.state.session_id, "session_path": str(session.path)}, ensure_ascii=False))

    @app.command("close-session")
    def close_session_command(
        session_id: Annotated[str, typer.Option("--session-id")],
        config: Annotated[Path, typer.Option("--config", exists=False)] = Path("config/default.toml"),
    ) -> None:
        _echo(close_session(Config.load(config), session_id).model_dump_json(indent=2))
else:
    # This fallback only handles environments where Typer cannot import because
    # of an unrelated Click version mismatch. The project still declares Typer
    # as its normal CLI dependency.
    app = None

    def ask(
        query: str | None = None,
        image: str | None = None,
        session_id: str | None = None,
        config: Path = Path("config/default.toml"),
        as_json: bool = False,
    ) -> int:
        return _ask_impl(query, image, session_id, config, as_json)


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


def _echo(message: str, *, err: bool = False) -> None:
    print(message, file=sys.stderr if err else sys.stdout)


def _interactive_impl(config_path: Path, session_id: str | None) -> int:
    try:
        config = Config.load(config_path)
        session = Session.open(config, session_id) if session_id else create_session(config)
    except Exception as exc:
        _print_error("runtime_error", str(exc), as_json=False)
        return 1

    session_id = session.state.session_id
    _echo(f"rsh session {session_id}")
    _echo("Type /help for commands.\n")
    while True:
        try:
            text = input("rsh> ")
        except (EOFError, KeyboardInterrupt):
            _echo(f"\nSession remains running: {session_id}")
            _echo(f"Resume with: uv run rsh --session-id {session_id}")
            return 0
        text = text.strip()
        if not text:
            continue
        if text == "/help":
            _echo("/help  show commands\n/status  show session status\n/close  close session\n/quit  leave session running")
            continue
        if text == "/status":
            session = Session.open(config, session_id)
            _echo(json.dumps(session.state.model_dump(mode="json"), ensure_ascii=False, indent=2))
            continue
        if text == "/quit":
            _echo(f"Session remains running: {session_id}")
            _echo(f"Resume with: uv run rsh --session-id {session_id}")
            return 0
        if text == "/close":
            _echo(session.close().model_dump_json(indent=2))
            return 0
        if text.startswith("/"):
            _echo(f"Unknown command: {text}", err=True)
            continue

        result = run(session_id, InputRequest(query=text), config)
        if result.status == "succeeded":
            _echo(result.answer or "")
        elif result.status == "not_supported":
            _echo("当前版本尚未支持该输入。", err=True)
        else:
            error = result.errors[0] if result.errors else None
            _echo(f"{error.code}: {error.message}" if error else "request failed", err=True)


def main() -> None:
    if typer is not None:
        app()
        return
    parser = argparse.ArgumentParser(prog="rsh")
    parser.add_argument("--config", dest="root_config", type=Path, default=Path("config/default.toml"))
    parser.add_argument("--session-id", dest="root_session_id")
    subparsers = parser.add_subparsers(dest="command", required=False)
    create_parser = subparsers.add_parser("create-session")
    create_parser.add_argument("--config", type=Path, default=Path("config/default.toml"))
    close_parser = subparsers.add_parser("close-session")
    close_parser.add_argument("--session-id", required=True)
    close_parser.add_argument("--config", type=Path, default=Path("config/default.toml"))
    ask_parser = subparsers.add_parser("ask", help="Ask the configured model a question")
    ask_parser.add_argument("query", nargs="?")
    ask_parser.add_argument("--image")
    ask_parser.add_argument("--session-id")
    ask_parser.add_argument("--config", type=Path, default=Path("config/default.toml"))
    ask_parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    if args.command == "create-session":
        session = create_session(Config.load(args.config))
        _echo(json.dumps({"session_id": session.state.session_id, "session_path": str(session.path)}, ensure_ascii=False))
        return
    if args.command == "close-session":
        _echo(close_session(Config.load(args.config), args.session_id).model_dump_json(indent=2))
        return
    if args.command is None:
        raise SystemExit(_interactive_impl(args.root_config, args.root_session_id))
    raise SystemExit(_ask_impl(args.query, args.image, args.session_id, args.config, args.as_json))


if __name__ == "__main__":  # pragma: no cover
    main()
