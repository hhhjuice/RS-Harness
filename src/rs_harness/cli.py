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

from .agent_loop import InputRequest, run
from .config import Config, ConfigError


def _ask_impl(query: str | None, image: str | None, config: Path, as_json: bool) -> int:
    request = InputRequest(query=query, image=image)
    # Empty input is intentionally silent and must not load Config or a model.
    if not request.has_input():
        return 0

    try:
        loaded_config = Config.load(config)
    except ConfigError as exc:
        _print_error(exc.code, exc.message, as_json=as_json)
        return 1

    result = run(request, loaded_config)
    if result is None:
        return 0
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
    app = typer.Typer(add_completion=False, no_args_is_help=True)

    @app.callback()
    def _root_callback() -> None:
        """RS-Harness command group."""
        return None

    @app.command()
    def ask(
        query: Annotated[str | None, typer.Argument(help="Natural-language query.")] = None,
        image: Annotated[
            str | None, typer.Option("--image", help="Opaque image path, URI or object ID.")
        ] = None,
        config: Annotated[Path, typer.Option("--config", exists=False, help="TOML Config path.")] = Path(
            "config/default.toml"
        ),
        as_json: Annotated[
            bool, typer.Option("--json", help="Print the complete SessionResult as JSON.")
        ] = False,
    ) -> None:
        raise typer.Exit(_ask_impl(query, image, config, as_json))
else:
    # This fallback only handles environments where Typer cannot import because
    # of an unrelated Click version mismatch. The project still declares Typer
    # as its normal CLI dependency.
    app = None

    def ask(
        query: str | None = None,
        image: str | None = None,
        config: Path = Path("config/default.toml"),
        as_json: bool = False,
    ) -> int:
        return _ask_impl(query, image, config, as_json)


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


def main() -> None:
    if typer is not None:
        app()
        return
    parser = argparse.ArgumentParser(prog="rs-harness")
    subparsers = parser.add_subparsers(dest="command", required=False)
    ask_parser = subparsers.add_parser("ask", help="Ask the configured model a question")
    ask_parser.add_argument("query", nargs="?")
    ask_parser.add_argument("--image")
    ask_parser.add_argument("--config", type=Path, default=Path("config/default.toml"))
    ask_parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    if args.command != "ask":
        parser.print_help()
        return
    raise SystemExit(_ask_impl(args.query, args.image, args.config, args.as_json))


if __name__ == "__main__":  # pragma: no cover
    main()
