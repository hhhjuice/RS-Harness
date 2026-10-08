"""Official SQLite Checkpointer lifecycle."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from ..config import Config


@asynccontextmanager
async def open_checkpointer(config: Config, *, read_only: bool = False) -> AsyncIterator[object]:
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError as exc:  # pragma: no cover - dependency setup failure
        raise RuntimeError("langgraph-checkpoint-sqlite is required") from exc

    if read_only:
        import aiosqlite

        uri = config.runtime.checkpoint_file.resolve().as_uri() + "?mode=ro"
        async with aiosqlite.connect(uri, uri=True) as connection:
            saver = AsyncSqliteSaver(connection)
            # This connection only reads an existing database; never run setup's
            # schema migrations or journal-mode writes on an inspection path.
            saver.is_setup = True
            yield saver
        return

    config.runtime.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(config.runtime.checkpoint_file)) as saver:
        setup = getattr(saver, "setup", None)
        if setup is not None:
            await setup()
        yield saver
