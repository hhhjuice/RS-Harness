"""Process-local background execution for delegated sub-agent turns.

The manager deliberately owns no model or session objects.  Callers provide a
runner closure that performs one child turn and an optional ``on_status``
callback which persists execution metadata through the Session API.  Workers
are daemon threads: unfinished jobs are therefore considered interrupted by
the next process that inspects the persisted session.
"""

from __future__ import annotations

import os
import queue
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Literal

ExecutionStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancel_requested",
    "cancelled",
    "interrupted",
]

Runner = Callable[[Callable[[], bool]], str | None]
StatusCallback = Callable[..., None]
CONSOLE_LOCK = threading.RLock()


@dataclass(frozen=True)
class TaskSnapshot:
    """Serializable view of a background execution."""

    child_id: str
    execution_id: str
    status: ExecutionStatus
    answer: str | None = None
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    name: str | None = None

    @property
    def execution_status(self) -> ExecutionStatus:
        """Alias matching the persisted SessionState field name."""
        return self.status

    @property
    def session_id(self) -> str:
        return self.child_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.child_id,
            "name": self.name,
            "execution_id": self.execution_id,
            "execution_status": self.status,
            "execution_answer": self.answer,
            "execution_error": self.error,
            "execution_started_at": self.started_at,
            "execution_finished_at": self.finished_at,
        }


@dataclass
class _Job:
    child_id: str
    prompt: str
    parent_id: str | None
    runner: Runner
    on_status: StatusCallback | None
    config: Any | None = None
    name: str | None = None
    execution_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    future: Any = None
    status: ExecutionStatus = "queued"
    answer: str | None = None
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None


class BackgroundTaskManager:
    """FIFO, bounded worker pool for sub-agent turns.

    ``submit`` never waits for the runner.  A child id may have only one
    queued/running job at a time; callers can inspect ``status`` before
    submitting a subsequent turn.
    """

    _instances: dict[tuple[str, int], "BackgroundTaskManager"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def for_sessions_dir(cls, sessions_dir: str | Path, max_workers: int = 4) -> "BackgroundTaskManager":
        key = (str(Path(sessions_dir).expanduser().resolve()), max(1, int(max_workers)))
        with cls._instances_lock:
            manager = cls._instances.get(key)
            if manager is None:
                manager = cls(Path(key[0]), key[1])
                cls._instances[key] = manager
            return manager

    def __init__(self, sessions_dir: str | Path, max_workers: int = 4):
        self.sessions_dir = Path(sessions_dir).expanduser().resolve()
        self.max_workers = max(1, int(max_workers))
        self._queue: queue.Queue[_Job | None] = queue.Queue()
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._shutdown = False
        self.console_lock = CONSOLE_LOCK
        self._threads = [
            threading.Thread(
                target=self._worker,
                name=f"rsh-subagent-{index + 1}",
                daemon=True,
            )
            for index in range(self.max_workers)
        ]
        for thread in self._threads:
            thread.start()

    def submit(
        self,
        child_id: str,
        prompt: str,
        runner: Runner,
        *,
        parent_id: str | None = None,
        on_status: StatusCallback | None = None,
        context: Any | None = None,
        config: Any | None = None,
        name: str | None = None,
    ) -> TaskSnapshot:
        """Queue one child turn and return immediately.

        ``context`` may bundle a config for automatic Session execution-state
        persistence; model/client/hooks remain owned by the runner closure.
        The runner receives a cancellation predicate as its sole argument.
        """
        if config is None and isinstance(context, dict):
            config = context.get("config")
        if not child_id:
            raise ValueError("child_id must not be empty")
        if not callable(runner):
            raise TypeError("runner must be callable")
        with self._condition:
            if self._shutdown:
                raise RuntimeError("background manager is shut down")
            existing = self._jobs.get(child_id)
            if existing and existing.status in ("queued", "running", "cancel_requested"):
                raise ValueError("child already has a running background task")
            if on_status is None and config is not None:
                on_status = self._session_status_callback(config, child_id)
            job = _Job(child_id, prompt, parent_id, runner, on_status, config=config, name=name)
            self._jobs[child_id] = job
            self._notify(job)
            self._queue.put(job)
            self._condition.notify_all()
            return self._snapshot(job)

    def status(self, child_id: str) -> TaskSnapshot | None:
        with self._condition:
            job = self._jobs.get(child_id)
            return None if job is None else self._snapshot(job)

    lookup = status

    def wait(
        self,
        child_id: str,
        timeout: float | None = None,
        *,
        timeout_ms: int | None = None,
    ) -> TaskSnapshot | None:
        """Wait for completion, returning the latest snapshot on timeout too."""
        if timeout is None and timeout_ms is not None:
            timeout = max(0, timeout_ms) / 1000
        deadline = None if timeout is None else monotonic() + max(0.0, timeout)
        with self._condition:
            while True:
                job = self._jobs.get(child_id)
                if job is None:
                    return None
                if job.status not in ("queued", "running", "cancel_requested"):
                    return self._snapshot(job)
                if deadline is None:
                    self._condition.wait()
                else:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        return self._snapshot(job)
                    self._condition.wait(remaining)

    def request_cancel(self, child_id: str) -> TaskSnapshot | None:
        with self._condition:
            job = self._jobs.get(child_id)
            if job is None:
                return None
            if job.status == "queued":
                job.cancel_event.set()
                job.status = "cancelled"
                job.finished_at = monotonic()
                self._notify(job)
            elif job.status == "running":
                job.cancel_event.set()
                job.status = "cancel_requested"
                # Do not call the persistence callback while a worker may
                # hold the child session lock. The worker records the
                # terminal cancellation after its current model/tool batch.
            self._condition.notify_all()
            return self._snapshot(job)

    def shutdown(self, *, cancel: bool = True, wait: bool = False) -> None:
        with self._condition:
            self._shutdown = True
            if cancel:
                for job in self._jobs.values():
                    if job.status in ("queued", "running"):
                        job.cancel_event.set()
                        job.status = "cancelled" if job.status == "queued" else "cancel_requested"
                        self._notify(job)
            for _ in self._threads:
                self._queue.put(None)
            self._condition.notify_all()
        if wait:
            for thread in self._threads:
                thread.join()
        # Allow a fresh manager to be created for this directory after an
        # explicit shutdown (useful for CLI/session lifecycle and tests).
        key = (str(self.sessions_dir), self.max_workers)
        with self._instances_lock:
            if self._instances.get(key) is self:
                self._instances.pop(key, None)

    cancel = request_cancel

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                with self._condition:
                    if job.status == "cancelled":
                        self._condition.notify_all()
                        continue
                    if job.status != "queued":
                        continue
                    job.status = "running"
                    job.started_at = monotonic()
                    claimed = self._notify(job)
                    if claimed is False:
                        job.cancel_event.set()
                        job.status = "cancelled"
                        job.finished_at = monotonic()
                        self._notify(job)
                        self._condition.notify_all()
                        continue
                    self._condition.notify_all()
                try:
                    answer = job.runner(job.cancel_event.is_set)
                    with self._condition:
                        job.answer = answer
                        job.status = "cancelled" if job.cancel_event.is_set() else "succeeded"
                        job.finished_at = monotonic()
                        self._notify(job)
                        self._condition.notify_all()
                except Exception as exc:  # runner errors become task observations
                    with self._condition:
                        job.error = str(exc)
                        job.status = "cancelled" if job.cancel_event.is_set() else "failed"
                        job.finished_at = monotonic()
                        self._notify(job)
                        self._condition.notify_all()
            finally:
                self._queue.task_done()

    def _snapshot(self, job: _Job) -> TaskSnapshot:
        return TaskSnapshot(
            child_id=job.child_id,
            execution_id=job.execution_id,
            status=job.status,
            name=job.name,
            answer=job.answer,
            error=job.error,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )

    def _notify(self, job: _Job) -> object | None:
        if job.on_status is None:
            return None
        snapshot = self._snapshot(job)
        try:
            return job.on_status(
                snapshot.status,
                execution_id=snapshot.execution_id,
                answer=snapshot.answer,
                error=snapshot.error,
                started_at=snapshot.started_at,
                finished_at=snapshot.finished_at,
                owner_pid=os.getpid(),
            )
        except Exception:
            # Persistence failures must not kill a worker; the parent can
            # observe the in-memory failure and retry status persistence.
            return None

    @staticmethod
    def _session_status_callback(config: Any, child_id: str) -> StatusCallback:
        """Adapt worker state updates to ``Session.update_execution``."""

        def persist(
            status: ExecutionStatus,
            *,
            execution_id: str,
            answer: str | None = None,
            error: str | None = None,
            **_: Any,
        ) -> bool | None:
            from .session import ErrorInfo, Session

            session = Session.open(config, child_id)
            if status == "queued":
                session.update_execution(status, execution_id=execution_id)
                return True
            if status == "running":
                # Worker claims queued work atomically before this callback.
                return session.claim_execution(execution_id)
            info = ErrorInfo(code="subagent_failed", message=error) if error else None
            session.update_execution(
                status,
                execution_id=execution_id,
                answer=answer,
                error=info,
            )
            return True

        return persist


def get_background_task_manager(
    sessions_dir: str | Path, max_workers: int = 4
) -> BackgroundTaskManager:
    """Return the process-local singleton for a sessions directory."""
    return BackgroundTaskManager.for_sessions_dir(sessions_dir, max_workers)


__all__ = [
    "BackgroundTaskManager",
    "CONSOLE_LOCK",
    "ExecutionStatus",
    "TaskSnapshot",
    "get_background_task_manager",
]
