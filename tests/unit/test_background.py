from __future__ import annotations

import threading
import time

from rs_harness.runtime.background import BackgroundTaskManager


def test_jobs_overlap_and_complete() -> None:
    manager = BackgroundTaskManager("/tmp/rsh-tests", max_workers=2)
    barrier = threading.Barrier(2)
    try:
        def runner(cancel_check):
            barrier.wait(timeout=2)
            return "ok"

        manager.submit("a", "one", runner)
        manager.submit("b", "two", runner)
        assert manager.wait("a", 2).status == "succeeded"
        assert manager.wait("b", 2).status == "succeeded"
    finally:
        manager.shutdown(wait=True)


def test_fifo_when_worker_is_bounded() -> None:
    manager = BackgroundTaskManager("/tmp/rsh-tests", max_workers=1)
    order: list[str] = []
    gate = threading.Event()
    try:
        def first(cancel_check):
            order.append("first")
            gate.wait(2)
            return "1"

        def second(cancel_check):
            order.append("second")
            return "2"

        manager.submit("a", "one", first)
        manager.submit("b", "two", second)
        time.sleep(0.02)
        assert order == ["first"]
        gate.set()
        assert manager.wait("b", 2).status == "succeeded"
        assert order == ["first", "second"]
    finally:
        manager.shutdown(wait=True)


def test_cancel_queued_and_running_jobs() -> None:
    manager = BackgroundTaskManager("/tmp/rsh-tests", max_workers=1)
    gate = threading.Event()
    try:
        def running(cancel_check):
            while not cancel_check():
                gate.wait(0.01)
            return "stopped"

        manager.submit("a", "one", running)
        assert manager.wait("a", 1).status in {"running", "cancel_requested"}
        assert manager.request_cancel("a").status == "cancel_requested"
        assert manager.wait("a", 1).status == "cancelled"

        manager.submit("b", "two", lambda check: "never")
        assert manager.request_cancel("b").status == "cancelled"
        assert manager.wait("b", 1).status == "cancelled"
    finally:
        manager.shutdown(wait=True)


def test_duplicate_submission_rejected_until_done() -> None:
    manager = BackgroundTaskManager("/tmp/rsh-tests", max_workers=1)
    gate = threading.Event()
    try:
        manager.submit("a", "one", lambda check: (gate.wait(1), "ok")[1])
        try:
            manager.submit("a", "again", lambda check: "bad")
        except ValueError as exc:
            assert "already" in str(exc)
        else:  # pragma: no cover - defensive assertion
            raise AssertionError("duplicate submission was accepted")
        gate.set()
        assert manager.wait("a", 1).status == "succeeded"
        manager.submit("a", "again", lambda check: "ok")
        assert manager.wait("a", 1).status == "succeeded"
    finally:
        manager.shutdown(wait=True)
