import json
from uuid import uuid4

import pytest

from rs_harness.sessions.catalog import SessionCatalog, SessionError


def test_create_open_list_and_close_session(config, tmp_path):
    session = SessionCatalog.create(config, workdir=tmp_path)
    assert (session.path / "session.json").is_file()
    assert (session.path / "trajectory.jsonl").is_file()
    meta = json.loads(session.state_path.read_text())
    assert meta["status"] == "running"
    assert meta["session_id"] == session.session_id
    assert "thread_id" not in meta

    reopened = SessionCatalog.open(config, session.session_id)
    assert reopened.meta.workdir == str(tmp_path.resolve())
    assert len(SessionCatalog.list(config)) == 1

    closed = SessionCatalog.close(config, session.session_id)
    assert closed.meta.status == "closed"
    with pytest.raises(SessionError, match="session_closed"):
        with closed.interaction_lock():
            pass


def test_session_identity_mismatch_is_rejected_and_not_listed(config):
    session = SessionCatalog.create(config)
    meta = json.loads(session.state_path.read_text())
    meta["session_id"] = uuid4().hex
    session.state_path.write_text(json.dumps(meta))

    with pytest.raises(SessionError, match="session_identity_mismatch"):
        SessionCatalog.open(config, session.session_id)
    assert SessionCatalog.list(config) == []


def test_thread_id_is_rejected_in_new_metadata(config):
    session = SessionCatalog.create(config)
    meta = json.loads(session.state_path.read_text())
    meta["thread_id"] = session.session_id
    session.state_path.write_text(json.dumps(meta))

    with pytest.raises(SessionError, match="invalid_session"):
        SessionCatalog.open(config, session.session_id)
    assert json.loads(session.state_path.read_text()) == meta
