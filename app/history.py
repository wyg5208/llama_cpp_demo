"""Chat session store: one index of metadata plus one JSON file per conversation.

Two tiers because a session keeps its uploaded images as base64 data URLs and can
therefore run to megabytes, while the sidebar has to list every conversation on
each load. The index carries a short digest of the message text, so searching a
conversation's body never means opening it.

Takes plain Paths and imports nothing from this package, so it can be driven
straight from a test against a temp directory. Everything here is synchronous;
callers that must not block the event loop wrap it in run_in_threadpool.

Reads swallow errors and report "absent" — a damaged listing must not take the
whole UI down, and the session bodies are still on disk. Writes raise instead:
silently discarding a conversation would leave the user believing it was saved,
which is the opposite of the trade-off models.py makes for the model choice.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

log = logging.getLogger(__name__)

INDEX_NAME = "index.json"
# Ids are minted here, never supplied by a caller, and this is checked before any
# of them reaches a path: nothing client-supplied is concatenated into a filename.
SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
TITLE_CHARS = 40
# Per-message rather than one running cap: an assistant answer is often 500-2000
# characters, so a flat 4000 would index only the first exchange or two and search
# would silently miss everything deeper in the conversation.
DIGEST_MSG_CHARS = 200
DIGEST_CHARS = 4000

# Serialises every read-modify-write of the index. run.py starts uvicorn without
# workers, so one process holds one lock and that is enough; two threads writing
# the same temp name would otherwise race, too.
_LOCK = threading.Lock()


def new_session_id() -> str:
    return uuid4().hex


def is_session_id(value: str) -> bool:
    return bool(SESSION_ID_RE.match(value or ""))


def _now() -> str:
    # Lexicographic order is chronological order, which is what list_sessions sorts on.
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _session_path(directory: Path, session_id: str) -> Path:
    return directory / f"{session_id}.json"


def _atomic_write_json(path: Path, payload) -> None:
    """Replace `path` whole, or leave the previous version untouched.

    Raises OSError on purpose — see the module docstring.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_index(directory: Path) -> list[dict]:
    path = directory / INDEX_NAME
    data = _read_json(path)
    if data is None:
        if path.is_file():
            log.warning("ignoring unreadable session index %s; the session files are still there", path)
        return []
    if not isinstance(data, list):
        log.warning("ignoring malformed session index %s", path)
        return []
    return [record for record in data if isinstance(record, dict)]


def _write_index(directory: Path, records: list[dict]) -> None:
    _atomic_write_json(directory / INDEX_NAME, records)


def _digest(messages: list[dict]) -> str:
    parts = [(m.get("content") or "")[:DIGEST_MSG_CHARS] for m in messages if isinstance(m, dict)]
    return "\n".join(p for p in parts if p.strip())[:DIGEST_CHARS]


def auto_title(messages: list[dict]) -> str:
    """First line of the opening user message.

    Never empty in practice — the browser refuses to send a text-less turn — but
    a migrated or hand-edited session may have none, so fall back.
    """
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        first = (msg.get("content") or "").strip().splitlines()
        if first and first[0].strip():
            return first[0].strip()[:TITLE_CHARS]
    return "新对话"


def _record(body: dict) -> dict:
    """The index entry for a session body — everything the sidebar needs, no images."""
    messages = body.get("messages") or []
    return {
        "id": body.get("id", ""),
        "title": body.get("title") or "新对话",
        "created": body.get("created", ""),
        "updated": body.get("updated", ""),
        "message_count": len(messages),
        "model": body.get("model", ""),
        "digest": _digest(messages),
    }


def list_sessions(directory: Path) -> list[dict]:
    """Every live session, most recently updated first."""
    with _LOCK:
        records = _read_index(directory)
        live = [
            r for r in records
            if is_session_id(str(r.get("id", ""))) and _session_path(directory, r["id"]).is_file()
        ]
        if len(live) != len(records):
            # A body removed by hand, or a crash between the two writes of a
            # create, must not leave a row in the sidebar that 404s when clicked.
            try:
                _write_index(directory, live)
            except OSError as exc:
                log.warning("could not prune the session index: %s", exc)
    live.sort(key=lambda r: r.get("updated", ""), reverse=True)
    return live


def read_session(directory: Path, session_id: str) -> dict | None:
    body = _read_json(_session_path(directory, session_id))
    if not isinstance(body, dict):
        return None
    if not isinstance(body.get("messages"), list):
        body["messages"] = []
    return body


def create_session(directory: Path, messages: list[dict], model: str, title: str = "") -> dict:
    """Store a new conversation and return its index entry."""
    session_id = new_session_id()
    stamp = _now()
    body = {
        "id": session_id,
        "title": title.strip()[:TITLE_CHARS] or auto_title(messages),
        "created": stamp,
        "updated": stamp,
        "model": model,
        "messages": messages,
    }
    record = _record(body)
    with _LOCK:
        # Body before index: a crash in between leaves an unlisted file, which is
        # invisible and wastes disk. The reverse would list a session that 404s.
        _atomic_write_json(_session_path(directory, session_id), body)
        _write_index(directory, [record, *_read_index(directory)])
    return record


def update_session(directory: Path, session_id: str, patch: dict) -> dict | None:
    """Merge `patch` into an existing session; None when there is nothing to update.

    Only the keys present are touched, so renaming a conversation from the sidebar
    does not have to ship its images back to the server.
    """
    with _LOCK:
        body = read_session(directory, session_id)
        if body is None or body.get("id") != session_id:
            return None
        if isinstance(patch.get("messages"), list):
            body["messages"] = patch["messages"]
        new_title = (patch.get("title") or "").strip()
        if new_title:
            body["title"] = new_title[:TITLE_CHARS]
        body["updated"] = _now()
        record = _record(body)
        path = _session_path(directory, session_id)
        _atomic_write_json(path, body)
        records = _read_index(directory)
        _write_index(directory, [record, *(r for r in records if r.get("id") != session_id)])
    return record


def delete_session(directory: Path, session_id: str) -> bool:
    with _LOCK:
        records = _read_index(directory)
        kept = [r for r in records if r.get("id") != session_id]
        if len(kept) == len(records):
            return False
        _write_index(directory, kept)
    try:
        _session_path(directory, session_id).unlink()
    except OSError as exc:
        # Already unlisted, so the worst case is a file nobody can reach.
        log.warning("could not remove session file %s: %s", session_id, exc)
    return True
