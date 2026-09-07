"""The model's long-term memory: one JSON file of short notes.

One tier, unlike history.py's two. history.py splits an index from the bodies
because a session carries its uploaded images as base64 data URLs and can run to
megabytes, so the sidebar must be able to list conversations without opening them.
A memory here is one sentence: 500 of them — the cap — is tens of kilobytes, so
there is nothing to index and no body too expensive to read.

Takes plain Paths and imports nothing from this package, so it can be driven
straight from a test against a temp directory. Everything is synchronous; callers
that must not block the event loop wrap it in run_in_threadpool.

Same asymmetric policy as history.py: reads swallow errors and report "absent"
because a damaged memory file must not take the chat UI down, while writes raise,
because silently discarding a note the model was told it had saved is worse than
an error it can retry.

KINDS is a closed set on purpose. The remember tool exposes it as an enum, but an
8B model will still send "note" or "info" now and then, and losing the memory over
a label would be the wrong trade — so an unknown kind degrades to "fact".
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


class MemoryStoreError(Exception):
    """A refusal whose message is safe to show the user verbatim.

    Named for the store, not `MemoryError`: that is a builtin, and shadowing it
    would turn every out-of-memory raise in the process into this class.
    """


MEMORY_NAME = "memory.json"

# Ids are minted here and never supplied by a caller, but delete_memory's id does
# arrive from the browser, so it is checked against this before anything else —
# same rule as history.SESSION_ID_RE. Unlike history there is no path to protect
# (one file for everything), but a bad id should still fail as a bad id.
MEMORY_ID_RE = re.compile(r"^[0-9a-f]{32}$")

# Over the cap, a new note is refused with an explanation rather than evicting the
# oldest: silent eviction means the model believes something is remembered when it
# is not, and the user has no way to notice. The panel can delete by hand.
MAX_MEMORIES = 500
# A memory is one sentence. Refusing an over-long one beats truncating it, because
# a half-remembered fact reads as a whole one.
MAX_MEMORY_CHARS = 500

KINDS = ("fact", "preference", "todo")
DEFAULT_KIND = "fact"

# Serialises every read-modify-write. run.py starts uvicorn without workers, so one
# process holds one lock and that is enough; two threads writing the same temp name
# would otherwise race, too.
_LOCK = threading.Lock()

_LATIN_RE = re.compile(r"[a-z0-9_]{2,}")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def new_memory_id() -> str:
    return uuid4().hex


def is_memory_id(value: str) -> bool:
    return bool(MEMORY_ID_RE.match(value or ""))


def _now() -> str:
    # Lexicographic order is chronological order, which is what the listings sort on.
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def _record_ok(record: dict) -> bool:
    return is_memory_id(str(record.get("id", ""))) and bool(str(record.get("text", "")).strip())


def _read_store(path: Path) -> list[dict]:
    """Every well-formed memory, oldest first. [] when the file is missing or broken."""
    data = _read_json(path)
    if data is None:
        if path.is_file():
            log.warning("ignoring unreadable memory file %s", path)
        return []
    if not isinstance(data, list):
        log.warning("ignoring malformed memory file %s", path)
        return []
    kept = [r for r in data if isinstance(r, dict) and _record_ok(r)]
    if len(kept) != len(data):
        log.warning("dropped %d malformed memories from %s", len(data) - len(kept), path.name)
    return kept


def list_memories(path: Path) -> list[dict]:
    """Every memory, newest first — the order the panel wants to show.

    The file is append-only, so reversing it is already newest-first and the sort
    only has to repair a hand-edited one. Both steps matter because `created` has
    second granularity: sorting the file order directly leaves everything saved in
    the same second oldest-first, while a stable sort over the REVERSED list keeps
    it newest-first.
    """
    return sorted(
        reversed(_read_store(path)), key=lambda r: str(r.get("created", "")), reverse=True
    )


def _terms(text: str) -> list[str]:
    """Matchable pieces of a query: latin words plus CJK bigrams.

    Splitting on whitespace alone is useless here, because Chinese has no spaces —
    导出格式 would arrive as a single token and match nothing. Bigrams are the
    zero-dependency answer, and they are also the honest limit of this search: it is
    keyword scoring, so 导出格式 finds a memory containing 导出 or 格式 but not one
    that says the same thing in other words. Nothing in the UI should imply more.
    """
    low = text.lower()
    terms = _LATIN_RE.findall(low)
    for run in _CJK_RE.findall(low):
        if len(run) == 1:
            terms.append(run)
        else:
            terms.extend(run[i : i + 2] for i in range(len(run) - 1))
    return terms


def _score(terms: set[str], record: dict) -> int:
    hay = str(record.get("text", "")).lower()
    kind = str(record.get("kind", "")).lower()
    total = 0
    for term in terms:
        if term in hay:
            # Length as weight: a longer match is a stronger signal than a bigram
            # that happens to appear.
            total += len(term)
        if term in kind:
            total += 1
    return total


def search_memories(path: Path, query: str, limit: int = 5) -> list[dict]:
    """The `limit` best keyword matches, best first, newest first on a tie.

    Returns [] for a blank query rather than the whole store: recall is meant to
    answer a question, and dumping 500 notes into the context is the failure this
    module exists to avoid.
    """
    terms = set(_terms(query or ""))
    if not terms:
        return []
    # Reversed for the same reason list_memories reverses: the sort below is stable,
    # so scoring newest-first is what makes equal scores at equal timestamps come out
    # newest-first rather than in file order.
    scored = [(_score(terms, r), r) for r in reversed(_read_store(path))]
    hits = [(s, r) for s, r in scored if s > 0]
    hits.sort(key=lambda pair: (pair[0], str(pair[1].get("created", ""))), reverse=True)
    return [r for _, r in hits[: max(1, limit)]]


def add_memory(path: Path, text: str, kind: str = DEFAULT_KIND) -> dict:
    """Store one memory and return the record.

    Raises MemoryStoreError, in Chinese, when the text is blank or over
    MAX_MEMORY_CHARS, or when the store is already at MAX_MEMORIES.
    """
    body = (text or "").strip()
    if not body:
        raise MemoryStoreError("要记住的内容是空的。")
    if len(body) > MAX_MEMORY_CHARS:
        raise MemoryStoreError(
            f"这条记忆有 {len(body)} 字，超过 {MAX_MEMORY_CHARS} 字上限。"
            f"记忆是一句话，不是一篇文章——请只留最关键的那句。"
        )
    chosen = kind if kind in KINDS else DEFAULT_KIND
    with _LOCK:
        records = _read_store(path)
        # Exact duplicates collapse. An 8B model repeats itself, and ten copies of
        # the same preference would crowd out nine real ones in every recall.
        needle = body.casefold()
        for existing in reversed(records):
            if str(existing.get("text", "")).strip().casefold() == needle:
                return existing
        if len(records) >= MAX_MEMORIES:
            raise MemoryStoreError(
                f"记忆已满（{MAX_MEMORIES} 条）。请在左下角的「记忆」面板里删掉不再需要的条目。"
            )
        record = {
            "id": new_memory_id(),
            "kind": chosen,
            "text": body,
            "created": _now(),
        }
        _atomic_write_json(path, [*records, record])
    return record


def delete_memory(path: Path, memory_id: str) -> bool:
    """True when a memory was removed. A malformed id is refused outright."""
    if not is_memory_id(memory_id):
        raise MemoryStoreError("记忆编号不合法。")
    with _LOCK:
        records = _read_store(path)
        kept = [r for r in records if r.get("id") != memory_id]
        if len(kept) == len(records):
            return False
        _atomic_write_json(path, kept)
    return True


def clear_memories(path: Path) -> int:
    """Drop every memory; returns how many there were. Deleting the file is enough —
    _read_store treats a missing file as an empty store."""
    with _LOCK:
        count = len(_read_store(path))
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise MemoryStoreError(f"无法清空记忆文件：{exc}") from exc
    return count
