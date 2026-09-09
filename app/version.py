"""The app's own version, read out of README.md rather than duplicated in code.

README.md's `**版本 / Version**: v...` line is the single source of truth for
versioning in this project (the four places a release touches are listed in its
section 8). A second literal here would need a fifth sync on every release, and
the two could disagree silently -- the About panel would then show a version that
no document claims. Reading the file keeps the documented number and the displayed
number the same object.

Deliberately not a Settings field: the version is not user-configurable, so it has
no business in .env, in .env.example, or in settings_store.SettingsPatch (whose
allowlist is a promise about what may change while the app runs).

Takes a plain Path and imports nothing from this package, so a test can point it at
a README in a temp directory.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

# The format of that line is a contract, pinned by tests/test_version.py: changing it
# here without changing README.md (or the other way round) yields a None and the panel
# quietly says "未知". Both files are edited in the same commit as part of a release.
VERSION_RE = re.compile(r"\*\*版本 / Version\*\*:\s*v(\d+\.\d+\.\d+)")

# (path, mtime_ns, size) -> version. Keyed on the file's identity so an edit shows up
# without a restart, and so a temp-directory README can never be served from the cache
# an earlier test filled. /api/about answers on the event loop thread, which is why a
# plain tuple swap needs no lock.
_cache: tuple[tuple[str, int, int], str | None] | None = None


def read_version(readme: Path = README) -> str | None:
    """`"1.1.0"` for a README saying `v1.1.0`, or None when it cannot be told.

    Never raises. A deployment that ships `app/` and `static/` without README.md, and a
    README whose version line was edited into an unrecognisable shape, both answer None
    so the panel can say "未知" -- the alternative is a 500 on the one endpoint the About
    pane needs to draw anything at all.
    """
    global _cache
    try:
        stat = readme.stat()
    except OSError:
        _log_unreadable(readme)
        return None
    key = (str(readme), stat.st_mtime_ns, stat.st_size)
    if _cache is not None and _cache[0] == key:
        return _cache[1]

    version: str | None
    try:
        match = VERSION_RE.search(readme.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        # Covers the file vanishing or being replaced between stat and read, and a
        # README saved in an encoding that is not UTF-8.
        _log_unreadable(readme)
        version = None
    else:
        if match:
            version = match.group(1)
        else:
            _log_unreadable(readme, "no line matching '**版本 / Version**: vX.Y.Z'")
            version = None
    _cache = (key, version)
    return version


def _log_unreadable(readme: Path, why: str = "unreadable") -> None:
    # Once per file change, not once per request: the result goes into the cache
    # alongside None, so a permanently missing README costs this one line.
    log.warning("cannot determine the app version from %s (%s)", readme, why)
