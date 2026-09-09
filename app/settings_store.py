"""Runtime-mutable settings: what the browser may change, and where it is kept.

The full set of settings lives in config.py's Settings; the eight enumerated by
SettingsPatch below are the only ones safe to change while it runs — the
ones read per request when building the generation payload. This module is the
single source of truth for that split: SettingsPatch is both the request
body type and the allowlist for the on-disk file, so the two cannot drift.

Overrides live in runtime/settings_override.json and win over .env. They are
applied by constructing Settings(**overrides) once in get_settings(), which
re-runs the field validators, and thereafter by setattr on the cached instance,
which works because Settings is not frozen and /api/chat re-reads it per request.

Takes plain Paths and imports nothing from this package at runtime, so it can be
driven straight from a test against a temp directory. Everything is synchronous;
the callers in main.py keep their write handlers await-free instead, which is a
stronger guarantee than a thread hop (see the note there).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # config.py imports this module, so the reverse edge is types only
    from .config import Settings

log = logging.getLogger(__name__)


class SettingsPatch(BaseModel):
    """The eight settings that take effect on the next message, no restart needed."""

    # forbid, not ignore: Settings itself ignores extras, so an unknown key here
    # would be written to the override file and rebound on the next boot while the
    # panel insists the field is read-only.
    model_config = ConfigDict(extra="forbid")

    system_prompt: str | None = Field(default=None, max_length=4000)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    # Shares one budget with thinking_max_tokens; both capped well under the
    # smallest per-model context (40960) so a typo cannot exceed the window.
    max_tokens: int | None = Field(default=None, ge=64, le=32768)
    thinking_max_tokens: int | None = Field(default=None, ge=64, le=32768)
    # Each round is a full model call, so this is a cost ceiling, not a tweak.
    max_tool_rounds: int | None = Field(default=None, ge=0, le=8)
    search_provider: Literal["auto", "bing", "bocha", "tavily"] | None = None
    search_max_results: int | None = Field(default=None, ge=1, le=10)


EDITABLE = frozenset(SettingsPatch.model_fields)

SECRET_FIELDS = frozenset({"bocha_api_key", "tavily_api_key"})

# Why the other 21 are read-only. Shown verbatim under the greyed-out value.
_SERVER_REASON = "需改 .env 并重启 llama-server"
_APP_REASON = "需改 .env 并重启应用"
REASONS = {
    # Everything build_args() turns into a llama-server command line.
    "runtime_backend": _SERVER_REASON,
    "llama_server_url": _SERVER_REASON,
    "llama_port": _SERVER_REASON,
    "n_gpu_layers": _SERVER_REASON,
    "n_ctx": _SERVER_REASON,
    "n_ctx_overrides": _SERVER_REASON,
    "server_extra_args": _SERVER_REASON,
    "server_startup_timeout": _SERVER_REASON,
    # Read once when the app builds its runtime, or when uvicorn binds.
    "host": _APP_REASON,
    "port": _APP_REASON,
    "enable_thinking": _APP_REASON,
    # None of these join EDITABLE, and not for want of a reason: SettingsPatch
    # promises "takes effect on the next message, no restart", while changing an
    # allowed root means starting a new MCP child process with a different argv.
    # Putting a restart-required field in a no-restart whitelist would be a lie.
    "mcp_enabled": _APP_REASON,
    "mcp_fs_roots": _APP_REASON + "；留空 = 本项目目录",
    "mcp_node_path": _APP_REASON + "；留空 = 自动发现",
    "mcp_startup_timeout": _APP_REASON,
    "memory_enabled": _APP_REASON,
    # Read once too, when setup_logging builds the handlers -- the same reason it
    # stays out of EDITABLE, and without an entry here describe() would render the
    # greyed-out value with nothing under it to say why.
    "log_level": _APP_REASON,
    # switch_model() moves the runtime, not Settings: what is shown here is the
    # boot default, and pointing the user at the dropdown is more useful than
    # telling them to restart.
    "model_path": "运行时切换请用顶栏的模型下拉框；这里是 .env 中的启动默认值",
    "mmproj_path": "随模型自动配对，见顶栏的模型下拉框",
    "bocha_api_key": "密钥只在 .env 中维护，不会显示或回传完整值",
    "tavily_api_key": "密钥只在 .env 中维护，不会显示或回传完整值",
}

# Serialises the read-modify-write of the override file. run.py starts uvicorn
# without workers, so one process holds one lock and that is enough.
_LOCK = threading.Lock()


def _mask(value: str) -> str:
    if not value:
        return "未配置"
    # A short key has no safe tail to show: value[-4:] of a 3-character string is
    # the whole string.
    return f"已配置 ····{value[-4:]}" if len(value) >= 8 else "已配置"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _atomic_write_json(path: Path, payload) -> bool:
    """Replace `path` whole, or leave the previous version untouched.

    Returns False when the disk refused — the opposite of history.py, which
    raises. The new value is already live in memory by the time this runs, so a
    read-only runtime/ must not turn an applied change into a reported failure;
    only the next restart would lose it, and the caller says so.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            os.replace(tmp, path)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
    except OSError as exc:
        log.warning("could not write %s: %s", path, exc)
        return False
    return True


def read_overrides(path: Path) -> dict:
    """Stored overrides, filtered to EDITABLE and re-validated. {} when unusable.

    Called from get_settings(), so this is on the boot path: a damaged or
    hand-edited file has to degrade to "no overrides". Never raise from here.
    """
    data = _read_json(path)
    if data is None:
        if path.is_file():
            log.warning("ignoring unreadable settings override %s", path)
        return {}
    if not isinstance(data, dict):
        log.warning("ignoring malformed settings override %s", path)
        return {}
    known = {k: v for k, v in data.items() if k in EDITABLE}
    unknown = sorted(set(data) - set(known))
    if unknown:
        log.warning("ignoring unknown keys in %s: %s", path.name, ", ".join(unknown))
    try:
        # One bad value discards the whole file rather than half-applying it: a
        # panel that shows six of seven saved changes is worse than one that
        # shows none and says why in the log.
        return SettingsPatch(**known).model_dump(exclude_none=True)
    except ValueError as exc:
        log.warning("ignoring invalid settings override %s: %s", path.name, exc)
        return {}


def merge_overrides(path: Path, patch: dict) -> bool:
    with _LOCK:
        stored = _read_json(path)
        stored = {k: v for k, v in stored.items() if k in EDITABLE} if isinstance(stored, dict) else {}
        stored.update(patch)
        return _atomic_write_json(path, stored)


def clear_overrides(path: Path) -> bool:
    with _LOCK:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("could not delete %s: %s", path, exc)
            return False
    return True


def describe(settings: Settings, overrides: dict) -> dict:
    """Every setting with its live value, for the panel to render.

    mode="json" is load-bearing: model_path and mmproj_path are WindowsPath,
    which JSONResponse cannot serialise.
    """
    fields = []
    for name, value in settings.model_dump(mode="json").items():
        if name in SECRET_FIELDS:
            shown = _mask(value)
        elif name not in EDITABLE and value == "":
            # llama_server_url and server_extra_args are legitimately empty, and
            # a blank row reads as "failed to load" rather than "not set".
            shown = "未配置"
        else:
            # Editable fields must carry the real value: the panel puts it
            # straight into an input, and substituting text there would save it.
            shown = value
        fields.append(
            {
                "name": name,
                "value": shown,
                "editable": name in EDITABLE,
                "overridden": name in overrides,
                "reason": REASONS.get(name, ""),
            }
        )
    return {"fields": fields, "overridden": sorted(overrides)}
