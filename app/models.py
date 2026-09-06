"""Local GGUF catalogue: scan a directory, pair mmproj files, remember the choice.

Takes plain Paths and imports nothing from this package, so it can be used from
both config-time and request-time code without a cycle.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

MMPROJ_PREFIX = "mmproj-"
# Trailing quantizer token (-Q4_K_M, -Q8_0, -F16, -IQ4_XS). IQ must precede Q.
QUANT_RE = re.compile(r"-(?:IQ|Q|F)\w*$")


@dataclass(frozen=True)
class ModelEntry:
    id: str
    name: str
    model_path: Path
    mmproj_path: Path | None
    size: int

    @property
    def vision(self) -> bool:
        return self.mmproj_path is not None

    def to_dict(self) -> dict:
        # Paths never leave the server; the browser only ever sends back an id.
        return {"id": self.id, "name": self.name, "size": self.size, "vision": self.vision}


def _base(stem: str) -> str:
    return QUANT_RE.sub("", stem)


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _pair(base: str, mmprojs: list[Path]) -> Path | None:
    """Most specific mmproj whose base is a '-'-delimited prefix of `base`.

    The delimiter check matters: plain startswith would let a generic
    mmproj-Qwen3-F16.gguf attach itself to every Qwen3-* model in the folder.
    Ties go to the longest base, then the largest file (a bigger projector is
    usually the higher-fidelity one).
    """
    scored = []
    for path in mmprojs:
        proj_base = _base(path.stem[len(MMPROJ_PREFIX):])
        if base == proj_base or base.startswith(proj_base + "-"):
            scored.append((len(proj_base), _size(path), path))
    return max(scored)[2] if scored else None


def scan_models(directory: Path) -> list[ModelEntry]:
    """Flat scan for *.gguf, sorted by name, each model paired with its mmproj."""
    try:
        found = sorted(directory.iterdir(), key=lambda p: p.name.lower())
    except OSError as exc:
        log.warning("cannot scan model dir %s: %s", directory, exc)
        return []
    gguf = [
        p
        for p in found
        if p.is_file() and p.suffix.lower() == ".gguf" and not p.name.startswith(".")
    ]
    mmprojs = [p for p in gguf if p.name.lower().startswith(MMPROJ_PREFIX)]
    return [
        ModelEntry(
            id=p.stem,
            name=p.stem,
            model_path=p,
            mmproj_path=_pair(_base(p.stem), mmprojs),
            size=_size(p),
        )
        for p in gguf
        if not p.name.lower().startswith(MMPROJ_PREFIX)
    ]


def find_model(directory: Path, model_id: str) -> ModelEntry | None:
    return next((e for e in scan_models(directory) if e.id == model_id), None)


def load_active_model(path: Path) -> tuple[Path, Path | None] | None:
    """The remembered choice, or None when it is absent, corrupt or stale."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        model = Path(data["model_path"])
        mmproj = Path(data["mmproj_path"]) if data.get("mmproj_path") else None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not model.is_file():
        log.info("ignoring %s: model no longer exists (%s)", path.name, model)
        return None
    if mmproj is not None and not mmproj.is_file():
        # Losing the projector costs vision; it is not worth discarding the model.
        log.info("ignoring mmproj from %s: %s", path.name, mmproj)
        mmproj = None
    return model, mmproj


def save_active_model(path: Path, entry: ModelEntry) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "id": entry.id,
                    "model_path": entry.model_path.as_posix(),
                    "mmproj_path": entry.mmproj_path.as_posix() if entry.mmproj_path else None,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        # A read-only runtime/ must not turn a successful switch into a failure.
        log.warning("could not remember the model choice: %s", exc)
