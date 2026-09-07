from __future__ import annotations

import json
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders

from . import history
from .config import ROOT, Settings, get_settings
from .documents import (
    DOC_MAX_CHARS,
    MAX_DOC_BYTES,
    DocumentError,
    budget_safety,
    check_filename,
    fit_budget,
    parse_document,
)
from .export import (
    MAX_EXPORT_BLOCKS,
    MAX_EXPORT_HTML_CHARS,
    ExportError,
    build as build_export,
    check_blocks,
)
from .llm import ChatMessage, stream_chat
from .models import find_model, save_active_model, scan_models
from .runtime import LlamaRuntime, RuntimeUnavailable
from .search import TOOLS
from .settings_store import (
    EDITABLE,
    SettingsPatch,
    clear_overrides,
    describe,
    merge_overrides,
    read_overrides,
)
from .sysstats import SystemStats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("app")

DATA_URL_RE = re.compile(r"^data:image/(?:png|jpeg|jpg|webp|gif);base64,[A-Za-z0-9+/=\s]+$")
MAX_IMAGE_BYTES = 12_000_000
MAX_IMAGES_PER_MESSAGE = 4
# Lower than the image cap on purpose: an image costs one encoder pass, while a
# document body costs thousands of prompt tokens on every turn it stays in the
# history, so the context budget binds long before the upload size does.
MAX_DOCS_PER_MESSAGE = 3
# The wire cap is the tighter one: llama-server pays for every message in prompt
# tokens. The archive is looser so a long conversation keeps the turns the wire has
# stopped sending instead of being silently truncated on disk.
MAX_CHAT_MESSAGES = 200
MAX_SESSION_MESSAGES = 400

runtime: LlamaRuntime | None = None
# Built once so the NVML handle stays open; re-initialising the driver per
# request would be the expensive part of an otherwise free endpoint.
stats = SystemStats()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global runtime
    settings = get_settings()
    runtime = LlamaRuntime(settings)
    try:
        await runtime.start()
        log.info("llama-server ready: %s", json.dumps(runtime.status(), ensure_ascii=False))
    except RuntimeUnavailable as exc:
        # Keep serving the UI so the browser can show what went wrong.
        runtime.state = "error"
        runtime.detail = str(exc)
        log.error("%s", exc)
    try:
        yield
    finally:
        await runtime.stop()


class RevalidateAssets:
    """Send `Cache-Control: no-cache` with the document and the static assets.

    StaticFiles emits Last-Modified and ETag but no Cache-Control, so Chrome
    falls back to heuristic freshness — roughly 10% of the file's age. An app.js
    cached a day ago therefore counts as fresh for hours, and any edit ships as
    new HTML driving old JS. `no-cache` keeps the cheap 304s while forcing a
    check on every load.

    Hand-written ASGI rather than @app.middleware("http"): BaseHTTPMiddleware
    buffers the response body, which would break the SSE stream on /api/chat.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        path = scope.get("path", "")
        if scope["type"] != "http" or not (path == "/" or path.startswith("/static/")):
            await self.app(scope, receive, send)
            return

        async def send_with_header(message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["Cache-Control"] = "no-cache"
            await send(message)

        await self.app(scope, receive, send_with_header)


app = FastAPI(title="llama.cpp 本地聊天", lifespan=lifespan)
app.add_middleware(RevalidateAssets)


class DocumentIn(BaseModel):
    """An attachment's extracted text, replayed by the browser on every turn.

    The body round-trips through the client instead of living in a server-side
    upload store: there is then nothing to expire or orphan, and a session
    restored from the archive still carries the text it was analysed with. Same
    trade-off as base64 images, already accepted for the archive.
    """

    name: str = Field(default="", max_length=200)
    text: str = Field(default="", max_length=DOC_MAX_CHARS)


class MessageIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = ""
    images: list[str] = Field(default_factory=list, max_length=MAX_IMAGES_PER_MESSAGE)
    documents: list[DocumentIn] = Field(default_factory=list, max_length=MAX_DOCS_PER_MESSAGE)

    @field_validator("images")
    @classmethod
    def _check_images(cls, value: list[str]) -> list[str]:
        for url in value:
            if not DATA_URL_RE.match(url):
                raise ValueError("图片必须是 png/jpeg/webp/gif 的 base64 data URL")
            if len(url) > MAX_IMAGE_BYTES:
                raise ValueError("单张图片过大（上限约 12MB）")
        return value


class ChatRequest(BaseModel):
    messages: list[MessageIn] = Field(min_length=1, max_length=MAX_CHAT_MESSAGES)
    web_search: bool = False
    thinking: bool = False


class ModelSwitchIn(BaseModel):
    # An id from GET /api/models, never a path: nothing client-supplied is
    # concatenated into the --model argument.
    id: str = Field(min_length=1, max_length=200)


class StoredSource(BaseModel):
    index: int = 0
    title: str = ""
    # Restored sessions put this straight into an anchor href, so it becomes
    # client-writable here for the first time: hold it to the same http(s) rule
    # safeLink() applies to rendered markdown.
    url: str = ""

    @field_validator("url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        if value and not re.match(r"^https?://", value, re.I):
            raise ValueError("来源链接必须是 http 或 https 地址")
        return value


class StoredMessage(MessageIn):
    """An archived message: what goes to the model, plus what the UI reads back."""

    reasoning: str = ""
    sources: list[StoredSource] = Field(default_factory=list, max_length=50)
    usage: str = ""
    error: bool = False
    # Stored, not derived from len(images) — a conversation migrated out of
    # localStorage has the flag but not the pixels, and deriving it would erase the
    # "images not saved" note the first time that session is re-saved.
    had_images: bool = False


class SessionCreate(BaseModel):
    # Empty means "title it from the first user message".
    title: str = Field(default="", max_length=120)
    messages: list[StoredMessage] = Field(min_length=1, max_length=MAX_SESSION_MESSAGES)


class SessionPatch(BaseModel):
    """Partial by design: renaming from the sidebar must not ship the images back."""

    title: str | None = Field(default=None, max_length=120)
    messages: list[StoredMessage] | None = Field(default=None, max_length=MAX_SESSION_MESSAGES)


class ExportIn(BaseModel):
    """One export's worth of already-rendered HTML.

    A list of blocks rather than one HTML string, because the browser has the DOM
    and this side does not: renderMarkdown's output is a flat sequence of top-level
    blocks, so `el.children` already IS the block list, and re-deriving it here
    would mean parsing the HTML straight back into a tree. The PDF packer needs
    those boundaries to paginate. Splitting them server-side was tried and does not
    work — a regex splitter returned 6 blocks for 7 with every offset shifted one
    tag, and html.parser's getpos() is a line/column pair, not a byte offset.

    Only pdf and docx arrive here. md, html and csv never leave the browser: it
    already holds the markdown source and the rendered DOM, so those three cost no
    round trip at all.
    """

    # A Literal so an unknown format is Pydantic's 422 before the handler runs;
    # the route needs no dispatch table of its own.
    format: Literal["pdf", "docx"]
    blocks: list[str] = Field(min_length=1, max_length=MAX_EXPORT_BLOCKS)
    # Decorative only: PDF document metadata. The filename belongs to the browser,
    # which is why the endpoint returns bare bytes and there is no
    # Content-Disposition and no RFC 5987 encoding anywhere in this file.
    title: str = Field(default="", max_length=120)


def sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/api/status")
async def status() -> JSONResponse:
    assert runtime is not None
    return JSONResponse(runtime.status())


@app.get("/api/stats")
async def sys_stats() -> JSONResponse:
    # Inline rather than offloaded to a thread pool: a full poll is ~0.2 ms, so
    # the hop would cost more than the work. Fields are null, never absent, when
    # a source is unavailable (no NVIDIA driver, non-Windows).
    return JSONResponse(stats.read())


@app.get("/api/models")
async def list_models() -> JSONResponse:
    assert runtime is not None
    model_dir = get_settings().model_dir
    return JSONResponse(
        {
            "dir": str(model_dir),
            "active": runtime.model_path.stem,
            "can_switch": runtime.can_switch,
            "models": [e.to_dict() for e in scan_models(model_dir)],
        }
    )


@app.post("/api/model")
async def switch_model(req: ModelSwitchIn) -> JSONResponse:
    assert runtime is not None
    if not runtime.can_switch:
        raise HTTPException(400, "外部 llama-server 模式下无法切换模型")
    if runtime.state == "switching":
        raise HTTPException(409, "正在切换模型，请稍候")
    model_dir = get_settings().model_dir
    entry = find_model(model_dir, req.id)
    if entry is None:
        raise HTTPException(404, f"模型不存在或已被移动：{req.id[:80]}")
    try:
        await runtime.switch_model(entry.model_path, entry.mmproj_path)
    except RuntimeUnavailable as exc:
        # 502: llama-server refused the new weights. The message says whether we
        # rolled back, and carries the server log tail that explains why.
        raise HTTPException(502, str(exc)[:1200]) from exc
    save_active_model(get_settings().active_model_file, entry)
    return JSONResponse(runtime.status())


@app.get("/api/settings")
async def settings_get() -> JSONResponse:
    s = get_settings()
    return JSONResponse(describe(s, read_overrides(s.settings_override_file)))


@app.patch("/api/settings")
async def settings_patch(req: SettingsPatch) -> JSONResponse:
    # Await-free on purpose, and the file write stays inline for the same reason:
    # asyncio only hands control to another coroutine at an await, so this runs
    # to completion before a live stream_chat can read a half-applied set of
    # values. Offloading a few hundred bytes to the thread pool would trade that
    # guarantee for nothing.
    #
    # exclude_none rather than exclude_unset: {"temperature": null} is unset-or-
    # null indistinguishable from "leave it alone", and writing null through
    # setattr would put None where llm.py expects a float.
    patch = req.model_dump(exclude_none=True)
    if not patch:
        raise HTTPException(400, "没有需要更新的内容")
    s = get_settings()
    for name, value in patch.items():
        # Settings is not frozen and /api/chat calls get_settings() per request,
        # so this is live from the next message onwards. SettingsPatch is the
        # only validation: setattr does not coerce or range-check.
        setattr(s, name, value)
    persisted = merge_overrides(s.settings_override_file, patch)
    body = describe(s, read_overrides(s.settings_override_file))
    # False means a read-only runtime/: the change is live but will not survive a
    # restart, and the browser says so rather than showing a silent lie.
    body["persisted"] = persisted
    return JSONResponse(body)


@app.delete("/api/settings")
async def settings_reset() -> JSONResponse:
    s = get_settings()
    # Not get_settings.cache_clear(): LlamaRuntime stored this exact instance
    # (runtime.py:28) and builds every llama-server argument through it, so a
    # second Settings would desync the runtime from what the browser shows.
    # Copying a fresh one's values back keeps a single authoritative object.
    fresh = Settings()
    for name in EDITABLE:
        setattr(s, name, getattr(fresh, name))
    cleared = clear_overrides(s.settings_override_file)
    body = describe(s, read_overrides(s.settings_override_file))
    body["persisted"] = cleared
    return JSONResponse(body)


def _history_stats(directory: Path) -> dict:
    """Sessions on disk, counted with a plain stat walk.

    Deliberately not history.list_sessions(): that prunes index.json when a body
    file is missing, so reading it from here would make merely opening the About
    panel mutate the session store.
    """
    count = 0
    total = 0
    try:
        for entry in directory.iterdir():
            # is_session_id also rejects the *.json.tmp of an interrupted write.
            if entry.suffix == ".json" and history.is_session_id(entry.stem) and entry.is_file():
                count += 1
                total += entry.stat().st_size
    except OSError:
        return {"count": 0, "bytes": 0, "readable": False}
    return {"count": count, "bytes": total, "readable": True}


@app.get("/api/about")
async def about() -> JSONResponse:
    assert runtime is not None
    s = get_settings()
    model = runtime.model_path
    try:
        model_bytes = model.stat().st_size
    except OSError:
        model_bytes = 0
    sessions = await run_in_threadpool(_history_stats, s.history_dir)
    # build_info comes from the /props payload the runtime already fetched, so
    # there is no subprocess: `llama-server --version` writes to stderr and the
    # executable is not necessarily ours to run in external mode.
    return JSONResponse(
        {
            "runtime": runtime.status(),
            "build_info": str(runtime.props.get("build_info") or ""),
            "external": s.uses_external_server,
            "model_path": str(model),
            "model_size_gb": round(model_bytes / 1024**3, 2),
            "mmproj_path": str(runtime.mmproj_path) if runtime.mmproj_path else "",
            "sessions": sessions,
        }
    )


def _checked_session_id(session_id: str) -> str:
    """Reject anything that is not an id minted by history.py, before it reaches a path."""
    if not history.is_session_id(session_id):
        # Logged because the browser should never send one, but answered as 404
        # rather than 400 so the route does not distinguish malformed from missing.
        log.warning("rejecting malformed session id: %r", session_id[:80])
        raise HTTPException(404, "会话不存在")
    return session_id


@app.get("/api/sessions")
async def sessions_index() -> JSONResponse:
    # Through the thread pool even though the index is only a few KB: it takes the
    # same lock as the writes, and waiting on that inline would stall the event
    # loop for as long as a multi-megabyte session body takes to serialise.
    sessions = await run_in_threadpool(history.list_sessions, get_settings().history_dir)
    return JSONResponse({"sessions": sessions})


@app.post("/api/sessions")
async def session_create(req: SessionCreate) -> JSONResponse:
    assert runtime is not None
    try:
        record = await run_in_threadpool(
            history.create_session,
            get_settings().history_dir,
            [m.model_dump() for m in req.messages],
            runtime.model_path.stem,
            req.title,
        )
    except OSError as exc:
        raise HTTPException(500, f"保存对话失败：{exc}") from exc
    return JSONResponse(record)


@app.get("/api/sessions/{session_id}")
async def session_get(session_id: str) -> JSONResponse:
    _checked_session_id(session_id)
    body = await run_in_threadpool(history.read_session, get_settings().history_dir, session_id)
    if body is None:
        raise HTTPException(404, "会话不存在")
    return JSONResponse(body)


@app.patch("/api/sessions/{session_id}")
async def session_patch(session_id: str, req: SessionPatch) -> JSONResponse:
    _checked_session_id(session_id)
    # Built by hand rather than with model_dump(exclude_unset=True), which would
    # also drop unset defaults from the nested messages and archive them as absent.
    patch: dict = {}
    if req.title is not None:
        patch["title"] = req.title
    if req.messages is not None:
        patch["messages"] = [m.model_dump() for m in req.messages]
    if not patch:
        raise HTTPException(400, "没有需要更新的内容")
    try:
        record = await run_in_threadpool(
            history.update_session, get_settings().history_dir, session_id, patch
        )
    except OSError as exc:
        raise HTTPException(500, f"保存对话失败：{exc}") from exc
    if record is None:
        raise HTTPException(404, "会话不存在")
    return JSONResponse(record)


@app.delete("/api/sessions/{session_id}")
async def session_delete(session_id: str) -> JSONResponse:
    _checked_session_id(session_id)
    try:
        deleted = await run_in_threadpool(
            history.delete_session, get_settings().history_dir, session_id
        )
    except OSError as exc:
        raise HTTPException(500, f"删除对话失败：{exc}") from exc
    if not deleted:
        raise HTTPException(404, "会话不存在")
    return JSONResponse({"ok": True})


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)) -> JSONResponse:
    """Parse an uploaded file and hand the extracted text back to the browser."""
    name = file.filename or ""
    try:
        # On the name alone, before reading a byte: an .exe or a .doc costs
        # nothing to refuse and everything to buffer first.
        check_filename(name)
    except DocumentError as exc:
        raise HTTPException(400, str(exc)) from exc

    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > MAX_DOC_BYTES:
            raise HTTPException(
                413, f"文件超过 {MAX_DOC_BYTES // 1_000_000}MB 上限，已中止读取。"
            )
        chunks.append(chunk)

    try:
        # Off the event loop: parsing is blocking CPU work, and the PDF path runs
        # an ONNX layout inference per page (~0.3-0.5 s each, measured).
        parsed = await run_in_threadpool(parse_document, name, b"".join(chunks))
    except DocumentError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - only the type reaches the browser
        log.exception("document parse failed: %s", name[:200])
        raise HTTPException(500, f"解析失败：{type(exc).__name__}") from exc
    return JSONResponse(parsed.to_dict())


@app.post("/api/export")
async def export_file(req: ExportIn) -> Response:
    """Render already-produced HTML into a PDF or a DOCX. Never calls the model."""
    try:
        # Field(max_length=...) above already bounds this, and it is re-checked here
        # anyway for the same reason _is_sendable below re-checks the browser: a
        # Pydantic violation is a 422 with an English body, while every other
        # refusal in this app speaks Chinese.
        #
        # Note what is NOT here: /api/documents needs its manual mid-read 413
        # because it streams an UploadFile in chunks and must stop early. This
        # endpoint receives a JSON body Pydantic has already sized, so the limit is
        # declarative and there is nothing to abort.
        if sum(len(b) for b in req.blocks) > MAX_EXPORT_HTML_CHARS:
            raise ExportError(f"导出内容超过 {MAX_EXPORT_HTML_CHARS // 1000}k 字符上限。")
        blocks = check_blocks(req.blocks)
        # Off the event loop: both writers are blocking CPU work, and the PDF one
        # costs a 17 ms fit probe per guess (measured 300 blocks -> 0.58 s).
        out = await run_in_threadpool(build_export, req.format, blocks, req.title)
    except ExportError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - only the type reaches the browser
        log.exception("export failed: %s", req.format)
        raise HTTPException(500, f"导出失败：{type(exc).__name__}") from exc
    # Bare bytes, not a Content-Disposition download: the browser already needs a
    # Blob and an <a download> for md/html/csv, so one saveBlob() serves all five
    # formats instead of two download paths that could drift, and the Chinese
    # session title never has to survive RFC 5987 encoding.
    #
    # Headers stay ASCII because Starlette encodes them as latin-1 — the truncation
    # warning is a flag the client words itself, not a sentence sent over the wire.
    return Response(
        content=out.data,
        media_type=out.media_type,
        headers={
            "X-Export-Pages": str(out.pages),
            "X-Export-Truncated": "1" if out.truncated else "0",
        },
    )


def _is_sendable(m: MessageIn) -> bool:
    """Mirror of isSendable() in static/app.js — read the comment there first.

    A turn that produced no words must not go back on the wire: the model is handed
    […, user X, assistant "", user X] and answers with an immediate EOS, which the
    user experiences as the conversation having stopped working. Clamped here as
    well as in the browser for the same reason as web_search below — a second tab
    with stale state, or a direct API call, would otherwise still send it.

    Only the empty-content half of the rule can live here. `error` is not a field on
    MessageIn and wireMessages() never puts it on the wire, and recognising our own
    "请求失败：" prefix back out of the content would be brittle; so error turns are
    filtered client-side only. That still self-heals, because toStored persists
    `error` and fromStored restores it, so a resumed session filters them too.

    A user turn survives on attachments alone: send() refuses a text-less turn, but
    one restored from the archive may be pixels-only.
    """
    if m.role == "user":
        return bool(m.content.strip()) or bool(m.documents) or bool(m.images)
    return bool(m.content.strip())


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    assert runtime is not None
    if runtime.state not in ("ready", "external"):
        reason = (
            "正在切换模型，请稍候"
            if runtime.state == "switching"
            else f"模型未就绪: {runtime.detail}"
        )
        return StreamingResponse(
            iter([sse("error", {"text": reason})]),
            media_type="text/event-stream",
        )

    # This local `history` shadows the history module; the module is not used below.
    history = [
        ChatMessage(
            role=m.role,
            content=m.content,
            images=m.images,
            documents=[(d.name, d.text) for d in m.documents],
        )
        for m in req.messages
        if _is_sendable(m)
    ]

    # Search is implemented as a tool call, so a model whose template cannot call
    # tools cannot search. Clamping here rather than trusting the browser covers a
    # second tab with stale state and any direct API call.
    use_search = req.web_search and runtime.supports_tools
    if req.web_search and not use_search:
        log.warning("ignoring web_search: %s does not support tool calls", runtime.model_path.name)

    s = get_settings()
    # The server's own figure, not a local guess: /props reports what llama-server
    # actually allocated for the slot, which n_ctx_overrides only approximates.
    n_ctx = int(runtime.status().get("n_ctx") or s.n_ctx_for(runtime.model_path))
    reserve = s.thinking_max_tokens if req.thinking else s.max_tokens
    # use_search, not req.web_search: llama-server only receives the schemas when the
    # model can actually call them (llm.py:140-142), so charging a tool-less model for
    # 807 estimated tokens it will never see is waste — and use_search above has just
    # been clamped on runtime.supports_tools.
    if history:
        safety = budget_safety(
            s.system_prompt,
            json.dumps(TOOLS, ensure_ascii=False) if use_search else "",
            len(history),
        )
        fitted, notes = fit_budget(history, n_ctx, reserve, safety)
        if notes:
            log.info(
                "context trimmed %d -> %d messages: %s", len(history), len(fitted), " ".join(notes)
            )
    else:
        # Every turn was filtered out as empty. fit_budget([]) would return ([], [])
        # with no notes, so the `if not fitted` branch below would blame the context
        # window for what is really an empty conversation.
        fitted, notes = [], ["上一轮回复为空，本轮没有可发送的内容，请重新提问。"]

    async def events() -> AsyncIterator[str]:
        # One joined line rather than an event per note: the UI keeps a single
        # status slot, so separate events would overwrite each other and only the
        # last would ever be read. Gated on `fitted` because the notes describe
        # trimming that happened; when nothing will be sent at all, the error event
        # below carries the same text and emitting both shows it to the user twice.
        if notes and fitted:
            yield sse("status", {"text": " ".join(notes)})
        if not fitted:
            # Either even the last message will not fit, or every turn was filtered
            # out as empty. notes[0] is the reason computed for exactly that case.
            yield sse("error", {"text": notes[0] if notes else "上下文长度不足，请新建对话。"})
            return
        try:
            async for event in stream_chat(
                get_settings(), fitted, use_search, req.thinking
            ):
                yield sse(event["type"], event)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the browser
            log.exception("chat failed")
            yield sse("error", {"text": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")
