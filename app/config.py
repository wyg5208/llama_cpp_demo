from __future__ import annotations

import logging
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .mcp import fs_entry_script
from .settings_store import read_overrides

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# --- logging -------------------------------------------------------------------
# A closed set rather than logging.getLevelName(), which answers an unknown name with
# the string "Level 7" -- a typo would then configure a level nobody asked for and say
# nothing about it.
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
# 1 MB active plus 3 backups. Sized against what this app writes: one line per chat
# request, so the cap holds tens of thousands of turns. What would blow straight
# through it is uvicorn's access log -- see setup_logging for why that stays on the
# console only.
APP_LOG_MAX_BYTES = 1_000_000
APP_LOG_BACKUPS = 3
# Marks the handlers setup_logging owns, so calling it twice replaces them instead of
# stacking a second copy of every line.
_HANDLER_TAG = "llama_cpp_demo_handler"


DEFAULT_SYSTEM_PROMPT = (
    "你是一个乐于助人的中文 AI 助手。回答准确、简洁、有条理，必要时使用 Markdown 排版。\n"
    "涉及实时信息、具体数据、新闻、版本发布、价格或你不确定的事实时，"
    "先调用 web_search 工具检索，再基于检索结果作答，并在正文中标注来源编号（如 [1]）。\n"
    "web_search 的 query 只能是 1~3 个关键词，不要把用户的问题原样传进去。\n"
    "检索结果通常会附上最相关页面的正文节选，优先据此作答并标注来源编号。\n"
    "若节选不足以回答，或你想看另一条结果的原文，再用 fetch_url 读取那个链接；"
    "不要用相近的关键词反复检索。\n"
    "闲聊、常识、写作、翻译、代码等不依赖实时信息的问题直接回答，不要检索。\n"
    "用户上传图片时，先客观描述图中内容，再回答用户的问题。"
    # The sentence that used to end this prompt — 「用户想要文件时，直接输出完整的
    # Markdown 正文，并提示他点消息右上角的「导出」按钮…」 — now lives in
    # tools.PROMPT_LINES["export_hint"] and is appended only while 生成文档 is OFF.
    # Unconditional here, it contradicted that box: the model got two instructions for
    # the same trigger and obeyed this one, telling the user 「由于我无法直接“生成文件”
    # 并让您下载…点消息右上角的「导出」按钮」 with save_document in its own tool list.
    # Measured over three rounds, 20 real runs on gemma-4-E4B-it: 4 called the tool, and
    # every one of those documents arrived intact — the longest 2,745 characters using
    # 1,484 of the 2,048 generation tokens, so the long escaped JSON argument was never
    # the risk this feature was rejected for. The other 16 wrote the document as prose
    # instead, or explained that the requested length was impossible. Exactly one of the
    # two lines is present per request now; .env.example carries the whole record.
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- llama.cpp runtime -------------------------------------------------
    # "external" reuses a llama-server you started yourself.
    runtime_backend: str = "vulkan"
    llama_server_url: str = ""
    llama_port: int = 8081
    n_gpu_layers: int = 99
    # Fallback context for any model not named in n_ctx_overrides.
    n_ctx: int = 16384
    # Per-model context as "stem=32768,other-stem=131072", keyed by GGUF file
    # stem. What a model can take is bounded by both its trained ceiling and its
    # weight size, so no single number serves all of them. Kept a str rather than
    # dict[str, int] because pydantic-settings demands JSON for complex fields.
    n_ctx_overrides: str = ""
    server_extra_args: str = ""
    server_startup_timeout: int = 300

    # --- model -------------------------------------------------------------
    model_path: Path = Path("F:/models/gguf/gemma-4-E4B-it-Q4_K_M.gguf")
    mmproj_path: Path = Field(default=Path("F:/models/gguf/mmproj-gemma-4-E4B-it-Q8_0.gguf"))
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # --- generation --------------------------------------------------------
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    # Several short searches per question (plus one retry) is the intended pattern.
    max_tool_rounds: int = 4
    # Reasoning models (Gemma 4, Qwen3) spend tokens on reasoning_content before
    # answering; left on, that can exhaust max_tokens and yield an empty reply.
    enable_thinking: bool = False
    # The chain of thought shares the completion budget with the answer.
    thinking_max_tokens: int = 4096

    # --- web search --------------------------------------------------------
    # auto: bocha > tavily > bing, picking the first one that is usable.
    search_provider: str = "auto"
    search_max_results: int = 5
    bocha_api_key: str = ""
    tavily_api_key: str = ""

    # --- MCP (filesystem server over stdio) --------------------------------
    # Optional in the strongest sense: with no Node.js on the machine the app runs
    # exactly as before, and the file checkbox greys out.
    mcp_enabled: bool = True
    # Comma-separated allowed roots; empty means this project's directory (see
    # fs_roots). A str for the same reason as n_ctx_overrides — pydantic-settings
    # demands JSON for a list, and a comma-separated path list is what a human types.
    # Pointing this somewhere narrower than the project is the single most effective
    # safety measure available: the default root contains .env.
    mcp_fs_roots: str = ""
    # Escape hatch for app.mcp.find_node(), whose shutil.which level normally hits.
    mcp_node_path: str = ""
    # Cold start measured 6.1 s (first Node module load off disk), warm 0.197 s.
    mcp_startup_timeout: int = 30

    # --- native memory -----------------------------------------------------
    # runtime/memory.json, reached through the remember / recall tools only.
    memory_enabled: bool = True

    # --- web app -----------------------------------------------------------
    host: str = "127.0.0.1"
    # 8000 is commonly taken by ComfyUI on this machine.
    port: int = 8123

    # --- logging -----------------------------------------------------------
    # Console plus runtime/app.log. Kept out of settings_store.EDITABLE on purpose:
    # the handlers are built once at startup, so a level changed from the settings
    # panel would look saved and do nothing until the next restart.
    log_level: str = "INFO"

    @field_validator("runtime_backend")
    @classmethod
    def _check_backend(cls, value: str) -> str:
        allowed = {"vulkan", "cuda", "cpu", "external"}
        if value not in allowed:
            raise ValueError(f"runtime_backend must be one of {sorted(allowed)}")
        return value

    @field_validator("n_ctx_overrides")
    @classmethod
    def _check_ctx_overrides(cls, value: str) -> str:
        # A malformed entry must fail here: silently skipping it would leave that
        # model running at the fallback context with nothing to say so.
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            stem, sep, ctx = item.rpartition("=")
            if not sep or not stem.strip() or not (ctx.strip().isdigit() and int(ctx) > 0):
                raise ValueError(f"N_CTX_OVERRIDES 每项应形如 模型名=正整数，收到: {item!r}")
        return value

    @field_validator("mcp_fs_roots")
    @classmethod
    def _check_fs_roots(cls, value: str) -> str:
        # A root that does not exist must fail here. Skipping it would hand the MCP
        # server a shorter argv than the user wrote, and the model would then be
        # refused access to a directory nobody can see was ever dropped.
        for item in value.split(","):
            item = item.strip()
            if item and not Path(item).is_dir():
                raise ValueError(f"MCP_FS_ROOTS 里的目录不存在: {item!r}")
        return value

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, value: str) -> str:
        # Normalised here so setup_logging can use the value directly, and rejected
        # rather than defaulted: a LOG_LEVEL typo that quietly left the app at INFO
        # would look like "DEBUG produced nothing" instead of like a misspelling.
        upper = value.strip().upper()
        if upper not in LOG_LEVELS:
            raise ValueError(f"LOG_LEVEL 应为 {list(LOG_LEVELS)} 之一，收到: {value!r}")
        return upper

    @property
    def uses_external_server(self) -> bool:
        return self.runtime_backend == "external" or bool(self.llama_server_url)

    @property
    def base_url(self) -> str:
        if self.llama_server_url:
            return self.llama_server_url.rstrip("/")
        return f"http://127.0.0.1:{self.llama_port}"

    @property
    def server_exe(self) -> Path | None:
        """Locate llama-server.exe inside the extracted runtime directory."""
        runtime_dir = ROOT / "runtime" / f"llama-{self.runtime_backend}"
        if not runtime_dir.is_dir():
            return None
        return next(iter(runtime_dir.rglob("llama-server.exe")), None)

    @property
    def server_log(self) -> Path:
        """llama-server's own stdout and stderr, redirected by runtime.py.

        Opened with "w", so it holds the current run only: a restart destroys the
        previous one. That is deliberate -- _tail_log() quotes it back in a 502 when
        the server refuses new weights, where stale lines would mislead -- but it does
        mean generation counts from an earlier run are gone. runtime/app.log is the
        durable one.
        """
        return ROOT / "runtime" / "llama-server.log"

    @property
    def app_log(self) -> Path:
        """This application's log. Rotating, UTF-8, and it survives restarts."""
        return ROOT / "runtime" / "app.log"

    @property
    def model_dir(self) -> Path:
        """Directory the model picker scans. Derived, not separately configured."""
        return self.model_path.parent

    @property
    def active_model_file(self) -> Path:
        return ROOT / "runtime" / "active_model.json"

    @property
    def settings_override_file(self) -> Path:
        """Settings the panel changed at runtime. Wins over .env; delete to reset."""
        return ROOT / "runtime" / "settings_override.json"

    @property
    def history_dir(self) -> Path:
        """Chat sessions: one index plus one JSON file per conversation."""
        return ROOT / "runtime" / "history"

    @property
    def mcp_dir(self) -> Path:
        """Where scripts/fetch_mcp.py npm-installs the filesystem server.

        Under runtime/, which .gitignore already covers — the install measured 31 MB
        across 4,026 files, and none of it belongs in version control.
        """
        return ROOT / "runtime" / "mcp"

    @property
    def mcp_entry(self) -> Path:
        """The server's entry script, and the file whose absence means "not installed".

        Derived by app.mcp rather than here because scripts/fetch_mcp.py needs the
        same path and cannot import the settings layer to get it.
        """
        return fs_entry_script(self.mcp_dir)

    @property
    def memory_file(self) -> Path:
        """The model's long-term memory. One file; delete it to start over."""
        return ROOT / "runtime" / "memory.json"

    @property
    def fs_roots(self) -> list[Path]:
        """Directories the MCP server is allowed to see. Derived, not hardcoded.

        Empty MCP_FS_ROOTS means this project, which is what makes app.tools' deny
        list load-bearing: .env is right here in the root, and the server was measured
        to serve it.
        """
        roots = [Path(item.strip()) for item in self.mcp_fs_roots.split(",") if item.strip()]
        return roots or [ROOT]

    def n_ctx_for(self, model_path: Path) -> int:
        """Context to request for this model; unlisted models fall back to n_ctx."""
        for item in self.n_ctx_overrides.split(","):
            stem, sep, ctx = item.strip().rpartition("=")
            if sep and stem.strip() == model_path.stem:
                return int(ctx)
        return self.n_ctx

    def resolved_search_provider(self) -> str:
        if self.search_provider != "auto":
            return self.search_provider
        if self.bocha_api_key:
            return "bocha"
        if self.tavily_api_key:
            return "tavily"
        return "bing"


@lru_cache
def get_settings() -> Settings:
    """Settings from .env with runtime/settings_override.json layered on top.

    Never call cache_clear() on this. LlamaRuntime keeps the returned instance
    (runtime.py:28) and builds every llama-server argument through it, so a
    second Settings would silently desync the runtime from /api/status and from
    the browser. Resetting means setattr back to a fresh Settings' values, which
    is what DELETE /api/settings does.
    """
    base = Settings()
    overrides = read_overrides(base.settings_override_file)
    if not overrides:
        return base
    try:
        return Settings(**overrides)
    except ValueError as exc:
        # read_overrides already validated these through SettingsPatch, so this
        # needs a hand-edited file that satisfies the patch model but not
        # Settings. Falling back keeps the app bootable: an override file is
        # subordinate to .env, so .env cannot be the thing that repairs it.
        log.warning("ignoring settings override that failed to apply: %s", exc)
        return base


def setup_logging(settings: Settings) -> None:
    """Console always; runtime/app.log whenever it can be opened. Safe to call twice.

    LOG_LEVEL governs this package's own loggers and uvicorn's, not third-party
    libraries' — the body below says why that distinction is the whole design.

    Called from run.py before uvicorn starts and again from app.main's lifespan, so the
    shipped entry point and a bare `uvicorn app.main:app` both get the file. The second
    call replaces the first one's handlers instead of adding to them: two copies of
    every line is the failure mode the tag exists to prevent, and it is invisible in a
    console scrollback but obvious in a file.
    """
    # getLevelName is bidirectional, and log_level came through _check_log_level, so
    # this is one of the five integers and never the "Level X" it returns for a name it
    # does not know.
    level = logging.getLevelName(settings.log_level)
    root = logging.getLogger()
    # A floor of WARNING on root rather than LOG_LEVEL. Root's level is the effective
    # level of every third-party logger that does not set its own, so LOG_LEVEL=DEBUG on
    # root turns on httpcore's per-socket-frame tracing too: measured on a real start,
    # six HTTP requests wrote ~80 httpcore lines into app.log, and this app streams every
    # token from llama-server through httpx, so a single chat turn at DEBUG would rotate
    # away everything worth keeping. max() keeps it monotone — asking for ERROR or
    # CRITICAL still narrows root, only DEBUG and INFO stop at the floor.
    #
    # Root's level does not filter records that propagate INTO it; only handler levels
    # do. So the "app" and uvicorn loggers set below still reach these handlers at
    # whatever LOG_LEVEL asked for.
    root.setLevel(max(level, logging.WARNING))
    # All fifteen modules in this package log to "app" or "app.<module>", so this one
    # logger's level covers every line the app itself writes.
    logging.getLogger("app").setLevel(level)

    for handler in [h for h in root.handlers if getattr(h, _HANDLER_TAG, False)]:
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    setattr(console, _HANDLER_TAG, True)
    root.addHandler(console)

    try:
        settings.app_log.parent.mkdir(parents=True, exist_ok=True)
        file_handler: logging.Handler = RotatingFileHandler(
            settings.app_log,
            maxBytes=APP_LOG_MAX_BYTES,
            backupCount=APP_LOG_BACKUPS,
            encoding="utf-8",
        )
    except OSError as exc:
        # A read-only runtime/ or a full disk must not stop the app. The console
        # handler is already attached, so nothing is lost but the file, and saying so
        # once here beats a "--- Logging error ---" traceback on every later record.
        # Rollover itself can still fail on Windows if a second process holds app.log
        # open -- renaming an open file is refused there -- but the launcher kills old
        # instances before starting a new one, and the failure is noisy, not fatal.
        log.warning("日志文件 %s 打不开，本次只输出到控制台: %s", settings.app_log, exc)
        return
    file_handler.setFormatter(formatter)
    setattr(file_handler, _HANDLER_TAG, True)
    root.addHandler(file_handler)

    # uvicorn 0.52 runs configure_logging() in Config.__init__, which is before it
    # imports this app, and gives "uvicorn" its own handler with propagate=False. Its
    # records would therefore never reach the root logger and never reach the file.
    # Rerouted rather than handed the file handler directly: a shared handler on a
    # logger that still propagates writes every line twice.
    #
    # "uvicorn.access" is deliberately left alone. The browser polls /api/stats every
    # 2 s (static/app.js, STATS_MS=2000), so an access log in the file is ~43,000
    # lines a day at ~100 bytes each -- about 4 MB, which is the whole rotation budget
    # spent daily on telemetry nobody reads back. Startup and error lines from
    # "uvicorn" and "uvicorn.error" are rare and worth keeping.
    for name in ("uvicorn", "uvicorn.error"):
        named = logging.getLogger(name)
        for handler in list(named.handlers):
            named.removeHandler(handler)
        named.propagate = True
        # Their own level, not just the root's: uvicorn sets both to INFO, which would
        # filter a DEBUG record out before the handler LOG_LEVEL=DEBUG just installed
        # ever saw it — and the WARNING floor on root above would filter it out too if
        # these inherited rather than carried a level of their own.
        named.setLevel(level)
