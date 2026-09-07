"""MCP over stdio, with the standard library only.

No `mcp` package: the whole protocol this app needs is newline-delimited JSON-RPC
over a child process's stdin/stdout, which `asyncio.create_subprocess_exec` already
speaks. Measured against @modelcontextprotocol/server-filesystem 2026.8.31 —
handshake, tools/list, tools/call, clean exit 0 on a closed stdin — and every
constant below is a number from that run rather than a guess.

Like documents.py, history.py and settings_store.py, this module imports nothing
from the rest of the package at runtime, so it stays drivable from a test on a
machine with no Node.js installed. `Settings` is a TYPE_CHECKING edge only.

Two properties of the wire format decide the design, and both were found by
getting them wrong first:

- Responses arrive OUT OF ORDER. Four tools/call fired back to back without an
  intervening read came home as ids [3, 2, 4, 5]. A client that sends one request
  and reads one line therefore answers callers with each other's results — which
  is how two perfectly healthy tools first appeared to return isError.
- One JSON-RPC line can be enormous. directory_tree over this project's root
  produced a single 2,805,068-byte line, and asyncio's StreamReader defaults to a
  64 KiB limit: past that, readline() clears its buffer and raises, killing the
  connection rather than merely returning a big result.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import shutil
import string
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # config.py sits above this module, so the edge is types only
    from .config import Settings

log = logging.getLogger(__name__)


class MCPError(Exception):
    """A failure whose message is safe to show the user verbatim.

    Same contract as DocumentError and ExportError: Chinese, actionable, and never
    carrying a path or a payload the browser should not see.
    """


# Measured single response: 2,805,068 bytes for directory_tree over this project's
# root, which contains .venv/ and runtime/llama-vulkan/. 32 MiB is ~11x that. Going
# over it is not a truncated result — readline() clears the buffer and raises, the
# reader task dies, and the stream is desynced from then on. The host treats that as
# a dead client and spawns a fresh one on the next call, so it self-heals.
_STREAM_LIMIT = 32 * 1024 * 1024

# search_files took 0.919 s and directory_tree 0.182 s on a cold-ish tree. 60 s is
# headroom for a pathological directory, not an expectation.
_CALL_TIMEOUT = 60.0

# runtime.py:169 uses the same figure for llama-server.
_STOP_TIMEOUT = 10.0

# What the server answered with, so asking for it is not a negotiation.
_PROTOCOL_VERSION = "2025-06-18"

CLIENT_NAME = "llama-cpp-demo"
CLIENT_VERSION = "1.0"

_STDERR_TAIL_LINES = 8

_NODE_NAMES = ("node.exe", "node") if sys.platform == "win32" else ("node",)

# The npm package, and the file whose presence means "installed". Both live here
# rather than in config.py because scripts/fetch_mcp.py needs the same answer and
# must not import the settings layer to get it: two independent derivations of one
# path drift, and that drift surfaces as the app saying "run fetch_mcp.py" while
# fetch_mcp.py says "already installed".
FS_PACKAGE = "@modelcontextprotocol/server-filesystem"

# Measured against this exact version: 14 tools, the annotations that tier them
# read-only vs writable, the deprecated read_file alias, protocolVersion
# 2025-06-18. fetch_mcp.py pins it for the same reason.
FS_PACKAGE_VERSION = "2026.8.31"


def fs_entry_script(mcp_dir: Path) -> Path:
    """The filesystem server's entry script under an npm prefix directory."""
    return mcp_dir / "node_modules" / FS_PACKAGE / "dist" / "index.js"


def _child_cwd() -> Path:
    """Where the Node child runs. NOT a security layer — see the note below.

    This used to claim it was the second of two layers against a relative path
    argument, on the theory that the server resolves one against its own cwd and so
    a cwd outside every allowed root would make the server refuse it. Read out of the
    installed source, that theory is wrong twice over:

    - dist/lib.js resolveRelativePathAgainstAllowedDirectories resolves a relative
      argument against each allowed root in turn, and falls back to process.cwd() only
      when `allowedDirectories.length === 0`.
    - dist/index.js refuses to operate with zero roots — usage-and-exit on an empty
      argv, `process.exit(1)` when none are accessible, a throw at startup when the
      client supplies none either. So the cwd branch is unreachable in a server that
      is serving at all.

    It stays for two honest reasons. Spawning a child in the project directory is a
    poor default in itself, whatever this server does today. And if upstream ever
    changes that resolution, a cwd outside the roots is strictly better than one
    inside them. The temp directory rather than runtime/mcp for the same reason as
    before: runtime/mcp sits INSIDE the default root, so ".." from it climbs into
    runtime/history.

    The only thing standing between the model and this project's .env is
    tools.is_denied, which refuses every non-absolute path outright. That is one
    layer, not two, and tests/test_mcp.py pins the server behaviour that makes one
    layer necessary.
    """
    return Path(tempfile.gettempdir())


@lru_cache
def find_node(explicit: str = "") -> Path | None:
    """Locate node.exe, or None. Never raises.

    MCP is optional, so its absence must not stop the app booting; every caller
    turns None into "this capability is unavailable" rather than an exception.

    Four levels, each earning its place:

    1. `explicit` (MCP_NODE_PATH) — the escape hatch, and the only way to pin a
       version once more than one Node is installed.
    2. shutil.which — the normal path. start_app.bat opens a fresh cmd window, so
       it sees PATH as installed.
    3. the registry PATH — for a shell opened BEFORE Node was installed, which
       carries a snapshot of PATH that no longer matches the machine. This is not
       hypothetical: Node sits in HKLM's Path here while the shell running the app
       was started earlier, and a child process inherits the stale copy, so level 2
       fails and this one is what finds it.
    4. the usual install locations, across every mounted drive.

    Cached because MCPHost.available() answers on every /api/status poll (15 s from
    the browser) and level 4 is a drive scan. The cost is that a Node installed
    after boot needs a restart — which MCP_NODE_PATH needs anyway. Tests call
    find_node.cache_clear(); production code never does.
    """
    if explicit:
        cand = Path(explicit)
        if cand.is_file():
            return cand
        # Warn and keep looking rather than fail: an override that points nowhere is
        # a mistake, and silently using a different node would hide it, but refusing
        # to find the one that is installed would be worse.
        log.warning("MCP_NODE_PATH does not exist, falling back to discovery: %s", explicit)

    found = shutil.which("node")
    if found:
        return Path(found)

    for directory in (*_registry_path_dirs(), *_fallback_dirs()):
        for name in _NODE_NAMES:
            cand = directory / name
            if cand.is_file():
                return cand
    return None


def _registry_path_dirs() -> list[Path]:
    """Directories named in the machine and user PATH, as the registry holds them.

    Read from the registry rather than os.environ on purpose: a process inherits its
    environment at spawn time, so os.environ["PATH"] is a snapshot that can be years
    — or one npm installer — out of date.
    """
    if sys.platform != "win32":
        return []
    import winreg  # Windows only; imported here so this module loads anywhere

    hives = (
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
        (winreg.HKEY_CURRENT_USER, r"Environment"),
    )
    dirs: list[Path] = []
    for hive, subkey in hives:
        try:
            with winreg.OpenKey(hive, subkey) as key:
                value, _ = winreg.QueryValueEx(key, "Path")
        except OSError:
            continue
        for part in str(value).split(os.pathsep):
            # REG_EXPAND_SZ entries carry %SystemRoot% and friends unexpanded.
            part = os.path.expandvars(part.strip()).strip('"')
            if part:
                dirs.append(Path(part))
    return dirs


def _fallback_dirs() -> list[Path]:
    """Where node.exe lives when nothing has told us. Last resort, so it may scan."""
    dirs: list[Path] = []
    if sys.platform != "win32":
        return [Path(p) for p in ("/usr/local/bin", "/usr/bin", "/opt/homebrew/bin")]

    home = Path(os.environ.get("USERPROFILE") or Path.home())
    for letter in string.ascii_uppercase:
        root = Path(f"{letter}:\\")
        if not root.exists():
            continue
        dirs.append(root / "Program Files" / "nodejs")
        dirs.append(root / "Program Files (x86)" / "nodejs")
    dirs.append(Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "chocolatey" / "bin")
    dirs.append(home / "scoop" / "apps" / "nodejs" / "current")
    dirs.append(home / "scoop" / "shims")
    nvm = Path(os.environ.get("APPDATA") or (home / "AppData" / "Roaming")) / "nvm"
    if nvm.is_dir():
        # Newest first: nvm keeps every version side by side.
        dirs.extend(sorted(nvm.glob("v*"), reverse=True))
    return dirs


class MCPClient:
    """One Node child process, and the id-routing that makes it safe to talk to.

    Concurrency is deliberate. Because responses are matched to callers by id rather
    than by arrival order, several requests may be in flight at once; `-np 1` makes
    chat effectively serial but two browser tabs can still reach /api/chat together.
    MCPHost guards *starting* with a lock, not *calling*.
    """

    def __init__(self, node: Path, entry: Path, roots: Sequence[Path]) -> None:
        self.node = node
        self.entry = entry
        # From MCP_FS_ROOTS, validated as existing directories by Settings and never
        # taken from a request body — these become the server's argv, and argv is the
        # only thing bounding what it will read.
        self.roots = [Path(r) for r in roots]
        # Not a parameter: a boundary a caller can pass in is a boundary a caller can
        # pass wrong, and there is no second caller to serve. See _child_cwd.
        self.cwd = _child_cwd()
        self.proc: asyncio.subprocess.Process | None = None
        self.server_info: dict = {}
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=_STDERR_TAIL_LINES)

    @property
    def alive(self) -> bool:
        """Running AND still able to answer — a dead reader means every call would
        sit out its full timeout, which is indistinguishable from a hung server."""
        return (
            self.proc is not None
            and self.proc.returncode is None
            and self._reader_task is not None
            and not self._reader_task.done()
        )

    async def start(self, timeout: float) -> None:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.proc = await asyncio.create_subprocess_exec(
            str(self.node),
            str(self.entry),
            *[str(r) for r in self.roots],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # Without this, the first legitimate directory_tree over a big root kills
            # the connection instead of returning. See _STREAM_LIMIT.
            limit=_STREAM_LIMIT,
            creationflags=flags,
            # Not inherited. See self.cwd — this is what stops a relative argument
            # resolving onto the project's own .env.
            cwd=str(self.cwd),
        )
        self._reader_task = asyncio.create_task(self._reader(), name="mcp-reader")
        self._stderr_task = asyncio.create_task(self._stderr_pump(), name="mcp-stderr")
        try:
            await asyncio.wait_for(self._initialize(), timeout)
        except (MCPError, asyncio.TimeoutError) as exc:
            detail = self.stderr_tail()
            await self.stop()
            raise MCPError(
                f"MCP 服务器启动失败：{'超时' if isinstance(exc, asyncio.TimeoutError) else exc}"
                + (f"\n{detail}" if detail else "")
            ) from exc

    async def _initialize(self) -> None:
        result = await self.call(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
            },
        )
        info = result.get("serverInfo")
        self.server_info = info if isinstance(info, dict) else {}
        await self._notify("notifications/initialized")

    async def _reader(self) -> None:
        """The only consumer of stdout. Routes by id; drops notifications."""
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break  # EOF: the server closed its end, or exited
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    # Framing is intact, this one payload is not. Dropping it is safe;
                    # its caller times out with a message that names the method.
                    log.debug("mcp: dropped %d unparseable bytes", len(line))
                    continue
                if not isinstance(msg, dict):
                    continue
                mid = msg.get("id")
                if mid is None:
                    log.debug("mcp notification: %s", msg.get("method") or "?")
                    continue
                fut = self._pending.pop(mid, None)
                if fut is not None and not fut.done():
                    fut.set_result(msg)
        except (asyncio.LimitOverrunError, ValueError) as exc:
            # readline() raises ValueError after clearing its buffer when a line
            # exceeds `limit`, so the byte stream is desynced and cannot be resumed.
            log.warning("mcp reader stopped, connection unusable: %s", exc)
        except asyncio.CancelledError:
            raise
        finally:
            # Without this every in-flight caller waits out its full 60 s timeout for
            # a server that is already gone.
            self._fail_pending("MCP 连接已断开")

    async def _stderr_pump(self) -> None:
        """Keep the child's stderr drained and remember its tail.

        Nobody else reads it, and an unread pipe fills: a server that logs enough
        would block on its own stderr mid-response, which looks exactly like a hang.
        """
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    return
                text = line.decode("utf-8", "replace").rstrip()
                if text:
                    self._stderr_tail.append(text)
                    log.debug("mcp server: %s", text)
        except (asyncio.LimitOverrunError, ValueError, OSError):
            return
        except asyncio.CancelledError:
            raise

    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail)

    def _fail_pending(self, reason: str) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(MCPError(reason))
        self._pending.clear()

    async def call(self, method: str, params: dict | None = None, timeout: float = _CALL_TIMEOUT) -> dict:
        """One request/response pair, matched by id. Returns the `result` object."""
        proc = self.proc
        if proc is None or proc.stdin is None or proc.returncode is not None:
            raise MCPError("MCP 服务器未运行")
        if self._reader_task is None or self._reader_task.done():
            # Fail fast: no reader means no reply is coming, and the alternative is
            # making the caller wait out the whole timeout to learn that.
            raise MCPError(f"MCP 连接不可用，无法调用 {method}")

        self._next_id += 1
        mid = self._next_id
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[mid] = fut
        payload = json.dumps(
            {"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}},
            ensure_ascii=False,
        )
        try:
            proc.stdin.write(payload.encode("utf-8") + b"\n")
            await proc.stdin.drain()
        except (OSError, ConnectionResetError) as exc:
            self._pending.pop(mid, None)
            raise MCPError(f"无法与 MCP 服务器通信：{type(exc).__name__}") from exc

        try:
            msg = await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            self._pending.pop(mid, None)
            raise MCPError(f"MCP 调用 {method} 在 {timeout:.0f}s 内没有响应") from None
        except asyncio.CancelledError:
            self._pending.pop(mid, None)
            raise
        error = msg.get("error")
        if error:
            text = error.get("message") if isinstance(error, dict) else str(error)
            raise MCPError(f"MCP {method} 失败：{text or error}")
        result = msg.get("result")
        return result if isinstance(result, dict) else {}

    async def _notify(self, method: str) -> None:
        """A message with no id, which the protocol says must not be answered."""
        proc = self.proc
        if proc is None or proc.stdin is None:
            return
        payload = json.dumps({"jsonrpc": "2.0", "method": method}, ensure_ascii=False)
        try:
            proc.stdin.write(payload.encode("utf-8") + b"\n")
            await proc.stdin.drain()
        except (OSError, ConnectionResetError) as exc:
            log.warning("mcp: could not send %s: %s", method, exc)

    async def list_tools(self) -> list[dict]:
        """Raw tool objects, `annotations` included — tools.py tiers on those."""
        tools = (await self.call("tools/list", {})).get("tools")
        return [t for t in tools if isinstance(t, dict)] if isinstance(tools, list) else []

    async def call_tool(self, name: str, args: dict | None = None) -> str:
        """Run one tool and return its text. An isError result comes back AS TEXT.

        "Access denied - path outside allowed directories: …" is precisely what the
        model needs to read in order to change course. Raising here would turn a
        recoverable wrong guess into a failed request.
        """
        result = await self.call("tools/call", {"name": name, "arguments": args or {}})
        content = result.get("content")
        parts = []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
        text = "\n".join(p for p in parts if p)
        return text or "（工具没有返回文本内容）"

    def _cancel(self, task: asyncio.Task | None) -> None:
        if task is not None and not task.done():
            task.cancel()

    async def stop(self) -> None:
        """Close stdin and wait; kill only if that is not enough.

        Same shape as runtime.py:161-175. Measured: the filesystem server exits 0 on
        a closed stdin, so the graceful path is the normal one.
        """
        proc, self.proc = self.proc, None
        if proc is None:
            self._cancel(self._reader_task)
            self._cancel(self._stderr_task)
            self._reader_task = self._stderr_task = None
            self._fail_pending("MCP 服务器已停止")
            return
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=_STOP_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        # Cancelling after wait() lets both pumps see a natural EOF first, so the
        # stderr tail still holds whatever the server said on its way out.
        self._cancel(self._reader_task)
        self._cancel(self._stderr_task)
        self._reader_task = self._stderr_task = None
        self._fail_pending("MCP 服务器已停止")


class MCPHost:
    """The app's one MCP server: discovered at boot, started on first use.

    Lazy rather than eager because a warm start measured 0.197 s against a
    multi-second local model call — invisible at the moment it is actually needed,
    while starting in lifespan would charge every user who never ticks the box 0.2 s
    plus a resident Node process.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.state = "stopped"
        self.detail = ""
        self.client: MCPClient | None = None
        self._lock = asyncio.Lock()
        self._tools: list[dict] | None = None

    def available(self) -> tuple[bool, str]:
        """Can MCP work? Answers without spawning anything.

        /api/status calls this on every poll, so it must stay cheap: find_node is
        cached and the entry check is one stat. The second element is a Chinese
        reason suitable for a tooltip — the same shape LlamaRuntime.status() uses.
        """
        if not self.settings.mcp_enabled:
            return False, "已在 .env 中关闭（MCP_ENABLED=false）"
        if find_node(self.settings.mcp_node_path) is None:
            return False, "未找到 Node.js：安装后重启应用，或在 .env 里设 MCP_NODE_PATH"
        if not self.settings.mcp_entry.is_file():
            return False, "未安装 filesystem MCP：先运行 python scripts/fetch_mcp.py"
        return True, ""

    async def ensure_started(self) -> MCPClient:
        ok, why = self.available()
        if not ok:
            self.state, self.detail = "error", why
            raise MCPError(why)
        async with self._lock:
            if self.client is not None and self.client.alive:
                return self.client
            if self.client is not None:
                # Died between calls — most likely a line over _STREAM_LIMIT, which
                # desyncs the stream. Replace it rather than report a dead client.
                log.info("replacing a dead MCP client")
                await self.client.stop()
                self.client = None
                self._tools = None
            client = MCPClient(
                find_node(self.settings.mcp_node_path),
                self.settings.mcp_entry,
                self.settings.fs_roots,
            )
            self.state, self.detail = "starting", ""
            try:
                await client.start(self.settings.mcp_startup_timeout)
            except MCPError as exc:
                self.state, self.detail = "error", str(exc)
                raise
            self.client = client
            self._tools = None
            self.state = "ready"
            info = client.server_info
            self.detail = " ".join(str(info.get(k, "")) for k in ("name", "version")).strip()
            log.info("mcp ready: %s", self.detail)
            return client

    async def tools(self) -> list[dict]:
        """Cached tools/list.

        Two concurrent first-callers may both list. That costs one wasted ~13 KB
        round trip and is not a correctness problem; serialising it would need a
        second lock, because asyncio.Lock is not reentrant and ensure_started()
        already holds this one.
        """
        if self._tools is None:
            client = await self.ensure_started()
            listed = await client.list_tools()
            if not listed:
                raise MCPError("MCP 服务器没有提供任何工具")
            self._tools = listed
        return self._tools

    async def stop(self) -> None:
        async with self._lock:
            client, self.client = self.client, None
            self._tools = None
            if client is not None:
                await client.stop()
            self.state, self.detail = "stopped", ""
