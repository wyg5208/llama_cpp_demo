"""The MCP client: id routing, Node discovery, and the real filesystem server.

Standard library only, by decision. Run from the project root:

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m unittest discover -s tests -v

Two groups, split by whether they need a machine that has Node.js and
`runtime/mcp` installed:

- Everything above `TestRealServer` runs anywhere. It drives MCPClient against a
  fake stdin/stdout pair, which is the only way to test the routing at all: the
  bug it pins (responses arriving out of order) needs four requests in flight and
  a chosen arrival order, and a real server will not reproduce that on demand.
- `TestRealServer` and `TestRealServerWrites` are `skipUnless(HAS_SERVER)`. They
  exist because the constants in app/mcp.py are measurements, not guesses, and a
  measurement that stops being true upstream has to fail a test rather than a
  user's request.

The guarded group is not decorative. It is what proved two things that the
unguarded group cannot: that a 2,805,068-byte directory_tree line survives
`_STREAM_LIMIT` (and dies without it — see the differential test), and that a
relative path argument is refused by the server's own sandbox once the child's
cwd sits outside every allowed root.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import string
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:  # annotations only, so the file stays importable on any Python
    from collections.abc import Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import mcp
from app.config import Settings
from app.mcp import (
    CLIENT_NAME,
    FS_PACKAGE_VERSION,
    MCPClient,
    MCPError,
    MCPHost,
    _child_cwd,
    find_node,
    fs_entry_script,
)
from app.tools import FS_DROP, _CONTENT_KEYS, _LOCATION_KEYS, _OPTION_KEYS, pick_fs_tools

NODE = find_node()
ENTRY = Settings(_env_file=None).mcp_entry
HAS_SERVER = NODE is not None and ENTRY.is_file()
SKIP = "needs Node.js plus runtime/mcp (python scripts/fetch_mcp.py)"

# The four project-root directory_tree numbers from the measurement in
# .env.example. Asserted as orders of magnitude rather than equalities, because
# the tree changes every time this repo does.
TREE_MIN_BYTES = 500_000


def _fresh_probe(statement: str) -> str:
    """Run `statement` in a child interpreter with a clean sys.modules.

    Same reason as test_export.py's: in-process, these assertions would depend on
    what some earlier test already imported.
    """
    done = subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT), check=True,
    )
    return done.stdout.strip()


def _response(mid: int, result) -> bytes:
    return json.dumps({"jsonrpc": "2.0", "id": mid, "result": result}).encode() + b"\n"


def _error(mid: int, message: str) -> bytes:
    payload = {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": message}}
    return json.dumps(payload).encode() + b"\n"


def _text_result(text: str, is_error: bool = False) -> bytes:
    """One tools/call result in the shape the filesystem server sends."""
    return _response(1, {"content": [{"type": "text", "text": text}], "isError": is_error})


class _Writer:
    """A subprocess stdin. `sent` is the point: it is what the payload tests read."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.sent: list[bytes] = []
        self.closed = False
        self._fail = fail

    def write(self, data: bytes) -> None:
        if self._fail is not None:
            raise self._fail
        self.sent.append(data)

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def requests(self) -> list[dict]:
        return [json.loads(chunk.decode("utf-8")) for chunk in self.sent]


class _Reader:
    """A subprocess stdout: a queue of lines, where b"" is EOF and an exception
    instance fed into the queue is raised by the next readline().

    Raising on demand rather than at construction is what makes two tests
    possible at all — a pending caller has to exist BEFORE the reader dies, or
    the death is observed as "fail fast" instead of "fail the pending future".
    """

    def __init__(self, lines: Iterable[bytes] = ()) -> None:
        self._queue: asyncio.Queue | None = None
        self._initial = list(lines)
        self.reads = 0

    def feed(self, item) -> None:
        assert self._queue is not None, "readline() has not been reached yet"
        self._queue.put_nowait(item)

    async def readline(self) -> bytes:
        if self._queue is None:
            self._queue = asyncio.Queue()
            for line in self._initial:
                self._queue.put_nowait(line)
        self.reads += 1
        item = await self._queue.get()
        if isinstance(item, BaseException):
            raise item
        return item


class _Proc:
    """The five attributes MCPClient touches on an asyncio.subprocess.Process."""

    def __init__(self, stdin=None, stdout=None, stderr=None, hang: bool = False) -> None:
        self.stdin, self.stdout, self.stderr = stdin, stdout, stderr
        self.returncode: int | None = None
        self.waits = 0
        self.kills = 0
        self._hang = hang

    async def wait(self) -> int:
        self.waits += 1
        # Hangs only until killed: stop() waits a second time after kill(), and a
        # fake that hung there too would make the kill-path test never return.
        if self._hang and not self.kills:
            await asyncio.sleep(3600)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self) -> None:
        self.kills += 1
        self.returncode = -9


def _wired(lines=(), stdin_fail=None) -> tuple[MCPClient, _Reader, _Writer]:
    """An MCPClient on fakes, with both pump tasks already running.

    Built by hand rather than through start(), because start() spawns a process
    and handshakes — and the thing under test here is what happens on the wire
    afterwards.
    """
    client = MCPClient(Path("node"), Path("index.js"), [Path("D:/allowed")])
    out = _Reader(lines)
    stdin = _Writer(stdin_fail)
    client.proc = _Proc(stdin, out, _Reader([b""]))
    client._reader_task = asyncio.create_task(client._reader(), name="test-reader")
    client._stderr_task = asyncio.create_task(client._stderr_pump(), name="test-stderr")
    return client, out, stdin


async def _settle() -> None:
    """Let every created task run to its first await.

    Two yields, not one: a task started this turn is only scheduled by the first,
    and `call` writes its payload before awaiting drain().
    """
    await asyncio.sleep(0)
    await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Module hygiene — the property that keeps every test above runnable offline
# ---------------------------------------------------------------------------


class TestModuleHygiene(unittest.TestCase):
    def test_importing_mcp_pulls_in_nothing_else_from_the_package(self):
        """config.py sits above this module, so the import edge has to stay TYPE_CHECKING.

        Pinned because a runtime `from .config import Settings` here would make
        every unguarded test in this file depend on the whole settings layer —
        including the .env on the machine running them.
        """
        self.assertEqual(
            _fresh_probe(
                "import sys, app.mcp;"
                "print(sorted(m for m in sys.modules"
                " if m.startswith('app.') and m != 'app.mcp'))"
            ),
            "[]",
        )

    def test_importing_mcp_does_not_look_for_node(self):
        """Discovery is a drive scan at its worst, so it belongs in available(),
        not at import time. `import app.main` was measured at 0.44 s offline and
        must stay there."""
        self.assertEqual(
            _fresh_probe("import app.mcp; print(app.mcp.find_node.cache_info().currsize)"),
            "0",
        )


class TestAvailableSpawnsNothing(unittest.TestCase):
    """/api/status calls MCPHost.available() on every 15 s poll.

    A capability check that started a Node child would make the status endpoint
    cost 0.197 s warm and 6.1 s cold, for a feature the user may never tick.
    """

    def setUp(self):
        find_node.cache_clear()
        self.addCleanup(find_node.cache_clear)

    def test_available_answers_without_starting_a_child(self):
        async def refuse(*args, **kwargs):
            raise AssertionError("available() tried to spawn a process")

        original = asyncio.create_subprocess_exec
        asyncio.create_subprocess_exec = refuse
        self.addCleanup(setattr, asyncio, "create_subprocess_exec", original)

        host = MCPHost(Settings(_env_file=None))
        ok, why = host.available()
        self.assertIs(ok, HAS_SERVER, why)
        self.assertIsNone(host.client)
        self.assertEqual(host.state, "stopped")


# ---------------------------------------------------------------------------
# Node discovery
# ---------------------------------------------------------------------------


class TestFindNode(unittest.TestCase):
    """Four levels, each with a reason to exist.

    Level 3 is the one that matters on this machine: `shutil.which("node")`
    returns None here because the shell running the tests carries a PATH snapshot
    from before Node was installed, while the registry already has it. A client
    that stopped at level 2 would report "未找到 Node.js" on a machine that has it.
    """

    def setUp(self):
        find_node.cache_clear()
        self.addCleanup(find_node.cache_clear)

    def _patch(self, target, name, value):
        original = getattr(target, name)
        setattr(target, name, value)
        self.addCleanup(setattr, target, name, original)

    def test_an_explicit_path_wins(self):
        with tempfile.TemporaryDirectory() as base:
            node = Path(base) / "node.exe"
            node.write_text("not really node")
            self.assertEqual(find_node(str(node)), node)

    def test_an_explicit_path_that_does_not_exist_falls_through(self):
        """Warn and keep looking, rather than refuse to find the Node that IS
        installed: MCP_NODE_PATH pointing nowhere is a mistake, but silently
        losing the whole capability over it would be a worse one."""
        found = find_node(str(Path(tempfile.gettempdir()) / "no-such-node.exe"))
        self.assertEqual(found, NODE)

    def test_which_beats_the_registry(self):
        with tempfile.TemporaryDirectory() as base:
            on_path = Path(base) / "node.exe"
            on_path.write_text("x")
            in_registry = Path(base) / "reg"
            in_registry.mkdir()
            (in_registry / "node.exe").write_text("x")
            self._patch(mcp.shutil, "which", lambda name: str(on_path))
            self._patch(mcp, "_registry_path_dirs", lambda: [in_registry])
            self.assertEqual(find_node(), on_path)

    def test_the_registry_beats_the_fallback_scan(self):
        with tempfile.TemporaryDirectory() as base:
            in_registry = Path(base) / "reg"
            in_registry.mkdir()
            (in_registry / "node.exe").write_text("x")
            in_fallback = Path(base) / "fb"
            in_fallback.mkdir()
            (in_fallback / "node.exe").write_text("x")
            self._patch(mcp.shutil, "which", lambda name: None)
            self._patch(mcp, "_registry_path_dirs", lambda: [in_registry])
            self._patch(mcp, "_fallback_dirs", lambda: [in_fallback])
            self.assertEqual(find_node(), in_registry / "node.exe")

    def test_the_fallback_scan_is_the_last_resort(self):
        with tempfile.TemporaryDirectory() as base:
            in_fallback = Path(base) / "fb"
            in_fallback.mkdir()
            (in_fallback / "node.exe").write_text("x")
            self._patch(mcp.shutil, "which", lambda name: None)
            self._patch(mcp, "_registry_path_dirs", lambda: [])
            self._patch(mcp, "_fallback_dirs", lambda: [in_fallback])
            self.assertEqual(find_node(), in_fallback / "node.exe")

    def test_nothing_found_returns_none_and_never_raises(self):
        """MCP is optional, so its absence must not stop the app booting."""
        self._patch(mcp.shutil, "which", lambda name: None)
        self._patch(mcp, "_registry_path_dirs", lambda: [])
        self._patch(mcp, "_fallback_dirs", lambda: [])
        self.assertIsNone(find_node())

    def test_a_directory_named_node_is_not_a_hit(self):
        with tempfile.TemporaryDirectory() as base:
            (Path(base) / "node.exe").mkdir()
            self._patch(mcp.shutil, "which", lambda name: None)
            self._patch(mcp, "_registry_path_dirs", lambda: [])
            self._patch(mcp, "_fallback_dirs", lambda: [Path(base)])
            self.assertIsNone(find_node())

    def test_the_result_is_cached(self):
        """available() answers every status poll, and level 4 is a drive scan."""
        calls = []

        def counting():
            calls.append(1)
            return []

        self._patch(mcp.shutil, "which", lambda name: None)
        self._patch(mcp, "_registry_path_dirs", counting)
        self._patch(mcp, "_fallback_dirs", lambda: [])
        find_node()
        find_node()
        find_node()
        self.assertEqual(len(calls), 1)

    def test_the_explicit_argument_is_part_of_the_cache_key(self):
        """Otherwise an override supplied after a cached miss would be ignored —
        the exact failure MCP_NODE_PATH exists to prevent."""
        with tempfile.TemporaryDirectory() as base:
            node = Path(base) / "node.exe"
            node.write_text("x")
            self._patch(mcp.shutil, "which", lambda name: None)
            self._patch(mcp, "_registry_path_dirs", lambda: [])
            self._patch(mcp, "_fallback_dirs", lambda: [])
            self.assertIsNone(find_node())
            self.assertEqual(find_node(str(node)), node)


class TestFallbackDirs(unittest.TestCase):
    def test_the_unix_branch_names_real_prefixes(self):
        if sys.platform == "win32":
            self.skipTest("POSIX branch")
        self.assertIn(Path("/usr/local/bin"), mcp._fallback_dirs())

    def test_registry_dirs_are_empty_off_windows(self):
        if sys.platform == "win32":
            self.skipTest("winreg branch")
        self.assertEqual(mcp._registry_path_dirs(), [])

    def test_on_windows_every_mounted_drive_is_considered(self):
        """Not a guess: Node lives on D: here, and a C:-only scan would miss it.
        Compared against the drives that actually exist rather than against a
        hardcoded pair, so the test does not depend on this machine's letters."""
        if sys.platform != "win32":
            self.skipTest("Windows only")
        mounted = {
            letter
            for letter in string.ascii_uppercase
            if Path(f"{letter}:\\").exists()
        }
        scanned = {
            str(directory)[0]
            for directory in mcp._fallback_dirs()
            if len(str(directory)) > 1 and str(directory)[1] == ":"
        }
        self.assertEqual(scanned, mounted)


# ---------------------------------------------------------------------------
# The child's working directory — hygiene, explicitly NOT a security layer
# ---------------------------------------------------------------------------


class TestChildCwd(unittest.TestCase):
    def test_it_is_outside_every_default_root(self):
        """Cheap insurance, not a defence. The server resolves a relative argument
        against its allowed roots and consults process.cwd() only when it has none
        (dist/index.js refuses to run in that state), so this changes nothing about
        what a relative path reaches today. It is still the right place to spawn a
        child, and it would matter immediately if upstream changed that resolution."""
        cwd = _child_cwd()
        for root in Settings(_env_file=None).fs_roots:
            self.assertNotIn(cwd, [root, *root.parents])
            self.assertFalse(str(cwd).lower().startswith(str(root).lower() + "\\"))

    def test_it_is_not_the_rejected_runtime_mcp(self):
        """runtime/mcp was the first candidate and was wrong: it sits INSIDE the
        default root, so ".." from it climbs to runtime/ and the conversation
        index is back in reach."""
        self.assertNotEqual(_child_cwd(), Settings(_env_file=None).mcp_dir)

    def test_it_exists_so_the_child_can_actually_start(self):
        self.assertTrue(_child_cwd().is_dir())

    def test_the_client_derives_it_rather_than_accepting_it(self):
        """A boundary a caller can pass in is a boundary a caller can pass wrong,
        and there is no second caller to serve."""
        parameters = list(inspect.signature(MCPClient.__init__).parameters)
        self.assertEqual(parameters, ["self", "node", "entry", "roots"])
        client = MCPClient(Path("node"), Path("index.js"), [Path("D:/allowed")])
        self.assertEqual(client.cwd, _child_cwd())


# ---------------------------------------------------------------------------
# Wire behaviour, against a fake transport
# ---------------------------------------------------------------------------


class TestRequestFraming(unittest.IsolatedAsyncioTestCase):
    async def test_a_request_is_one_newline_terminated_json_rpc_line(self):
        client, out, stdin = _wired()
        task = asyncio.create_task(client.call("tools/list", {}))
        await _settle()
        out.feed(_response(1, {"tools": []}))
        await task
        self.assertEqual(stdin.sent[0].count(b"\n"), 1, "one line, terminated")
        self.assertTrue(stdin.sent[0].endswith(b"\n"))
        request = stdin.requests()[0]
        self.assertEqual(request["jsonrpc"], "2.0")
        self.assertEqual(request["method"], "tools/list")
        self.assertEqual(request["params"], {})

    async def test_ids_increment_from_one(self):
        client, out, stdin = _wired()
        for expected in (1, 2, 3):
            task = asyncio.create_task(client.call("ping"))
            await _settle()
            out.feed(_response(expected, {}))
            await task
        self.assertEqual([r["id"] for r in stdin.requests()], [1, 2, 3])

    async def test_a_missing_params_argument_is_sent_as_an_empty_object(self):
        client, out, stdin = _wired()
        task = asyncio.create_task(client.call("ping"))
        await _settle()
        out.feed(_response(1, {}))
        await task
        self.assertEqual(stdin.requests()[0]["params"], {})

    async def test_a_notification_is_written_without_an_id(self):
        client, out, stdin = _wired()
        await client._notify("notifications/initialized")
        request = stdin.requests()[0]
        self.assertEqual(request["method"], "notifications/initialized")
        self.assertNotIn("id", request)

    async def test_a_dead_stdin_is_reported_not_swallowed(self):
        client, out, stdin = _wired(stdin_fail=ConnectionResetError())
        with self.assertRaises(MCPError) as caught:
            await client.call("ping")
        self.assertIn("ConnectionResetError", str(caught.exception))
        self.assertEqual(client._pending, {}, "the abandoned future must not linger")

    async def test_a_non_dict_result_comes_back_empty(self):
        client, out, stdin = _wired()
        task = asyncio.create_task(client.call("ping"))
        await _settle()
        out.feed(_response(1, "nonsense"))
        self.assertEqual(await task, {})


class TestIdRouting(unittest.IsolatedAsyncioTestCase):
    async def test_responses_arriving_out_of_order_reach_their_own_caller(self):
        """THE regression pin. Measured against the real server: four tools/call
        fired back to back without an intervening read came home as ids
        [3, 2, 4, 5]. A client that sends one request and reads one line answers
        callers with each other's results — which is how two perfectly healthy
        tools first appeared to return isError.
        """
        client, out, stdin = _wired()
        client._next_id = 1  # as if initialize had already taken id 1
        calls = [
            asyncio.create_task(client.call("tools/call", {"want": n}))
            for n in range(4)
        ]
        await _settle()
        self.assertEqual([r["id"] for r in stdin.requests()], [2, 3, 4, 5])

        for mid in (3, 2, 4, 5):  # the measured arrival order
            out.feed(_response(mid, {"echo": mid}))
        results = await asyncio.gather(*calls)

        self.assertEqual([r["echo"] for r in results], [2, 3, 4, 5])
        # The point of the test, stated against the payload each caller actually
        # wrote rather than against its id: an id-routing bug would still satisfy
        # `echo == id` if the ids themselves were handed out wrong.
        for request, result in zip(stdin.requests(), results):
            self.assertEqual(
                result["echo"], request["params"]["want"] + 2,
                f"caller {request['params']['want']} got somebody else's response",
            )

    async def test_a_notification_does_not_block_the_response_behind_it(self):
        client, out, stdin = _wired()
        task = asyncio.create_task(client.call("ping"))
        await _settle()
        out.feed(json.dumps({"jsonrpc": "2.0", "method": "notifications/progress"}).encode() + b"\n")
        out.feed(_response(1, {"ok": True}))
        self.assertEqual(await task, {"ok": True})

    async def test_an_unparseable_line_is_dropped_without_desyncing(self):
        """Framing is intact, this one payload is not. Its caller times out with
        a message naming the method; the next line still routes."""
        client, out, stdin = _wired()
        first = asyncio.create_task(client.call("tools/list"))
        second = asyncio.create_task(client.call("ping"))
        await _settle()
        out.feed(b"{ this is not json \n")
        out.feed(_response(2, {"second": True}))
        self.assertEqual(await second, {"second": True})
        first.cancel()

    async def test_a_response_that_is_not_an_object_is_ignored(self):
        client, out, stdin = _wired()
        task = asyncio.create_task(client.call("ping"))
        await _settle()
        out.feed(b"[1, 2, 3]\n")
        out.feed(_response(1, {"ok": True}))
        self.assertEqual(await task, {"ok": True})

    async def test_eof_fails_every_pending_caller(self):
        """Without this, every in-flight caller waits out its full 60 s timeout
        for a server that is already gone."""
        client, out, stdin = _wired()
        calls = [asyncio.create_task(client.call("ping")) for _ in range(3)]
        await _settle()
        out.feed(b"")
        for task in calls:
            with self.assertRaises(MCPError) as caught:
                await task
            self.assertIn("断开", str(caught.exception))
        self.assertEqual(client._pending, {})

    async def test_a_reader_that_dies_mid_line_fails_its_callers(self):
        """readline() raises ValueError AFTER clearing its buffer when a line
        exceeds `limit`, so the byte stream cannot be resumed. The reader must
        stop and say so rather than pretend the connection is usable."""
        client, out, stdin = _wired()
        task = asyncio.create_task(client.call("ping"))
        await _settle()
        out.feed(ValueError("Separator is not found, and chunk exceed the limit"))
        with self.assertRaises(MCPError):
            await task
        await _settle()
        self.assertTrue(client._reader_task.done())

    async def test_the_next_call_after_a_dead_reader_fails_fast(self):
        """No reader means no reply is coming, and the alternative is making the
        caller wait out the whole timeout to learn that."""
        client, out, stdin = _wired()
        task = asyncio.create_task(client.call("ping"))
        await _settle()
        out.feed(b"")
        with self.assertRaises(MCPError):
            await task
        with self.assertRaises(MCPError) as caught:
            await client.call("ping")
        self.assertIn("不可用", str(caught.exception))

    async def test_a_json_rpc_error_becomes_an_mcperror_carrying_the_message(self):
        client, out, stdin = _wired()
        task = asyncio.create_task(client.call("tools/call"))
        await _settle()
        out.feed(_error(1, "Method not found"))
        with self.assertRaises(MCPError) as caught:
            await task
        self.assertIn("Method not found", str(caught.exception))
        self.assertIn("tools/call", str(caught.exception))

    async def test_a_timeout_frees_the_slot(self):
        client, out, stdin = _wired()
        with self.assertRaises(MCPError) as caught:
            await client.call("directory_tree", timeout=0.05)
        self.assertIn("directory_tree", str(caught.exception))
        self.assertIn("没有响应", str(caught.exception))
        self.assertEqual(client._pending, {}, "a late response must not find a future")

    async def test_a_late_response_to_a_timed_out_call_is_discarded(self):
        client, out, stdin = _wired()
        with self.assertRaises(MCPError):
            await client.call("ping", timeout=0.05)
        out.feed(_response(1, {"too": "late"}))
        await _settle()
        self.assertEqual(client._pending, {})

    async def test_no_process_at_all_is_a_clear_refusal(self):
        client = MCPClient(Path("node"), Path("index.js"), [])
        with self.assertRaises(MCPError) as caught:
            await client.call("ping")
        self.assertIn("未运行", str(caught.exception))


class TestLiveness(unittest.IsolatedAsyncioTestCase):
    async def test_alive_needs_a_process_that_has_not_exited(self):
        client, out, stdin = _wired()
        self.assertTrue(client.alive)
        client.proc.returncode = 0
        self.assertFalse(client.alive)

    async def test_alive_needs_a_reader_that_can_still_answer(self):
        """A running process with a dead reader is worse than a dead process:
        every call would sit out its full timeout."""
        client, out, stdin = _wired()
        await _settle()  # the reader has to be parked on readline() before EOF means anything
        out.feed(b"")
        await _settle()
        await _settle()
        self.assertTrue(client._reader_task.done())
        self.assertFalse(client.alive, "returncode is still None, and that is the point")

    async def test_a_client_that_never_started_is_not_alive(self):
        self.assertFalse(MCPClient(Path("node"), Path("index.js"), []).alive)


class TestResultExtraction(unittest.IsolatedAsyncioTestCase):
    async def _call_tool(self, response: bytes, name="read_text_file", args=None):
        client, out, stdin = _wired()
        task = asyncio.create_task(client.call_tool(name, args))
        await _settle()
        out.feed(response)
        return await task

    async def test_text_blocks_are_joined(self):
        payload = _response(1, {"content": [
            {"type": "text", "text": "one"},
            {"type": "text", "text": "two"},
        ]})
        self.assertEqual(await self._call_tool(payload), "one\ntwo")

    async def test_non_text_blocks_are_skipped(self):
        payload = _response(1, {"content": [
            {"type": "image", "data": "…"},
            {"type": "text", "text": "kept"},
        ]})
        self.assertEqual(await self._call_tool(payload), "kept")

    async def test_an_is_error_result_comes_back_as_text_and_is_not_raised(self):
        """'Access denied - path outside allowed directories: …' is precisely what
        the model needs to read in order to change course. Raising here would turn
        a recoverable wrong guess into a failed request."""
        text = await self._call_tool(_text_result("Access denied - outside roots", True))
        self.assertEqual(text, "Access denied - outside roots")

    async def test_empty_content_says_so_rather_than_returning_a_blank(self):
        """A blank tool result reads to the model as success with no output."""
        self.assertEqual(await self._call_tool(_response(1, {"content": []})),
                         "（工具没有返回文本内容）")

    async def test_missing_content_is_handled_the_same_way(self):
        self.assertEqual(await self._call_tool(_response(1, {})), "（工具没有返回文本内容）")

    async def test_the_arguments_reach_the_wire_verbatim(self):
        client, out, stdin = _wired()
        task = asyncio.create_task(client.call_tool("read_text_file", {"path": "D:\\a.txt"}))
        await _settle()
        out.feed(_text_result("hi"))
        await task
        self.assertEqual(stdin.requests()[0]["params"],
                         {"name": "read_text_file", "arguments": {"path": "D:\\a.txt"}})

    async def test_list_tools_drops_non_dict_entries(self):
        client, out, stdin = _wired()
        task = asyncio.create_task(client.list_tools())
        await _settle()
        out.feed(_response(1, {"tools": [{"name": "a"}, "junk", None, {"name": "b"}]}))
        self.assertEqual(await task, [{"name": "a"}, {"name": "b"}])

    async def test_list_tools_survives_a_missing_or_misshapen_field(self):
        for result in ({}, {"tools": "nope"}, {"tools": None}):
            client, out, stdin = _wired()
            task = asyncio.create_task(client.list_tools())
            await _settle()
            out.feed(_response(1, result))
            self.assertEqual(await task, [], result)


class TestShutdown(unittest.IsolatedAsyncioTestCase):
    async def test_stopping_closes_stdin_and_waits(self):
        """Measured: the filesystem server exits 0 on a closed stdin, so the
        graceful path is the normal one and kill() is the exception."""
        client, out, stdin = _wired()
        proc = client.proc  # stop() drops the reference, so hold on to it
        await client.stop()
        self.assertTrue(stdin.closed, "a closed stdin is what tells the server to exit")
        self.assertEqual(proc.waits, 1)
        self.assertEqual(proc.kills, 0)
        self.assertIsNone(client.proc)
        self.assertIsNone(client._reader_task)
        self.assertIsNone(client._stderr_task)
        self.assertEqual(client._pending, {})

    async def test_stopping_kills_only_when_the_graceful_path_hangs(self):
        """kill() is the exception path. The reference is held here rather than
        read back off the client, because stop() deliberately drops it."""
        client = MCPClient(Path("node"), Path("index.js"), [])
        stdin = _Writer()
        proc = _Proc(stdin, _Reader([b""]), _Reader([b""]), hang=True)
        client.proc = proc
        original = mcp._STOP_TIMEOUT
        mcp._STOP_TIMEOUT = 0.05
        self.addCleanup(setattr, mcp, "_STOP_TIMEOUT", original)
        await client.stop()
        self.assertTrue(stdin.closed, "the graceful path is still tried first")
        self.assertEqual(proc.waits, 2, "waited once gracefully, once after the kill")
        self.assertEqual(proc.kills, 1)

    async def test_stopping_twice_is_not_an_error(self):
        """lifespan's finally calls this whether or not anything ever started."""
        client, out, stdin = _wired()
        await client.stop()
        await client.stop()

    async def test_stopping_never_started_client_cancels_the_pumps(self):
        client = MCPClient(Path("node"), Path("index.js"), [])
        await client.stop()
        self.assertIsNone(client._reader_task)

    async def test_a_pending_caller_is_told_when_the_client_stops(self):
        client, out, stdin = _wired()
        task = asyncio.create_task(client.call("ping", timeout=30))
        await _settle()
        await client.stop()
        with self.assertRaises(MCPError) as caught:
            await task
        self.assertIn("已停止", str(caught.exception))


class _StubSettings:
    """The three attributes MCPHost.available() reads, and nothing else.

    Used where the real Settings cannot be bent to the case: mcp_dir and
    mcp_entry are derived properties, so "the package is not installed" cannot be
    expressed by passing a field.
    """

    def __init__(self, *, enabled=True, node=True, entry: Path | None = None) -> None:
        self.mcp_enabled = enabled
        self.mcp_node_path = "" if node else str(Path(tempfile.gettempdir()) / "gone.exe")
        self.mcp_entry = entry or (Path(tempfile.gettempdir()) / "no-such-entry.js")


class TestHostGating(unittest.TestCase):
    def setUp(self):
        find_node.cache_clear()
        self.addCleanup(find_node.cache_clear)

    def _patch(self, target, name, value):
        original = getattr(target, name)
        setattr(target, name, value)
        self.addCleanup(setattr, target, name, original)

    def test_disabled_in_env_says_so(self):
        host = MCPHost(Settings(_env_file=None, mcp_enabled=False))
        ok, why = host.available()
        self.assertFalse(ok)
        self.assertIn("MCP_ENABLED=false", why)

    def test_missing_node_names_the_escape_hatch(self):
        self._patch(mcp, "find_node", lambda explicit="": None)
        ok, why = MCPHost(Settings(_env_file=None)).available()
        self.assertFalse(ok)
        self.assertIn("MCP_NODE_PATH", why)

    def test_missing_package_points_at_the_installer(self):
        ok, why = MCPHost(_StubSettings()).available()
        self.assertFalse(ok)
        self.assertIn("fetch_mcp.py", why)

    def test_the_checks_run_in_the_order_that_costs_least(self):
        """The disabled check first: a user who turned MCP off should be told
        that, not sent to install a package they deliberately do not want."""
        self._patch(mcp, "find_node", lambda explicit="": None)
        _, why = MCPHost(_StubSettings(enabled=False)).available()
        self.assertIn("MCP_ENABLED", why)

    def test_the_reasons_are_presentable_chinese(self):
        """available()'s second element goes straight into a tooltip."""
        self._patch(mcp, "find_node", lambda explicit="": None)
        for host in (
            MCPHost(Settings(_env_file=None, mcp_enabled=False)),
            MCPHost(Settings(_env_file=None)),
            MCPHost(_StubSettings()),
        ):
            _, why = host.available()
            self.assertTrue(why)
            self.assertFalse(any(ch in why for ch in "{}[]"), why)


class TestHostLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_ensure_started_refuses_when_unavailable(self):
        """state and detail are set first: the settings panel shows them, and an
        error that only surfaces as a raised exception leaves the panel blank."""
        host = MCPHost(Settings(_env_file=None, mcp_enabled=False))
        with self.assertRaises(MCPError):
            await host.ensure_started()
        self.assertEqual(host.state, "error")
        self.assertIn("MCP_ENABLED", host.detail)

    async def test_tools_refuses_an_empty_tool_list(self):
        host = MCPHost(Settings(_env_file=None))

        class _Empty:
            alive = True

            async def list_tools(self):
                return []

        async def fake_ensure():
            return _Empty()

        host.ensure_started = fake_ensure
        with self.assertRaises(MCPError) as caught:
            await host.tools()
        self.assertIn("没有提供任何工具", str(caught.exception))

    async def test_tools_is_cached_across_calls(self):
        host = MCPHost(Settings(_env_file=None))
        listed = []

        class _Once:
            alive = True

            async def list_tools(self):
                listed.append(1)
                return [{"name": "read_text_file"}]

        async def fake_ensure():
            return _Once()

        host.ensure_started = fake_ensure
        first = await host.tools()
        second = await host.tools()
        self.assertEqual(len(listed), 1)
        self.assertIs(first, second)

    async def test_stopping_an_unstarted_host_is_quiet(self):
        host = MCPHost(Settings(_env_file=None))
        await host.stop()
        self.assertEqual(host.state, "stopped")


class TestEntryScript(unittest.TestCase):
    def test_the_entry_path_is_derived_from_the_package_name(self):
        """Both live in app/mcp.py rather than config.py because
        scripts/fetch_mcp.py needs the same answer and must not import the
        settings layer to get it: two independent derivations of one path drift."""
        entry = fs_entry_script(Path("D:/x"))
        self.assertEqual(
            entry,
            Path("D:/x/node_modules/@modelcontextprotocol/server-filesystem/dist/index.js"),
        )

    def test_the_pinned_version_is_the_measured_one(self):
        self.assertEqual(FS_PACKAGE_VERSION, "2026.8.31")

    def test_the_client_name_is_stable(self):
        """It appears in the server's log, so changing it silently would make a
        bug report about this app unrecognisable."""
        self.assertEqual(CLIENT_NAME, "llama-cpp-demo")


# ---------------------------------------------------------------------------
# The real server. Everything below is skipped without Node + runtime/mcp.
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_SERVER, SKIP)
class TestRealServer(unittest.IsolatedAsyncioTestCase):
    """One handshake per test class, because a cold start measured 6.104 s.

    Rooted at a temp directory rather than the project so that nothing in here
    can reach this repo's own .env — the opposite of what the leak test below
    does on purpose, which builds its own client for that reason.
    """

    async def asyncSetUp(self):
        base = tempfile.TemporaryDirectory()
        self.addCleanup(base.cleanup)
        self.root = Path(base.name) / "root"
        self.root.mkdir()
        self.outside = Path(base.name) / "outside"
        self.outside.mkdir()
        (self.outside / "secret.txt").write_text("not for you", encoding="utf-8")
        self.client = MCPClient(NODE, ENTRY, [self.root])
        await self.client.start(30)
        self.addAsyncCleanup(self.client.stop)

    async def test_the_handshake_reports_the_measured_server(self):
        info = self.client.server_info
        self.assertEqual(info.get("name"), "secure-filesystem-server")
        self.assertEqual(info.get("version"), "0.2.0")

    async def test_the_protocol_version_is_the_measured_one(self):
        """Not a negotiation: _PROTOCOL_VERSION is what the server answered with."""
        self.assertEqual(mcp._PROTOCOL_VERSION, "2025-06-18")

    async def test_tools_list_returns_the_measured_fourteen(self):
        tools = await self.client.list_tools()
        self.assertEqual(len(tools), 14, sorted(t["name"] for t in tools))

    async def test_every_tool_carries_annotations(self):
        """tools.py tiers on `annotations.readOnlyHint` rather than on hardcoded
        names, so a tool without annotations has to fail closed — and this is the
        test that says upstream still sends them."""
        tools = await self.client.list_tools()
        for tool in tools:
            with self.subTest(tool["name"]):
                self.assertIsInstance(tool.get("annotations"), dict)
                self.assertIn("readOnlyHint", tool["annotations"])

    async def test_the_live_read_tier_is_six_and_the_write_increment_is_four(self):
        """The measured prices in .env.example (2184 and 1361 schema tokens) are
        per-tool counts in disguise. If these two numbers move, those prices and
        the 16384-context cliff percentage are both wrong."""
        tools = await self.client.list_tools()
        self.assertEqual(len(pick_fs_tools(tools, False)), 6)
        self.assertEqual(len(pick_fs_tools(tools, True)), 10)

    async def test_every_dropped_tool_still_exists_upstream(self):
        """FS_DROP is a hardcoded list of four names. If upstream renames one,
        the entry goes silently dead and the token it was saving comes back."""
        live = {t["name"] for t in await self.client.list_tools()}
        self.assertLessEqual(FS_DROP, live)

    async def test_every_live_parameter_is_classified(self):
        """The security pin, and the reason this file talks to the real server.

        _paths_in now collects every string argument whose key is in neither
        exemption set, so an unrecognised parameter fails closed. What that leaves to
        go wrong is an exemption that is too wide: a real location parameter sitting in
        _CONTENT_KEYS or _OPTION_KEYS would be skipped, and the deny list would stop
        covering it silently. Both halves are checked here against the live schema.

        If this fails because upstream added a parameter, the new name needs a
        decision — which bucket it belongs in — not a one-line test update.
        """
        tools = await self.client.list_tools()
        seen: set[str] = set()
        for tool in tools:
            properties = tool.get("inputSchema", {}).get("properties", {})
            seen.update(str(name).casefold() for name in properties)

        classified = _LOCATION_KEYS | _CONTENT_KEYS | _OPTION_KEYS
        self.assertEqual(
            seen - classified, set(),
            "a live parameter is in no bucket; _paths_in will treat it as a path, "
            "which is safe but probably not what upstream meant",
        )
        self.assertEqual(
            _LOCATION_KEYS & seen, _LOCATION_KEYS,
            "a location parameter this app relies on has disappeared upstream",
        )
        for exempt in (_CONTENT_KEYS | _OPTION_KEYS) & seen:
            with self.subTest(exempt):
                self.assertNotIn(
                    exempt, _LOCATION_KEYS, "an exemption would disarm the deny list"
                )

    async def test_a_read_inside_the_root_works(self):
        (self.root / "notes.md").write_text("# 标题\n正文\n", encoding="utf-8")
        text = await self.client.call_tool(
            "read_text_file", {"path": str(self.root / "notes.md")}
        )
        self.assertIn("标题", text)

    async def test_a_path_outside_the_root_is_refused_and_comes_back_as_text(self):
        """The server's own sandbox. Returned as text, not raised, so the model
        can read it and change course."""
        text = await self.client.call_tool(
            "read_text_file", {"path": str(self.outside / "secret.txt")}
        )
        self.assertIn("Access denied", text)
        self.assertNotIn("not for you", text)

    async def test_a_relative_argument_resolves_against_the_root_not_the_cwd(self):
        """The server behaviour that makes tools.is_denied the only real layer.

        dist/lib.js resolveRelativePathAgainstAllowedDirectories resolves a relative
        argument against each allowed root in turn. process.cwd() is consulted only
        when the server has no roots at all, and dist/index.js refuses to run in that
        state — so the child's cwd (temp, by _child_cwd) is irrelevant here.

        The sentinel name is unique and is asserted absent from the cwd, because
        otherwise a successful read would prove nothing about which base was used.
        """
        marker = f"mcp-sentinel-{uuid4().hex}.txt"
        (self.root / marker).write_text("found via the allowed root", encoding="utf-8")
        self.assertFalse(
            (Path(tempfile.gettempdir()) / marker).exists(),
            "the cwd must not hold a copy, or the test cannot tell the two apart",
        )
        self.assertNotIn(self.root, [self.client.cwd, *self.client.cwd.parents])
        text = await self.client.call_tool("read_text_file", {"path": marker})
        self.assertIn("found via the allowed root", text)

    async def test_a_relative_env_name_reaches_the_projects_real_env(self):
        """The consequence, pinned as a characterization test.

        This is the measurement that makes the absoluteness rule in tools.is_denied
        load-bearing rather than tidy: at this layer, ".env" and the absolute spelling
        of it are the same request. If upstream ever changes that, this fails and the
        rule can be reconsidered — until then, is_denied is the only thing between the
        model and the search API keys.

        Asserted through a computed boolean on purpose. An assertIn against the file's
        text would print this repo's real .env, keys included, into the test log on
        failure.
        """
        client = MCPClient(NODE, ENTRY, [ROOT])
        await client.start(30)
        self.addAsyncCleanup(client.stop)
        relative = await client.call_tool("read_text_file", {"path": ".env"})
        absolute = await client.call_tool("read_text_file", {"path": str(ROOT / ".env")})
        resolved_against_the_root = relative == absolute and "Access denied" not in relative
        self.assertTrue(
            resolved_against_the_root,
            "a relative name no longer resolves onto the allowed root — the "
            "absoluteness rule in tools.is_denied may now be redundant",
        )

    async def test_a_large_single_line_does_not_kill_the_connection(self):
        """asyncio's StreamReader defaults to 64 KiB; past that readline() clears
        its buffer and raises, killing the connection rather than merely
        returning a big result. 200 KB is 3x the default and 1/14 of the
        measured directory_tree line."""
        (self.root / "big.txt").write_text("x" * 200_000, encoding="utf-8")
        text = await self.client.call_tool(
            "read_text_file", {"path": str(self.root / "big.txt")}
        )
        self.assertEqual(len(text), 200_000)
        self.assertTrue(self.client.alive, "the connection must survive its own result")

    async def test_the_stream_limit_is_load_bearing(self):
        """The differential. Same file, same server, one constant changed — and
        the call fails. Without this the test above would pass at any limit large
        enough for 200 KB, and 32 MiB would be decorative."""
        (self.root / "big.txt").write_text("x" * 200_000, encoding="utf-8")
        original = mcp._STREAM_LIMIT
        mcp._STREAM_LIMIT = 64 * 1024
        self.addCleanup(setattr, mcp, "_STREAM_LIMIT", original)
        small = MCPClient(NODE, ENTRY, [self.root])
        await small.start(30)
        self.addAsyncCleanup(small.stop)
        with self.assertRaises(MCPError):
            await small.call_tool("read_text_file", {"path": str(self.root / "big.txt")})

    async def test_directory_tree_over_the_project_root_survives(self):
        """The measurement that set _STREAM_LIMIT: one 2,805,068-byte line,
        because the project root contains .venv/ and runtime/llama-vulkan/.
        Read-only, and measured at 0.182 s."""
        client = MCPClient(NODE, ENTRY, [ROOT])
        await client.start(30)
        self.addAsyncCleanup(client.stop)
        text = await client.call_tool("directory_tree", {"path": str(ROOT)})
        self.assertGreater(len(text), TREE_MIN_BYTES)
        self.assertIn("app", text)
        self.assertTrue(client.alive)

    async def test_concurrent_calls_reach_the_real_server_in_one_piece(self):
        """The fake-transport routing test, against a server that actually
        reorders. Four calls in flight, no intervening read."""
        (self.root / "a.txt").write_text("A", encoding="utf-8")
        (self.root / "b.txt").write_text("B", encoding="utf-8")
        calls = [
            self.client.call_tool("read_text_file", {"path": str(self.root / name)})
            for name in ("a.txt", "b.txt", "a.txt", "b.txt")
        ]
        results = await asyncio.gather(*calls)
        self.assertEqual(results, ["A", "B", "A", "B"])

    async def test_closing_stdin_exits_cleanly(self):
        """Measured exit code 0. kill() is the exception path, not the norm."""
        client = MCPClient(NODE, ENTRY, [self.root])
        await client.start(30)
        proc = client.proc
        await client.stop()
        self.assertEqual(proc.returncode, 0)

    async def test_stderr_is_drained_and_its_tail_remembered(self):
        """Nobody else reads the child's stderr, and an unread pipe fills: a
        server that logs enough would block on its own stderr mid-response,
        which looks exactly like a hang."""
        self.assertIsInstance(self.client.stderr_tail(), str)
        self.assertFalse(self.client._stderr_task.done())


@unittest.skipUnless(HAS_SERVER, SKIP)
class TestRealServerWrites(unittest.IsolatedAsyncioTestCase):
    """The write tier, which is what the 写入 checkbox actually exposes.

    Confined to a temp root: a write test aimed at the project directory would
    be indistinguishable from the failure mode this whole feature is supposed to
    prevent.
    """

    async def asyncSetUp(self):
        base = tempfile.TemporaryDirectory()
        self.addCleanup(base.cleanup)
        self.root = Path(base.name) / "root"
        self.root.mkdir()
        self.client = MCPClient(NODE, ENTRY, [self.root])
        await self.client.start(30)
        self.addAsyncCleanup(self.client.stop)

    async def test_create_directory_then_write_then_read_back(self):
        target = self.root / "made" / "note.md"
        await self.client.call_tool("create_directory", {"path": str(target.parent)})
        await self.client.call_tool(
            "write_file", {"path": str(target), "content": "# 写进去了\n"}
        )
        self.assertEqual(target.read_text(encoding="utf-8"), "# 写进去了\n")
        text = await self.client.call_tool("read_text_file", {"path": str(target)})
        self.assertIn("写进去了", text)

    async def test_move_file_really_moves(self):
        """The tooltip says move_file is effectively delete-plus-rename. Pinned
        so the warning in .env.example stays true."""
        source = self.root / "from.txt"
        source.write_text("payload", encoding="utf-8")
        destination = self.root / "to.txt"
        await self.client.call_tool(
            "move_file", {"source": str(source), "destination": str(destination)}
        )
        self.assertFalse(source.exists())
        self.assertEqual(destination.read_text(encoding="utf-8"), "payload")

    async def test_a_write_outside_the_root_is_refused_and_writes_nothing(self):
        outside = self.root.parent / "escaped.txt"
        text = await self.client.call_tool(
            "write_file", {"path": str(outside), "content": "no"}
        )
        self.assertIn("Access denied", text)
        self.assertFalse(outside.exists())

    async def test_search_files_finds_what_was_written(self):
        (self.root / "findme.py").write_text("x = 1\n", encoding="utf-8")
        text = await self.client.call_tool(
            "search_files", {"path": str(self.root), "pattern": "*.py"}
        )
        self.assertIn("findme.py", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
