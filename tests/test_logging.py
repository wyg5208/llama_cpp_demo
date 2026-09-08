"""Logging: the LOG_LEVEL setting, setup_logging()'s handler wiring, the chat line.

No server and no model is started anywhere in here. The chat-line tests call
app.main.chat() directly with three module globals replaced — the same seam the rest
of this suite uses for the parts that cannot run offline.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import re
import tempfile
import unittest
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError

from app import main as main_module
from app.config import (
    APP_LOG_BACKUPS,
    APP_LOG_MAX_BYTES,
    LOG_FORMAT,
    LOG_LEVELS,
    Settings,
    _HANDLER_TAG,
    setup_logging,
)
from app.main import ChatRequest, MessageIn
from app.settings_store import EDITABLE, describe

# The logger app/config.py logs to, and the one app/main.py logs to. They are not the
# same, so the two halves of this file watch different names.
CONFIG_LOG = "app.config"
APP_LOG = "app"

ELAPSED = re.compile(r"\bchat \S+ \d+\.\ds\b")


class _Collector(logging.Handler):
    """Keeps records instead of text, and formats them by calling getMessage().

    Explicit rather than left to a Formatter: a %-placeholder count that does not match
    the args raises from getMessage(), where the assertion can see it. A real handler
    swallows the same mistake into a "--- Logging error ---" block on stderr and the
    test still passes, having asserted on a record that was never rendered.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        self.messages.append(record.getMessage())


def _stub_settings(path: Path, level: str = "INFO") -> SimpleNamespace:
    """setup_logging reads exactly two attributes off Settings, so this is all it gets.

    A stub rather than a real Settings keeps these tests independent of .env, whose
    MODEL_PATH and search keys have nothing to do with logging.
    """
    return SimpleNamespace(app_log=path, log_level=level)


class _LoggerSandbox:
    """Snapshots the loggers setup_logging touches, and puts them back.

    setup_logging mutates process-global state. A handler left behind pointing at a
    temp directory that has since been deleted prints "--- Logging error ---" over the
    rest of the suite, so an unrelated test starts failing for a reason that has nothing
    to do with it. Everything below is therefore undone in the same order it was done.
    """

    WATCHED = ("", "uvicorn", "uvicorn.error", "uvicorn.access", CONFIG_LOG, APP_LOG)

    def setUp(self) -> None:  # noqa: N802 - unittest's name
        super().setUp()
        self._saved = {
            name: (
                list(logging.getLogger(name).handlers),
                logging.getLogger(name).level,
                logging.getLogger(name).propagate,
            )
            for name in self.WATCHED
        }
        # The console handler setup_logging attaches writes to sys.stderr, captured at
        # construction, so swapping the stream here silences it for the whole test.
        # Registered first so it is undone last: addCleanup is LIFO, and both drain() and
        # the temp-dir removal below should happen while it is still redirected. The
        # runner printed its own summary through a reference it took at startup, so
        # nothing unittest writes is affected.
        self.console = io.StringIO()
        self._stderr = contextlib.redirect_stderr(self.console)
        self._stderr.__enter__()
        self.addCleanup(self._stderr.__exit__, None, None, None)
        # Directory registered before _restore for the same LIFO reason: drain() is what
        # closes the handler holding app.log open, and Windows refuses to delete a file
        # that is still open.
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.log_path = Path(self._tmp.name) / "app.log"
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        self.drain()
        for name, (handlers, level, propagate) in self._saved.items():
            named = logging.getLogger(name)
            named.handlers[:] = handlers
            named.setLevel(level)
            named.propagate = propagate

    def drain(self) -> None:
        """Flush and detach the handlers these tests own, so the file can be read.

        Only the tagged ones and the collectors: closing a handler the test runner or
        another module installed, and then restoring it closed, would leave the rest of
        the suite writing into a dead stream.
        """
        for name in self.WATCHED:
            named = logging.getLogger(name)
            for handler in list(named.handlers):
                if getattr(handler, _HANDLER_TAG, False) or isinstance(handler, _Collector):
                    handler.flush()
                    named.removeHandler(handler)
                    handler.close()

    def collect(self, name: str) -> _Collector:
        """Attach a collector to a logger and make sure it accepts everything.

        Appended rather than replacing the logger's handlers, and propagate is left
        alone: the unwritable-path test needs the same warning to reach both this
        collector and the console handler setup_logging put on root, and silencing
        propagation would leave "degrades to console-only" with no way to check the
        console half. Root's own level does not filter a propagated record — only the
        originating logger's effective level does — so DEBUG here is enough.
        """
        collector = _Collector()
        named = logging.getLogger(name)
        named.addHandler(collector)
        named.setLevel(logging.DEBUG)
        return collector

    def read_log(self) -> str:
        self.drain()
        if not self.log_path.exists():
            return ""
        return self.log_path.read_text(encoding="utf-8")

    def tagged_root_handlers(self) -> list[logging.Handler]:
        return [h for h in logging.getLogger().handlers if getattr(h, _HANDLER_TAG, False)]


class TestLogLevelSetting(unittest.TestCase):
    def test_default_is_info(self):
        self.assertEqual(Settings(_env_file=None).log_level, "INFO")

    def test_every_level_name_is_accepted(self):
        for level in LOG_LEVELS:
            with self.subTest(level=level):
                self.assertEqual(Settings(_env_file=None, log_level=level).log_level, level)

    def test_case_and_padding_are_normalised_in_the_validator(self):
        # Normalised there rather than in setup_logging so the value read back out of
        # the settings object is the one that was configured, not the one typed.
        self.assertEqual(Settings(_env_file=None, log_level="  debug ").log_level, "DEBUG")
        self.assertEqual(Settings(_env_file=None, log_level="Warning").log_level, "WARNING")

    def test_a_typo_is_rejected_rather_than_defaulted(self):
        # A LOG_LEVEL=VERBOSE that quietly left the app at INFO would be diagnosed as
        # "DEBUG produced nothing" instead of as a misspelling. The message has to say
        # which values do work, or the fix is a trip to the source.
        with self.assertRaises(ValidationError) as caught:
            Settings(_env_file=None, log_level="verbose")
        text = str(caught.exception)
        self.assertIn("LOG_LEVEL", text)
        self.assertIn("verbose", text)
        for level in LOG_LEVELS:
            self.assertIn(level, text)

    def test_not_editable_from_the_settings_panel(self):
        # The handlers are built once at startup, so a level changed from the panel
        # would look saved and do nothing until the next restart. Better to not offer it
        # than to offer a control that lies.
        self.assertNotIn("log_level", EDITABLE)

    def test_the_panel_says_why_it_is_read_only(self):
        # Being read-only is half of it: describe() renders the reason verbatim under
        # the greyed-out value, and a blank row there reads as "failed to load". Every
        # other non-editable field already has one.
        row = next(
            f for f in describe(Settings(_env_file=None), {})["fields"]
            if f["name"] == "log_level"
        )
        self.assertFalse(row["editable"])
        self.assertTrue(row["reason"])
        self.assertEqual(row["value"], "INFO")

    def test_the_app_log_sits_beside_the_server_log(self):
        s = Settings(_env_file=None)
        self.assertEqual(s.app_log, s.server_log.parent / "app.log")
        self.assertEqual(s.app_log.name, "app.log")


class TestSetupLogging(_LoggerSandbox, unittest.TestCase):
    def test_it_writes_chinese_to_the_file_as_utf8(self):
        setup_logging(_stub_settings(self.log_path))
        logging.getLogger(APP_LOG).info("模型已就绪：gemma-4-E4B-it")
        # Read back as UTF-8 explicitly: the console here is cp936, and a file written
        # in the platform default would be unreadable the moment it left this machine.
        self.assertIn("模型已就绪：gemma-4-E4B-it", self.read_log())

    def test_the_console_handler_survives_the_file_handler(self):
        setup_logging(_stub_settings(self.log_path))
        logging.getLogger(APP_LOG).info("console line")
        # `is` rather than isinstance: RotatingFileHandler subclasses StreamHandler, so
        # the file handler on its own would satisfy an isinstance check and this would
        # pass with the console line missing. The buffer is the stronger half of the
        # evidence — setUp swapped sys.stderr for it, so this is what the console got.
        self.assertIn(logging.StreamHandler, [type(h) for h in self.tagged_root_handlers()])
        self.assertIn("console line", self.console.getvalue())

    def test_exactly_one_console_and_one_file_handler_both_formatted(self):
        setup_logging(_stub_settings(self.log_path))
        handlers = self.tagged_root_handlers()
        self.assertEqual(len(handlers), 2)
        self.assertEqual(sum(type(h) is logging.StreamHandler for h in handlers), 1)
        self.assertEqual(sum(isinstance(h, RotatingFileHandler) for h in handlers), 1)
        for handler in handlers:
            self.assertEqual(handler.formatter._fmt, LOG_FORMAT)

    def test_rotation_is_configured_rather_than_switched_off(self):
        # The wiring is asserted rather than 1 MB written: rollover itself is stdlib
        # behaviour and needs no proof here, while maxBytes=0 silently turns rotation
        # off and lets app.log grow without bound. That is the mistake worth catching.
        setup_logging(_stub_settings(self.log_path))
        handler = next(
            h for h in self.tagged_root_handlers() if isinstance(h, RotatingFileHandler)
        )
        self.assertEqual(handler.maxBytes, APP_LOG_MAX_BYTES)
        self.assertEqual(handler.backupCount, APP_LOG_BACKUPS)
        self.assertGreater(handler.maxBytes, 0)
        self.assertGreater(handler.backupCount, 0)
        self.assertEqual(handler.encoding, "utf-8")
        # Append, never truncate: server_log is opened with "w" on purpose because a
        # stale llama-server tail in a 502 misleads, and app.log is the opposite — it is
        # the only record that outlives a restart.
        self.assertEqual(handler.mode, "a")
        self.assertEqual(handler.baseFilename, os.path.abspath(self.log_path))

    def test_calling_it_twice_replaces_the_handlers(self):
        # run.py calls this before uvicorn starts and lifespan calls it again, so the
        # second call has to undo the first. Stacking is invisible in a console
        # scrollback and obvious in a file, where every line then appears twice.
        setup_logging(_stub_settings(self.log_path))
        setup_logging(_stub_settings(self.log_path))
        self.assertEqual(len(self.tagged_root_handlers()), 2)
        logging.getLogger(APP_LOG).info("once")
        self.assertEqual(self.read_log().count("once"), 1)

    def test_debug_reaches_the_file_only_when_asked_for(self):
        setup_logging(_stub_settings(self.log_path, "INFO"))
        logging.getLogger(APP_LOG).debug("one of app/mcp.py's three debug sites")
        # The handler opens the file at construction, so it exists and is empty rather
        # than absent: the difference matters when reading a real runtime/ by hand.
        self.assertEqual(self.read_log(), "")

        setup_logging(_stub_settings(self.log_path, "DEBUG"))
        logging.getLogger(APP_LOG).debug("one of app/mcp.py's three debug sites")
        self.assertIn("app/mcp.py", self.read_log())

    def test_library_noise_stays_out_even_at_debug(self):
        # The measured failure this guards: root at LOG_LEVEL=DEBUG makes every library
        # logger that does not set its own level inherit it, and httpcore traces each
        # socket frame. Six HTTP requests on a real start wrote ~80 lines into app.log,
        # and this app streams every token from llama-server through httpx, so one chat
        # turn at DEBUG would rotate away everything worth keeping.
        setup_logging(_stub_settings(self.log_path, "DEBUG"))

        logging.getLogger(APP_LOG).debug("our own debug line")
        logging.getLogger("httpcore.http11").debug("send_request_headers.started")
        logging.getLogger("httpx").info("HTTP Request: POST http://127.0.0.1:8081/v1/chat")
        logging.getLogger("asyncio").debug("Using proactor: IocpProactor")

        text = self.read_log()
        self.assertIn("our own debug line", text)
        self.assertNotIn("send_request_headers", text)
        self.assertNotIn("HTTP Request", text)
        self.assertNotIn("proactor", text)

    def test_the_root_floor_is_warning_and_still_follows_a_stricter_level(self):
        # max(), so DEBUG and INFO stop at the floor while ERROR and CRITICAL keep
        # narrowing root: asking for less of your own code should not mean more of
        # everybody else's.
        for level, expected in (
            ("DEBUG", logging.WARNING),
            ("INFO", logging.WARNING),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL),
        ):
            with self.subTest(level=level):
                setup_logging(_stub_settings(self.log_path, level))
                self.assertEqual(logging.getLogger().level, expected)
                self.assertEqual(logging.getLogger(APP_LOG).level, logging.getLevelName(level))

    def test_an_unwritable_path_degrades_to_console_only(self):
        # A read-only runtime/ or a full disk must not stop the app. The parent here is
        # a file, so mkdir raises FileExistsError, which is the OSError the handler is
        # wrapped in.
        blocker = Path(self._tmp.name) / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        collector = self.collect(CONFIG_LOG)

        setup_logging(_stub_settings(blocker / "app.log"))

        handlers = self.tagged_root_handlers()
        self.assertEqual(len(handlers), 1)
        self.assertIs(type(handlers[0]), logging.StreamHandler)
        self.assertFalse(any(isinstance(h, RotatingFileHandler) for h in handlers))
        self.assertEqual(len(collector.messages), 1)
        self.assertEqual(collector.records[0].levelno, logging.WARNING)
        self.assertIn("打不开", collector.messages[0])
        # "Degrades to console-only" is only true if the console still works, and the
        # buffer setUp captured says it did.
        self.assertIn("打不开", self.console.getvalue())

    def test_uvicorn_startup_lines_reach_the_file(self):
        # uvicorn 0.52 runs configure_logging() in Config.__init__, before it imports
        # this app, and gives "uvicorn" its own handler with propagate=False. Left
        # alone, its records never reach root and never reach the file.
        uvicorn_log = logging.getLogger("uvicorn")
        uvicorn_log.addHandler(logging.NullHandler())
        uvicorn_log.propagate = False
        error_log = logging.getLogger("uvicorn.error")
        error_log.addHandler(logging.NullHandler())

        setup_logging(_stub_settings(self.log_path))

        # Rerouted by clearing handlers and propagating, not by handing them the file
        # handler: a shared handler on a logger that still propagates writes twice.
        self.assertEqual(uvicorn_log.handlers, [])
        self.assertTrue(uvicorn_log.propagate)
        self.assertEqual(error_log.handlers, [])
        self.assertTrue(error_log.propagate)

        uvicorn_log.info("Uvicorn running on http://127.0.0.1:8123")
        error_log.info("Started server process [12345]")
        text = self.read_log()
        self.assertIn("Uvicorn running on http://127.0.0.1:8123", text)
        self.assertIn("Started server process [12345]", text)

    def test_uvicorn_loggers_follow_log_level(self):
        # Their own level, not just root's: uvicorn sets both to INFO, which would
        # filter a DEBUG record out before the handler LOG_LEVEL=DEBUG just installed
        # ever saw it.
        uvicorn_log = logging.getLogger("uvicorn")
        uvicorn_log.setLevel(logging.INFO)

        setup_logging(_stub_settings(self.log_path, "DEBUG"))

        self.assertEqual(uvicorn_log.level, logging.DEBUG)
        uvicorn_log.debug("a debug line from uvicorn")
        self.assertIn("a debug line from uvicorn", self.read_log())

    def test_the_access_log_stays_out_of_the_file(self):
        # The browser polls /api/stats every 2 s (static/app.js, STATS_MS=2000), so an
        # access log in the file is ~43,000 lines a day at ~100 bytes each — about 4 MB,
        # which is the whole rotation budget spent daily on telemetry nobody reads back.
        access = logging.getLogger("uvicorn.access")
        sentinel = logging.NullHandler()
        access.addHandler(sentinel)
        access.propagate = False
        access.setLevel(logging.WARNING)

        setup_logging(_stub_settings(self.log_path))

        self.assertIn(sentinel, access.handlers)
        self.assertFalse(access.propagate)
        self.assertEqual(access.level, logging.WARNING)

        access.info('127.0.0.1:52000 - "GET /api/stats HTTP/1.1" 200')
        self.assertNotIn("/api/stats", self.read_log())


class _FakeRuntime:
    state = "ready"
    detail = ""
    # False, so every capability below is clamped off and no tool schema is built: the
    # line under test reports what was offered, and this keeps the offer at zero.
    supports_tools = False
    model_path = Path("F:/models/gguf/gemma-4-E4B-it-Q4_K_M.gguf")

    def status(self) -> dict:
        # chat() reads n_ctx off the server's own /props rather than guessing, so this
        # is the number the log line reports.
        return {"n_ctx": 40960}


class _FakeMCPHost:
    def available(self) -> tuple[bool, str]:
        return False, "filesystem MCP server not installed"


async def _fake_stream_chat(settings, messages, tools, runner, thinking):
    """The event shapes app/llm.py really yields, in the order it yields them."""
    yield {"type": "delta", "text": "你"}
    yield {"type": "reasoning", "text": "先想一下"}
    yield {"type": "delta", "text": "好"}
    yield {
        "type": "done",
        "sources": [],
        "usage": {"prompt_tokens": 1234, "completion_tokens": 56},
    }


class TestTheChatRequestLine(_LoggerSandbox, unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:  # noqa: N802 - unittest's name
        super().setUp()
        self._patched = (
            main_module.runtime,
            main_module.mcp_host,
            main_module.stream_chat,
        )
        main_module.runtime = _FakeRuntime()
        main_module.mcp_host = _FakeMCPHost()
        main_module.stream_chat = _fake_stream_chat
        self.collector = self.collect(APP_LOG)

    def tearDown(self) -> None:  # noqa: N802 - unittest's name
        (
            main_module.runtime,
            main_module.mcp_host,
            main_module.stream_chat,
        ) = self._patched
        super().tearDown()

    async def _drain(self, response) -> list[str]:
        return [chunk async for chunk in response.body_iterator]

    async def test_a_completed_request_logs_one_line(self):
        request = ChatRequest(
            messages=[MessageIn(role="user", content="你好，介绍一下你自己")]
        )

        chunks = await self._drain(await main_module.chat(request))

        self.assertEqual(len(chunks), 4)
        self.assertEqual(len(self.collector.messages), 1)
        self.assertEqual(self.collector.records[0].levelno, logging.INFO)
        line = self.collector.messages[0]
        self.assertRegex(line, ELAPSED)
        self.assertTrue(line.startswith("chat gemma-4-E4B-it-Q4_K_M.gguf "))
        for field in (
            "sent=1/1",
            "ctx=40960",
            "tools=0",
            "features=export_hint",
            "prompt_tok=1234",
            "completion_tok=56",
            "deltas=2",
            "reasoning=1",
            "tool_results=0",
            "artifacts=0",
            "notes=0",
            "errors=0",
        ):
            self.assertIn(field, line)

    async def test_the_line_carries_counts_and_never_the_conversation(self):
        # The archive already keeps the text, so a log that repeats it is a second copy
        # of everything anyone typed into this app, in a file nobody reviews. A live
        # search API key was printed into a test log here once, which is why the rule is
        # that this line reports what it counted and never a value it was handed.
        secret = "hunter2 sk-live-abcdef0123456789"
        request = ChatRequest(
            messages=[
                MessageIn(role="user", content=f"我的密码是 {secret}"),
                MessageIn(role="assistant", content="好的，我不会告诉别人。"),
            ]
        )

        await self._drain(await main_module.chat(request))

        line = self.collector.messages[0]
        for word in (*secret.split(), "密码", "不会告诉别人"):
            self.assertNotIn(word, line)
        self.assertNotIn("http", line)

    async def test_a_client_disconnect_is_still_logged(self):
        # GeneratorExit is thrown into whichever yield was suspended. This is the 停止
        # button and the closed tab, and it is exactly the case that used to leave no
        # trace at all — which is why the finally wraps the whole body.
        request = ChatRequest(messages=[MessageIn(role="user", content="你好")])

        iterator = (await main_module.chat(request)).body_iterator
        await iterator.__anext__()
        await iterator.aclose()

        self.assertEqual(len(self.collector.messages), 1)
        line = self.collector.messages[0]
        self.assertIn("deltas=1", line)
        # The done event was never reached, so there are no token counts to report.
        # None rather than 0: zero tokens is a real answer from llama-server, and the
        # two have to stay distinguishable.
        self.assertIn("prompt_tok=None", line)
        self.assertIn("completion_tok=None", line)
        self.assertIn("errors=0", line)

    async def test_a_request_that_sends_nothing_is_still_logged(self):
        # Every turn filtered out as empty: the degenerate replay shape behind the
        # 罢工 report. It sends nothing, and it still has to appear.
        request = ChatRequest(messages=[MessageIn(role="assistant", content="")])

        chunks = await self._drain(await main_module.chat(request))

        self.assertIn("event: error", chunks[0])
        self.assertEqual(len(self.collector.messages), 1)
        line = self.collector.messages[0]
        self.assertIn("sent=0/0", line)
        self.assertIn("errors=1", line)

    async def test_a_refused_chat_is_logged_too(self):
        # This path returns before events() is constructed, so it needs its own line:
        # without it, a chat that produced nothing at all would be the one request
        # app.log never mentions.
        main_module.runtime = _FakeRuntime()
        main_module.runtime.state = "switching"
        request = ChatRequest(messages=[MessageIn(role="user", content="你好")])

        chunks = await self._drain(await main_module.chat(request))

        self.assertIn("正在切换模型", chunks[0])
        self.assertEqual(len(self.collector.messages), 1)
        self.assertEqual(self.collector.records[0].levelno, logging.WARNING)
        self.assertIn("chat refused: runtime state=switching", self.collector.messages[0])


if __name__ == "__main__":
    unittest.main()
