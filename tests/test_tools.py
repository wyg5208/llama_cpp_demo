"""Tool assembly, the filesystem deny list, and the runner for everything but search.

Standard library only, by decision — pytest was explicitly rejected as a new
dependency. Run from the project root:

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m unittest discover -s tests -v

Two groups here pin numbers rather than behaviour, and both are worth the brittleness.
TestNativeToolCosts records what a checkbox costs on a 16384 window — the fallback for
any GGUF missing from N_CTX_OVERRIDES — so that adding a tool fails a test instead of
quietly eating another few percent of the context. TestDenyList records a leak that was
measured, not imagined: read_text_file reached this project's real .env and returned
the search API keys in it, in four different spellings, and only one of the four was
refused at the time.

Filesystem *tiering* runs against synthetic tool objects rather than the live server,
so this whole file passes on a machine with no Node.js. tests/test_mcp.py pins the same
behaviour against the real thing when one is installed, and compares its tool list to
UPSTREAM_TOOLS below.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.config import DEFAULT_SYSTEM_PROMPT, ROOT, Settings
from app.documents import budget_safety, estimate_tokens
from app.search import TOOLS as SEARCH_TOOLS
from app.tools import (
    ENV_ALLOW,
    FS_DROP,
    MAX_THOUGHTS,
    MAX_TOOL_RESULT_CHARS,
    MAX_TOOL_TOTAL_CHARS,
    PROMPT_LINES,
    RECALL_LIMIT,
    RECALL_TOOL,
    REMEMBER_TOOL,
    SENSITIVE_DIRS,
    THINK_TOOL,
    ToolRunner,
    _CONTENT_KEYS,
    _LOCATION_KEYS,
    _OPTION_KEYS,
    _paths_in,
    build_tools,
    clip,
    effective_prompt,
    fs_tool_names,
    is_denied,
    mcp_to_openai,
    pick_fs_tools,
)

# The 14 tools @modelcontextprotocol/server-filesystem 2026.8.31 reports, recorded from
# a real tools/list. Used to build the synthetic tool objects below, so the tiering
# tests run on a machine with no Node.js installed; test_mcp.py checks the live list
# separately, and skips itself when it cannot.
UPSTREAM_TOOLS = frozenset(
    {
        "create_directory",
        "directory_tree",
        "edit_file",
        "get_file_info",
        "list_allowed_directories",
        "list_directory",
        "list_directory_with_sizes",
        "move_file",
        "read_file",
        "read_media_file",
        "read_multiple_files",
        "read_text_file",
        "search_files",
        "write_file",
    }
)

# The location parameters across all 14, also recorded from the real schemas. Exactly
# what app/tools._LOCATION_KEYS has to cover.
UPSTREAM_LOCATION_PARAMS = frozenset({"path", "paths", "source", "destination"})

# Every parameter name across all 14, lowercased as _paths_in sees them. The three
# buckets in app/tools.py have to cover all of these: a name in none of them is now
# treated as a location, which is safe but almost certainly not what upstream meant.
UPSTREAM_PARAMS = UPSTREAM_LOCATION_PARAMS | frozenset(
    {"content", "pattern", "excludepatterns", "sortby", "dryrun", "edits", "head", "tail"}
)


def _tool(name, read_only=True, annotations=True, **extra_hints):
    """One tools/list entry in the shape the real server sends, noise included.

    title / outputSchema / execution / annotations all come back from the server and
    none of them are things llama-server reads, so they are here to be stripped.
    """
    entry = {
        "name": name,
        "description": f"{name} does a thing.",
        "title": name.replace("_", " ").title(),
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        "outputSchema": {"type": "object"},
        "execution": {"mode": "sync"},
    }
    if annotations:
        entry["annotations"] = {"readOnlyHint": read_only, "openWorldHint": False, **extra_hints}
    return entry


def _offer(*names):
    """A tool list shaped the way build_tools returns it, without the schemas."""
    return [
        {"type": "function", "function": {"name": n, "description": "", "parameters": {}}}
        for n in names
    ]


def _tokens(tools) -> int:
    return estimate_tokens(json.dumps(tools, ensure_ascii=False))


class _Settings:
    """ToolRunner reads exactly one attribute off settings, so that is all this is."""

    def __init__(self, roots):
        self.fs_roots = [Path(r) for r in roots]


class _FakeHost:
    """Stands in for MCPHost and records what reached it.

    Recording is the point: "the deny list stopped this" has to be asserted, not
    inferred from the wording of the refusal, because a refusal produced after the
    server was already called is the bug these tests exist to catch.
    """

    def __init__(self, text="（内容）", error=None):
        self.calls = []
        self.text = text
        self.error = error

    async def ensure_started(self):
        return self

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if self.error is not None:
            raise self.error
        return self.text


async def _collect(runner, name, args):
    return [event async for event in runner.run(name, args)]


def _result(events) -> str:
    found = [e["content"] for e in events if e.get("type") == "tool_result"]
    assert len(found) == 1, f"expected exactly one tool_result, got {events}"
    return found[0]


class TestNativeToolCosts(unittest.TestCase):
    """What each checkbox costs, measured with this repo's own estimate_tokens.

    Pinned as absolute numbers because the number IS the fact worth protecting: on a
    16384 window everything at once is over 40% of the context before one message, and
    that is why none of the four boxes is on by default.
    """

    COMBOS = (
        ("search only", dict(search=True), 2, 807),
        ("+ think", dict(search=True, think=True), 3, 1080),
        ("+ memory", dict(search=True, memory=True), 4, 1492),
        ("+ think + memory", dict(search=True, memory=True, think=True), 5, 1764),
    )

    def test_schema_cost_of_each_combination(self):
        for label, kwargs, count, tokens in self.COMBOS:
            with self.subTest(label):
                tools = build_tools(**kwargs)
                self.assertEqual(len(tools), count)
                self.assertEqual(_tokens(tools), tokens)

    def test_single_native_tools(self):
        for label, tool, tokens in (
            ("remember", REMEMBER_TOOL, 402),
            ("recall", RECALL_TOOL, 281),
            ("think", THINK_TOOL, 271),
        ):
            with self.subTest(label):
                self.assertEqual(_tokens(tool), tokens)

    def test_prompt_lines(self):
        """The per-feature system prompt lines, which cost tokens every single turn.

        export_hint and doc_gen are the two halves of one instruction: the first tells a
        model without save_document to point at the 导出 button, the second tells a model
        with it to call the tool. Both are pinned here because they are mutually exclusive
        and a rewording that grows one shrinks the other's share of the same budget.
        """
        for key, tokens in (
            ("memory", 128),
            ("think", 89),
            ("fs_write", 117),
            ("export_hint", 57),
            ("doc_gen", 299),
        ):
            with self.subTest(key):
                self.assertEqual(estimate_tokens(PROMPT_LINES[key]), tokens)

    def test_budget_safety_of_the_status_quo(self):
        """The one row that needs no filesystem fixture: today's shipped default.

        Measured through effective_prompt rather than against the bare constant, because
        that is what a real request sends: 生成文档 off means export_hint is appended.
        The two are equal by design -- export_hint is verbatim the sentence that used to
        end DEFAULT_SYSTEM_PROMPT unconditionally, so the figure did not move when it was
        moved out, and this row stays 1737 either way.
        """
        tools = build_tools(search=True)
        prompt = effective_prompt(DEFAULT_SYSTEM_PROMPT, {"export_hint"})
        self.assertEqual(
            budget_safety(prompt, json.dumps(tools, ensure_ascii=False), 8), 1737
        )

    def test_disabled_features_add_nothing(self):
        out = effective_prompt(DEFAULT_SYSTEM_PROMPT, set())
        self.assertEqual(out, DEFAULT_SYSTEM_PROMPT.rstrip())
        for key, line in PROMPT_LINES.items():
            with self.subTest(key):
                self.assertNotIn(line, out)


class TestBuildTools(unittest.TestCase):
    def test_nothing_enabled_is_no_tools(self):
        self.assertEqual(build_tools(), [])

    def test_search_tools_are_the_same_objects(self):
        """app/search.py's TOOLS is reused, not copied.

        tests/test_documents.py imports that constant directly to compute the image
        budget, so a second copy here would be two spellings of one fact.
        """
        for mine, theirs in zip(build_tools(search=True), SEARCH_TOOLS):
            self.assertIs(mine, theirs)

    def test_order_is_stable_regardless_of_the_call(self):
        """Search, then memory, then think, then filesystem.

        llama-server caches the common prompt prefix, so a list whose order depended on
        a set's iteration would miss that cache on every request.
        """
        raw = [_tool("list_directory"), _tool("write_file", read_only=False)]
        names = [t["function"]["name"] for t in build_tools(
            search=True, memory=True, think=True, fs_read=True, fs_write=True, fs_raw=raw,
        )]
        self.assertEqual(names, [
            *[t["function"]["name"] for t in SEARCH_TOOLS],
            "remember", "recall", "think", "list_directory", "write_file",
        ])

    def test_fs_write_without_fs_read_yields_nothing(self):
        raw = [_tool("write_file", read_only=False)]
        self.assertEqual(build_tools(fs_write=True, fs_raw=raw), [])

    def test_fs_read_without_a_server_yields_nothing(self):
        self.assertEqual(build_tools(fs_read=True), [])
        self.assertEqual(build_tools(fs_read=True, fs_raw=[]), [])

    def test_duplicate_names_are_not_silently_deduplicated(self):
        """Assembly is concatenation; a name appearing twice upstream is a real change
        and should be visible in the count rather than absorbed."""
        raw = [_tool("list_directory"), _tool("list_directory")]
        self.assertEqual(len(build_tools(fs_read=True, fs_raw=raw)), 2)


class TestMcpToOpenai(unittest.TestCase):
    def test_keeps_only_what_llama_server_reads(self):
        converted = mcp_to_openai(_tool("read_text_file"))
        self.assertEqual(converted, {
            "type": "function",
            "function": {
                "name": "read_text_file",
                "description": "read_text_file does a thing.",
                "parameters": _tool("read_text_file")["inputSchema"],
            },
        })

    def test_drops_the_four_unread_fields(self):
        fn = mcp_to_openai(_tool("read_text_file"))["function"]
        for field in ("title", "outputSchema", "execution", "annotations"):
            self.assertNotIn(field, fn)

    def test_missing_input_schema_becomes_an_empty_object(self):
        entry = _tool("list_allowed_directories")
        del entry["inputSchema"]
        self.assertEqual(
            mcp_to_openai(entry)["function"]["parameters"],
            {"type": "object", "properties": {}},
        )

    def test_non_dict_input_schema_becomes_an_empty_object(self):
        entry = _tool("x")
        entry["inputSchema"] = ["not", "a", "schema"]
        self.assertEqual(
            mcp_to_openai(entry)["function"]["parameters"],
            {"type": "object", "properties": {}},
        )

    def test_missing_name_becomes_empty_rather_than_none(self):
        """llama-server serialises this into the prompt; a literal "None" would be a
        tool the model can call and nothing can dispatch."""
        self.assertEqual(mcp_to_openai({"description": "d"})["function"]["name"], "")


class TestPickFsTools(unittest.TestCase):
    RAW = [
        _tool("read_text_file", read_only=True),
        _tool("write_file", read_only=False, destructiveHint=True),
        _tool("a_new_reader", read_only=True),
        _tool("a_new_writer", read_only=False),
        _tool("mystery", annotations=False),
        _tool("read_file", read_only=True),
    ]

    def test_read_tier_is_by_annotation_not_by_name(self):
        self.assertEqual(
            [t["function"]["name"] for t in pick_fs_tools(self.RAW, allow_write=False)],
            ["read_text_file", "a_new_reader"],
        )

    def test_write_tier_adds_everything_else(self):
        self.assertEqual(
            [t["function"]["name"] for t in pick_fs_tools(self.RAW, allow_write=True)],
            ["read_text_file", "write_file", "a_new_reader", "a_new_writer", "mystery"],
        )

    def test_a_reader_name_annotated_as_writing_is_withheld(self):
        """The pin behind tiering on annotations: upstream may rename or repurpose a
        tool, and the write checkbox has to follow the annotation rather than the name."""
        raw = [_tool("read_text_file", read_only=False)]
        self.assertEqual(pick_fs_tools(raw, allow_write=False), [])
        self.assertEqual(len(pick_fs_tools(raw, allow_write=True)), 1)

    def test_no_annotations_is_treated_as_writing(self):
        """Fail-closed: with the write box unticked an unknown tool is withheld rather
        than waved through."""
        self.assertEqual(pick_fs_tools([_tool("mystery", annotations=False)], False), [])

    def test_annotations_of_the_wrong_type_is_treated_as_writing(self):
        entry = _tool("mystery")
        entry["annotations"] = "readOnly"
        self.assertEqual(pick_fs_tools([entry], False), [])

    def test_read_only_hint_of_the_wrong_type_is_treated_as_writing(self):
        """`is not True` rather than `not`: the string "false" is truthy."""
        entry = _tool("x")
        entry["annotations"] = {"readOnlyHint": "true"}
        self.assertEqual(pick_fs_tools([entry], False), [])

    def test_dropped_tools_never_appear(self):
        for name in sorted(FS_DROP):
            with self.subTest(name):
                raw = [_tool(name, read_only=True), _tool(name, read_only=False)]
                self.assertEqual(pick_fs_tools(raw, allow_write=True), [])

    def test_fs_drop_only_names_real_upstream_tools(self):
        """Guards the drop list against a typo, which would silently re-expose a tool."""
        self.assertTrue(FS_DROP <= UPSTREAM_TOOLS, FS_DROP - UPSTREAM_TOOLS)

    def test_junk_entries_are_skipped(self):
        raw = [None, "x", 7, {}, _tool(""), _tool("list_directory")]
        self.assertEqual(
            [t["function"]["name"] for t in pick_fs_tools(raw, allow_write=False)],
            ["list_directory"],
        )

    def test_fs_tool_names_agrees_with_pick_fs_tools(self):
        """One derivation, not two: main.py passes these names to ToolRunner and the
        runner's gate must not disagree with the schemas in the payload."""
        for allow_write in (False, True):
            with self.subTest(allow_write):
                picked = pick_fs_tools(self.RAW, allow_write)
                self.assertEqual(
                    fs_tool_names(self.RAW, allow_write),
                    {t["function"]["name"] for t in picked},
                )

    def test_the_recorded_upstream_list_tiers_to_six_and_four(self):
        """What the plan was built on, restated as a fixture so a synthetic change here
        cannot drift away from what the real server does (test_mcp.py checks the real
        one when Node is installed)."""
        raw = [
            _tool(n, read_only=n not in {"write_file", "edit_file", "move_file", "create_directory"})
            for n in sorted(UPSTREAM_TOOLS)
        ]
        self.assertEqual(len(fs_tool_names(raw, False)), 6)
        self.assertEqual(len(fs_tool_names(raw, True) - fs_tool_names(raw, False)), 4)


class TestEffectivePrompt(unittest.TestCase):
    BASE = "你是一个助手。"

    def test_appends_one_line_per_enabled_feature(self):
        out = effective_prompt(self.BASE, {"memory", "think"})
        self.assertEqual(out.split("\n"), [self.BASE, PROMPT_LINES["memory"], PROMPT_LINES["think"]])

    def test_order_follows_the_table_not_the_argument(self):
        """So the prompt is byte-identical however the caller builds the set — it is
        charged to the context budget and cached by llama-server as a prefix."""
        self.assertEqual(
            effective_prompt(self.BASE, {"fs_write", "memory", "think"}),
            effective_prompt(self.BASE, ["memory", "think", "fs_write"]),
        )

    def test_unknown_feature_names_are_ignored(self):
        self.assertEqual(effective_prompt(self.BASE, {"web_search", "nonsense"}), self.BASE)

    def test_the_users_own_text_is_untouched(self):
        """The distinction from a saved-prompt rewrite: this appends per request and
        never edits what the user typed in the settings panel."""
        edited = "我自己写的提示词，请不要改动它。"
        self.assertTrue(effective_prompt(edited, {"memory"}).startswith(edited + "\n"))

    def test_trailing_whitespace_on_the_base_is_collapsed(self):
        self.assertEqual(effective_prompt(self.BASE + "\n\n  ", {"memory"}),
                         effective_prompt(self.BASE, {"memory"}))

    def test_empty_base_yields_only_the_feature_lines(self):
        self.assertEqual(effective_prompt("", {"think"}), PROMPT_LINES["think"])
        self.assertEqual(effective_prompt("", set()), "")

    def test_fs_read_names_the_roots(self):
        """The reason list_allowed_directories could be dropped: the model is told where
        it may look instead of spending 226 tokens asking."""
        roots = [Path("D:/somewhere"), Path("E:/elsewhere")]
        out = effective_prompt(self.BASE, {"fs_read"}, fs_roots=roots)
        self.assertNotIn("{roots}", out)
        for root in roots:
            self.assertIn(str(root), out)

    def test_fs_read_without_roots_says_so(self):
        out = effective_prompt(self.BASE, {"fs_read"})
        self.assertNotIn("{roots}", out)
        self.assertIn("没有配置允许的目录", out)

    def test_only_fs_read_is_formatted(self):
        """A literal brace in any other line must survive: .format over all of them
        would raise KeyError part way through a chat request."""
        self.assertIn("{", PROMPT_LINES["fs_read"])
        for key, line in PROMPT_LINES.items():
            if key != "fs_read":
                self.assertNotIn("{", line)

    def test_the_real_default_roots_to_the_project(self):
        """The decision behind the whole deny list: an empty MCP_FS_ROOTS means this
        project's own directory, which is where .env lives."""
        self.assertEqual(Settings(_env_file=None).fs_roots, [ROOT])


class TestDenyList(unittest.TestCase):
    """Every case here was measured against the live server before it was written."""

    ROOTS = [ROOT]

    def test_relative_paths_are_refused(self):
        """The regression pin. All four of these returned the real .env, keys included,
        when the list only recognised absolute-looking arguments."""
        for spelling in (".env", "./.env", "app/../.env", ".ENV", "app\\..\\.env"):
            with self.subTest(spelling):
                reason = is_denied(spelling, self.ROOTS)
                self.assertIsNotNone(reason)
                self.assertIn("绝对路径", reason)

    def test_relative_paths_outside_the_deny_list_are_also_refused(self):
        """Not just sensitive ones: where a relative path lands depends on a cwd the
        checker cannot see, so guessing is the thing being refused."""
        for spelling in ("..", ".", "notes.md", "app/config.py", "../history/index.json",
                         "runtime/memory.json"):
            with self.subTest(spelling):
                self.assertIn("绝对路径", is_denied(spelling, self.ROOTS) or "")

    def test_absolute_env_is_refused_in_both_slash_directions(self):
        for spelling in (str(ROOT / ".env"), str(ROOT / ".env").replace("\\", "/")):
            with self.subTest(spelling):
                reason = is_denied(spelling, self.ROOTS)
                self.assertIsNotNone(reason)
                self.assertIn(".env", reason)
                self.assertIn(".env.example", reason)

    def test_env_is_matched_case_insensitively(self):
        for spelling in (".ENV", ".Env", ".eNv"):
            with self.subTest(spelling):
                self.assertIsNotNone(is_denied(str(ROOT / spelling), self.ROOTS))

    def test_env_example_is_allowed(self):
        """The refusal message points the model at it, so refusing it too would send the
        model somewhere that does not work either."""
        self.assertIsNone(is_denied(str(ROOT / ".env.example"), self.ROOTS))
        self.assertIn(".env.example", ENV_ALLOW)

    def test_every_dot_env_variant_is_refused(self):
        for spelling in (".env.local", ".env.production", ".envrc"):
            with self.subTest(spelling):
                self.assertIsNotNone(is_denied(str(ROOT / spelling), self.ROOTS))

    def test_credential_globs(self):
        for name in ("server.key", "cert.pem", "bundle.p12", "bundle.pfx", "id_rsa",
                     "id_rsa.pub", "id_ed25519", "credentials.json", "secrets.yaml"):
            with self.subTest(name):
                self.assertIsNotNone(is_denied(str(ROOT / "docs" / name), self.ROOTS))

    def test_sensitive_directories_at_any_depth(self):
        for segment in sorted(SENSITIVE_DIRS):
            with self.subTest(segment):
                self.assertIsNotNone(is_denied(str(ROOT / segment / "x.txt"), self.ROOTS))
                self.assertIsNotNone(is_denied(str(ROOT / "a" / "b" / segment / "x.txt"), self.ROOTS))

    def test_traversal_cannot_hide_a_sensitive_segment(self):
        self.assertIsNotNone(is_denied(str(ROOT / "app" / ".." / ".env"), self.ROOTS))
        self.assertIsNotNone(is_denied(str(ROOT / "app" / ".." / "runtime" / "m.json"), self.ROOTS))

    def test_a_root_under_a_sensitive_name_is_not_denied_wholesale(self):
        """Root-relative, so pointing MCP_FS_ROOTS at a directory that happens to be
        called node_modules does not refuse its entire tree."""
        root = Path("D:/work/node_modules/my-package")
        self.assertIsNone(is_denied(str(root / "index.js"), [root]))

    def test_outside_every_root_still_gets_the_list(self):
        """The server would refuse this anyway; the list applies because a path we
        cannot place is a path we cannot vouch for."""
        self.assertIsNone(is_denied("C:/Windows/win.ini", self.ROOTS))
        self.assertIsNotNone(is_denied("C:/Windows/system32/.env", self.ROOTS))

    def test_unc_paths_are_recognised_as_absolute(self):
        self.assertIsNotNone(is_denied(r"\\server\share\.env", self.ROOTS))

    def test_blank_paths_are_refused(self):
        for value in ("", "   ", None):
            with self.subTest(repr(value)):
                self.assertIn("路径为空", is_denied(value, self.ROOTS) or "")

    def test_no_roots_refuses_everything_relative_and_checks_the_rest(self):
        self.assertIsNotNone(is_denied(".env", []))
        self.assertIsNotNone(is_denied("C:/x/.env", []))
        self.assertIsNone(is_denied("C:/x/notes.md", []))

    def test_reasons_are_presentable_chinese(self):
        """MCPError's contract: the string goes straight to the browser."""
        for path in (".env", str(ROOT / ".env"), str(ROOT / "runtime" / "x"), ""):
            with self.subTest(path):
                reason = is_denied(path, self.ROOTS)
                self.assertTrue(reason)
                self.assertNotIn("Traceback", reason)
                self.assertEqual(reason, reason.strip())


class TestPathsIn(unittest.TestCase):
    def test_location_keys_are_collected_whatever_they_look_like(self):
        self.assertEqual(_paths_in({"path": ".env"}), [".env"])
        self.assertEqual(_paths_in({"source": ".env"}), [".env"])
        self.assertEqual(_paths_in({"destination": "x"}), ["x"])

    def test_paths_arrays_are_collected(self):
        self.assertEqual(
            _paths_in({"paths": ["a.env", "D:/p/b.txt"]}),
            ["a.env", "D:/p/b.txt"],
        )

    def test_content_keys_are_skipped_even_when_absolute(self):
        """Writing a file that mentions a path is not reaching for that path."""
        for key in ("content", "text", "oldText", "newText", "pattern", "query"):
            with self.subTest(key):
                self.assertEqual(_paths_in({key: "D:/p/.env"}), [])

    def test_write_file_only_yields_the_path(self):
        args = {"path": "D:/p/notes.md", "content": "see D:/p/other.md for details"}
        self.assertEqual(_paths_in(args), ["D:/p/notes.md"])

    def test_edit_file_skips_the_edit_bodies(self):
        args = {
            "path": "D:/p/a.py",
            "edits": [{"oldText": "D:/p/b.py", "newText": "D:/p/c.py"}],
            "dryRun": False,
        }
        self.assertEqual(_paths_in(args), ["D:/p/a.py"])

    def test_exclude_patterns_are_not_collected(self):
        """The exemption that makes "collect every string" survivable: node_modules is
        on the sensitive list, and excluding it is the legitimate thing to ask for."""
        args = {"path": "D:/p", "excludePatterns": ["node_modules", "__pycache__"]}
        self.assertEqual(_paths_in(args), ["D:/p"])

    def test_every_option_key_is_exempt(self):
        for key in sorted(_OPTION_KEYS):
            with self.subTest(key):
                self.assertEqual(_paths_in({"path": "D:/p", key: "node_modules"}), ["D:/p"])

    def test_an_unknown_key_is_collected_whatever_it_looks_like(self):
        """Fail closed, for a parameter upstream adds next release.

        This used to be two rules with a hole between them: a known key collected any
        value, an unknown key collected only absolute-looking ones, and an unknown key
        with a relative value was seen by neither. That hole was not theoretical — the
        server resolves a relative argument against its allowed roots, so the value
        would have reached a real file.
        """
        self.assertEqual(_paths_in({"target": "D:/p/.env"}), ["D:/p/.env"])
        self.assertEqual(_paths_in({"target": ".env"}), [".env"])
        self.assertEqual(_paths_in({"whatever": ["../x"]}), ["../x"])

    def test_an_unknown_key_still_reaches_the_deny_list(self):
        """The point of collecting it: is_denied has to be the thing that sees it."""
        self.assertIsNotNone(is_denied(_paths_in({"target": ".env"})[0], [Path("D:/p")]))

    def test_keys_are_matched_case_insensitively(self):
        self.assertEqual(_paths_in({"Path": ".env", "CONTENT": "D:/x"}), [".env"])

    def test_nested_structures_are_walked(self):
        args = {"path": "D:/a", "options": {"path": "D:/b", "note": "D:/c"}}
        self.assertEqual(_paths_in(args), ["D:/a", "D:/b", "D:/c"])

    def test_empty_and_missing_arguments(self):
        self.assertEqual(_paths_in({}), [])
        self.assertEqual(_paths_in(None), [])

    def test_non_string_locations_are_ignored(self):
        self.assertEqual(_paths_in({"path": 7, "head": 2}), [])

    def test_every_upstream_location_parameter_is_covered(self):
        """The enumeration behind _LOCATION_KEYS, restated as a check.

        _paths_in no longer reads this set — it collects every non-exempt string — so
        what the set is for now is exactly this: the list the two exemptions are proved
        disjoint from. A location parameter that ended up in _CONTENT_KEYS or
        _OPTION_KEYS would be skipped, and the deny list would silently stop covering it.
        """
        self.assertEqual(_LOCATION_KEYS, UPSTREAM_LOCATION_PARAMS)
        for key in sorted(_LOCATION_KEYS):
            with self.subTest(key):
                self.assertEqual(_paths_in({key: ".env"}), [".env"])
        self.assertEqual(_LOCATION_KEYS & _CONTENT_KEYS, frozenset())
        self.assertEqual(_LOCATION_KEYS & _OPTION_KEYS, frozenset())
        self.assertEqual(_CONTENT_KEYS & _OPTION_KEYS, frozenset())

    def test_the_three_buckets_cover_every_upstream_parameter(self):
        """Offline twin of test_mcp.py's live check, and the one that still runs on a
        machine with no Node.js. Nothing may fall outside the buckets unnoticed."""
        self.assertLessEqual(
            UPSTREAM_PARAMS, _LOCATION_KEYS | _CONTENT_KEYS | _OPTION_KEYS
        )


class TestClip(unittest.TestCase):
    def test_short_text_is_untouched(self):
        self.assertEqual(clip("短结果"), "短结果")

    def test_exactly_at_the_cap_is_untouched(self):
        text = "x" * MAX_TOOL_RESULT_CHARS
        self.assertEqual(clip(text), text)

    def test_one_over_is_clipped_and_explained(self):
        text = "y" * (MAX_TOOL_RESULT_CHARS + 1)
        out = clip(text)
        self.assertTrue(out.startswith("y" * MAX_TOOL_RESULT_CHARS))
        self.assertIn("已截断", out)
        self.assertIn(str(len(text)), out)
        self.assertIn(str(MAX_TOOL_RESULT_CHARS), out)

    def test_the_note_points_at_a_narrower_retry(self):
        """A clipped 33,762-line tree is a fragment; the useful response is a smaller
        question, so the note names the two tools that can ask one."""
        out = clip("z" * (MAX_TOOL_RESULT_CHARS * 2))
        self.assertIn("list_directory", out)
        self.assertIn("read_text_file", out)


class TestToolRunnerBase(unittest.IsolatedAsyncioTestCase):
    READ = ("list_directory", "read_text_file", "read_multiple_files")
    WRITE = ("write_file", "move_file")
    FS = READ + WRITE

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name)
        self.root = base / "proj"
        self.root.mkdir()
        self.memory_file = base / "runtime" / "memory.json"

    def runner(self, host, *, names=None, write=False):
        offered = list(names if names is not None else (*self.READ, *(self.WRITE if write else ())))
        # Intersected rather than taken whole: ToolRunner tests fs_tools BEFORE the
        # native names, so a helper that put "remember" in there would dispatch it down
        # the filesystem branch and every memory test would pass for the wrong reason.
        fs = set(offered) & set(self.FS)
        return ToolRunner(
            _Settings([self.root]),
            host,
            self.memory_file,
            _offer(*offered),
            fs_tools=fs,
            fs_write_tools=set(self.WRITE) & fs if write else set(),
        )

    def absolute(self, *parts):
        return str(self.root.joinpath(*parts))


class TestToolRunnerDenials(TestToolRunnerBase):
    async def test_a_relative_env_never_reaches_the_server(self):
        """The leak, end to end: four spellings, one host, zero calls."""
        for spelling in (".env", "./.env", "app/../.env", ".ENV"):
            with self.subTest(spelling):
                host = _FakeHost(text="BOCHA_API_KEY=should-never-appear")
                events = await _collect(self.runner(host), "read_text_file", {"path": spelling})
                self.assertEqual(host.calls, [])
                self.assertIn("绝对路径", _result(events))

    async def test_an_absolute_env_never_reaches_the_server(self):
        target = self.absolute(".env")
        host = _FakeHost(text="BOCHA_API_KEY=should-never-appear")
        events = await _collect(self.runner(host), "read_text_file", {"path": target})
        self.assertEqual(host.calls, [])
        self.assertIn("API 密钥", _result(events))

    async def test_a_sensitive_directory_never_reaches_the_server(self):
        host = _FakeHost()
        await _collect(self.runner(host), "list_directory", {"path": self.absolute("runtime")})
        self.assertEqual(host.calls, [])

    async def test_an_allowed_path_does_reach_the_server(self):
        """The positive control: without it every test above would pass on a runner
        that simply never calls anything."""
        host = _FakeHost(text="[FILE] config.py")
        target = self.absolute("app")
        events = await _collect(self.runner(host), "list_directory", {"path": target})
        self.assertEqual(host.calls, [("list_directory", {"path": target})])
        self.assertEqual(_result(events), "[FILE] config.py")

    async def test_one_denied_path_in_a_batch_denies_the_whole_call(self):
        """A batch is refused whole rather than filtered: silently dropping one path
        would return a result the model believes is complete."""
        host = _FakeHost()
        args = {"paths": [self.absolute("requirements.txt"), ".env"]}
        events = await _collect(self.runner(host), "read_multiple_files", args)
        self.assertEqual(host.calls, [])
        self.assertIn("不允许访问", _result(events))

    async def test_a_denied_move_is_refused_before_it_can_delete(self):
        host = _FakeHost()
        events = await _collect(
            self.runner(host, write=True),
            "move_file",
            {"source": ".env", "destination": self.absolute("runtime", "stolen.txt")},
        )
        self.assertEqual(host.calls, [])
        self.assertIn("绝对路径", _result(events))

    async def test_a_denial_emits_a_status_line_first(self):
        host = _FakeHost()
        events = await _collect(self.runner(host), "read_text_file", {"path": ".env"})
        self.assertEqual([e["type"] for e in events], ["status", "tool_result"])
        self.assertIn("已拒绝访问", events[0]["text"])

    async def test_a_denied_write_is_logged(self):
        """The audit trail: a checkbox is not one, and every write names its target.
        Compared by base name because the log renders the path list with str(), which
        doubles backslashes on Windows."""
        host = _FakeHost()
        with self.assertLogs("app.tools", level="WARNING") as caught:
            await _collect(self.runner(host, write=True), "write_file",
                           {"path": self.absolute("notes.md"), "content": "hi"})
        self.assertTrue(any("mcp write via write_file" in line for line in caught.output),
                        caught.output)
        self.assertTrue(any("notes.md" in line for line in caught.output), caught.output)

    async def test_a_denied_write_is_logged_even_when_refused(self):
        host = _FakeHost()
        with self.assertLogs("app.tools", level="WARNING") as caught:
            await _collect(self.runner(host, write=True), "write_file",
                           {"path": ".env", "content": "hi"})
        self.assertTrue(any("denied write_file" in line for line in caught.output), caught.output)
        self.assertEqual(host.calls, [])


class TestToolRunnerBudget(TestToolRunnerBase):
    async def test_a_result_is_clipped_to_the_per_result_cap(self):
        host = _FakeHost(text="x" * (MAX_TOOL_RESULT_CHARS * 3))
        events = await _collect(self.runner(host), "read_text_file", {"path": self.absolute("a.txt")})
        out = _result(events)
        self.assertIn("已截断", out)
        self.assertLess(len(out), MAX_TOOL_RESULT_CHARS + 200)

    async def test_the_total_cap_stops_further_calls(self):
        """Tool results never pass through fit_budget, which trims only the incoming
        history — this cap is the only bound there is."""
        host = _FakeHost(text="x" * MAX_TOOL_RESULT_CHARS)
        runner = self.runner(host)
        target = self.absolute("a.txt")
        for _ in range(MAX_TOOL_TOTAL_CHARS // MAX_TOOL_RESULT_CHARS):
            await _collect(runner, "read_text_file", {"path": target})
        self.assertEqual(len(host.calls), MAX_TOOL_TOTAL_CHARS // MAX_TOOL_RESULT_CHARS)

        events = await _collect(runner, "read_text_file", {"path": target})
        self.assertEqual(len(host.calls), MAX_TOOL_TOTAL_CHARS // MAX_TOOL_RESULT_CHARS)
        self.assertIn("预算已用完", _result(events))
        self.assertIn(str(MAX_TOOL_TOTAL_CHARS), _result(events))

    async def test_native_results_count_against_the_same_cap(self):
        """They are appended to the same payload, so charging only the file tools would
        leave a long remember/recall chain unbounded — the exact hole the cap exists to
        close. A think step is a sentence, so in practice it barely moves the total."""
        host = _FakeHost(text="x" * MAX_TOOL_RESULT_CHARS)
        runner = self.runner(host, names=(*self.READ, "remember", "recall", "think"))
        await _collect(runner, "read_text_file", {"path": self.absolute("a.txt")})
        self.assertGreater(runner._spent, 0)
        events = await _collect(runner, "think", {"thought": "第一步"})
        self.assertIn("已记录第 1 步", _result(events))
        self.assertLess(runner._spent, MAX_TOOL_TOTAL_CHARS)

    async def test_a_small_result_leaves_the_budget_available(self):
        host = _FakeHost(text="ok")
        runner = self.runner(host)
        for _ in range(20):
            await _collect(runner, "read_text_file", {"path": self.absolute("a.txt")})
        self.assertEqual(len(host.calls), 20)


class TestToolRunnerThink(TestToolRunnerBase):
    def runner_with_think(self):
        return self.runner(_FakeHost(), names=("think",))

    async def test_emits_reasoning_before_the_result(self):
        """A `reasoning` event rather than a new type: the browser already renders those
        into the 「思考过程」 block and already persists them, so the steps survive a
        refresh without any front-end change."""
        events = await _collect(self.runner_with_think(), "think", {"thought": "先分三步"})
        self.assertEqual([e["type"] for e in events], ["reasoning", "tool_result"])
        self.assertIn("先分三步", events[0]["text"])
        self.assertIn("第 1/%d 步" % MAX_THOUGHTS, events[0]["text"])

    async def test_steps_are_numbered_in_order(self):
        runner = self.runner_with_think()
        for step in range(1, 4):
            events = await _collect(runner, "think", {"thought": f"step {step}"})
            self.assertIn(f"第 {step}/{MAX_THOUGHTS} 步", events[0]["text"])

    async def test_the_result_says_how_many_steps_are_left(self):
        events = await _collect(self.runner_with_think(), "think", {"thought": "x"})
        self.assertIn(f"还可以再想 {MAX_THOUGHTS - 1} 步", _result(events))

    async def test_the_cap_stops_a_model_that_would_only_think(self):
        """think spends a tool round per step, and MAX_TOOL_ROUNDS is 4 — the cap exists
        so thinking cannot consume every round and leave nothing for an answer."""
        runner = self.runner_with_think()
        for _ in range(MAX_THOUGHTS):
            await _collect(runner, "think", {"thought": "x"})
        events = await _collect(runner, "think", {"thought": "x"})
        self.assertEqual([e["type"] for e in events], ["tool_result"])
        self.assertIn(f"已达上限（{MAX_THOUGHTS} 步）", _result(events))

    async def test_a_blank_thought_is_answered_not_stored(self):
        runner = self.runner_with_think()
        events = await _collect(runner, "think", {"thought": "   "})
        self.assertEqual([e["type"] for e in events], ["tool_result"])
        self.assertIn("thought", _result(events))
        events = await _collect(runner, "think", {"thought": "real"})
        self.assertIn("第 1/", events[0]["text"])


class TestToolRunnerMemory(TestToolRunnerBase):
    def runner_with_memory(self, host=None):
        return self.runner(host or _FakeHost(), names=("remember", "recall"))

    async def test_remember_writes_to_the_store_and_reports_the_kind(self):
        runner = self.runner_with_memory()
        events = await _collect(runner, "remember", {"text": "用户偏好深色主题", "kind": "preference"})
        self.assertIn("已记住（preference）", _result(events))
        self.assertTrue(self.memory_file.is_file())
        stored = json.loads(self.memory_file.read_text(encoding="utf-8"))
        self.assertEqual([r["text"] for r in stored], ["用户偏好深色主题"])

    async def test_an_unknown_kind_degrades_rather_than_losing_the_memory(self):
        events = await _collect(self.runner_with_memory(), "remember", {"text": "x", "kind": "note"})
        self.assertIn("已记住（fact）", _result(events))

    async def test_a_blank_remember_is_reported_as_a_failure(self):
        """An unwritten note must never be reported as saved."""
        events = await _collect(self.runner_with_memory(), "remember", {"text": "  "})
        self.assertIn("没能记住", _result(events))
        self.assertFalse(self.memory_file.exists())

    async def test_recall_finds_what_remember_stored(self):
        runner = self.runner_with_memory()
        await _collect(runner, "remember", {"text": "导出格式默认用 PDF", "kind": "preference"})
        events = await _collect(runner, "recall", {"query": "导出 格式"})
        self.assertIn("导出格式默认用 PDF", _result(events))

    async def test_recall_warns_that_memories_may_be_stale(self):
        runner = self.runner_with_memory()
        await _collect(runner, "remember", {"text": "导出格式默认用 PDF"})
        events = await _collect(runner, "recall", {"query": "导出"})
        self.assertIn("可能已过时", _result(events))

    async def test_recall_returns_at_most_the_limit(self):
        runner = self.runner_with_memory()
        for index in range(RECALL_LIMIT + 4):
            await _collect(runner, "remember", {"text": f"关于导出的第 {index} 条"})
        events = await _collect(runner, "recall", {"query": "导出"})
        self.assertEqual(_result(events).count("关于导出的第"), RECALL_LIMIT)

    async def test_recall_without_a_query_asks_for_one(self):
        events = await _collect(self.runner_with_memory(), "recall", {"query": ""})
        self.assertIn("需要一个关键词", _result(events))

    async def test_recall_with_no_hits_says_so_and_invites_a_retry(self):
        """Keyword scoring, not semantic search: a miss is a normal outcome and the
        answer has to say what to do about it rather than return nothing."""
        runner = self.runner_with_memory()
        await _collect(runner, "remember", {"text": "用户偏好深色主题"})
        events = await _collect(runner, "recall", {"query": "量子纠缠"})
        self.assertIn("没有找到", _result(events))
        self.assertIn("换一种说法", _result(events))

    async def test_a_status_line_precedes_each_memory_call(self):
        runner = self.runner_with_memory()
        for name, args, expected in (
            ("remember", {"text": "abc"}, "正在记住"),
            ("recall", {"query": "abc"}, "正在查记忆"),
        ):
            with self.subTest(name):
                events = await _collect(runner, name, args)
                self.assertEqual(events[0]["type"], "status")
                self.assertIn(expected, events[0]["text"])

    async def test_the_status_line_clips_a_long_memory(self):
        events = await _collect(self.runner_with_memory(), "remember", {"text": "长" * 500})
        self.assertLessEqual(len(events[0]["text"]), len("正在记住：") + 40)


class TestToolRunnerDispatch(TestToolRunnerBase):
    async def test_an_unknown_tool_names_the_ones_on_offer(self):
        """A model that hallucinates a name gets a usable answer rather than silence."""
        runner = self.runner(_FakeHost())
        events = await _collect(runner, "web_search", {"query": "x"})
        out = _result(events)
        self.assertIn("没有名为 web_search 的工具", out)
        for name in self.READ:
            self.assertIn(name, out)

    async def test_a_hallucinated_search_with_the_box_unticked_does_not_search(self):
        """Why llm.py gates both search branches on search_on: the schemas go out as soon
        as ANY capability is on, so an unticked 联网检索 must not be reachable by name."""
        runner = self.runner(_FakeHost(), names=("recall",))
        self.assertIn("没有名为 web_search", _result(await _collect(runner, "web_search", {})))

    async def test_no_tools_at_all_says_so(self):
        runner = ToolRunner(_Settings([self.root]), _FakeHost(), self.memory_file, [])
        self.assertIn("本次没有任何可用工具", _result(await _collect(runner, "think", {})))

    async def test_an_offered_but_unimplemented_tool_is_reported_not_silent(self):
        runner = self.runner(_FakeHost(), names=("read_text_file", "some_new_thing"))
        with self.assertLogs("app.tools", level="WARNING"):
            events = await _collect(runner, "some_new_thing", {})
        self.assertIn("没有执行成功", _result(events))

    async def test_a_file_tool_without_a_host_says_why(self):
        runner = ToolRunner(
            _Settings([self.root]), None, self.memory_file,
            _offer("list_directory"), fs_tools={"list_directory"},
        )
        self.assertIn("MCP 未启用", _result(await _collect(runner, "list_directory",
                                                           {"path": self.absolute("app")})))

    async def test_a_server_failure_costs_one_call_not_the_request(self):
        host = _FakeHost(error=RuntimeError("child died"))
        events = await _collect(self.runner(host), "read_text_file", {"path": self.absolute("a")})
        self.assertIn("文件工具调用失败", _result(events))

    async def test_an_error_message_is_passed_through_verbatim(self):
        """The server's own "Access denied - path outside allowed directories" is what
        the model needs to read in order to change course."""
        host = _FakeHost(error=RuntimeError("Access denied - path outside allowed directories"))
        events = await _collect(self.runner(host), "read_text_file", {"path": self.absolute("a")})
        self.assertIn("Access denied - path outside allowed directories", _result(events))

    async def test_a_read_status_line_names_the_target(self):
        host = _FakeHost()
        target = self.absolute("app")
        events = await _collect(self.runner(host), "list_directory", {"path": target})
        self.assertEqual(events[0]["type"], "status")
        self.assertIn(target[:90], events[0]["text"])

    async def test_a_write_status_line_says_it_is_writing(self):
        host = _FakeHost()
        events = await _collect(self.runner(host, write=True), "write_file",
                                {"path": self.absolute("n.md"), "content": "x"})
        self.assertIn("正在写入", events[0]["text"])

    async def test_missing_arguments_do_not_raise(self):
        for name, args in (("read_text_file", {}), ("read_text_file", None),
                           ("remember", {}), ("recall", {}), ("think", {})):
            with self.subTest(f"{name} {args}"):
                runner = self.runner(_FakeHost(), names=(*self.READ, "remember", "recall", "think"))
                events = await _collect(runner, name, args)
                self.assertEqual(len([e for e in events if e["type"] == "tool_result"]), 1)


if __name__ == "__main__":
    unittest.main()
