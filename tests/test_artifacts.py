"""save_document: the tool, the filename sanitiser, the archive field, and two truncation bugs.

Standard library only, by decision — pytest was explicitly rejected as a new
dependency. Run from the project root:

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m unittest discover -s tests -v

Three groups here pin things that would otherwise fail silently in front of a user.

TestTruncation is the most important one in the file, and it pins two bugs that predate
this feature. `_parse_args` used to return {} for both "the model sent no arguments" and
"the arguments are half a JSON document because max_tokens cut them off", so a truncated
save_document executed with an empty content and nothing happened. And stream_chat's break
on `finish_reason != "tool_calls"` discarded the buffered call entirely when generation hit
the limit — llama-server reports "length" there — which, on a turn that only called a tool,
produced a completely blank reply. Both are extracted as pure functions so they can be
tested without faking a streaming HTTP response.

TestSaveDocumentDispatch pins an arithmetic fact rather than a stylistic one: the
tool_result must stay tiny, because ToolRunner.run charges every result against
MAX_TOOL_TOTAL_CHARS and echoing a 2,400-character document back would spend a tenth of
the request's whole tool budget on text the model wrote one round ago.

TestSafeArtifactName's cases are measured, not imagined. The real session index in
runtime/history/index.json holds a title starting '> **【角色设定】**：…', and both > and *
are illegal in a Windows filename.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from app.config import DEFAULT_SYSTEM_PROMPT, Settings
from app.documents import budget_safety, estimate_tokens
from app.llm import _parse_args, _truncation_note
from app.main import StoredArtifact, StoredMessage
from app.tools import (
    ARTIFACT_FORMATS,
    DOC_GEN_CHAR_HINT,
    MAX_ARTIFACT_CHARS,
    MAX_ARTIFACT_NAME,
    PROMPT_LINES,
    SAVE_DOCUMENT,
    SAVE_DOCUMENT_TOOL,
    ToolRunner,
    build_tools,
    effective_prompt,
    safe_artifact_name,
    unwrap_markdown_fence,
)


def _tokens(tool) -> int:
    return estimate_tokens(json.dumps(tool, ensure_ascii=False))


def _offer(*names):
    """A tool list shaped the way build_tools returns it, without the schemas."""
    return [
        {"type": "function", "function": {"name": n, "description": "", "parameters": {}}}
        for n in names
    ]


class _Settings:
    """ToolRunner reads exactly one attribute off settings, so that is all this is."""

    def __init__(self, roots=()):
        self.fs_roots = [Path(r) for r in roots]


async def _collect(runner, name, args):
    return [event async for event in runner.run(name, args)]


def _result(events) -> str:
    found = [e["content"] for e in events if e.get("type") == "tool_result"]
    assert len(found) == 1, f"expected exactly one tool_result, got {events}"
    return found[0]


def _artifacts(events) -> list[dict]:
    return [e for e in events if e.get("type") == "artifact"]


# A document shaped like the ones this feature actually produces: headings, bullets, a
# table and a code fence. Pinned as the sample behind the character budget because the
# density of a report is nothing like the density of prose — the table and the fence are
# what pull it down to 1.045 chars/token.
SAMPLE = """# 季度复盘

## 一、总体结论

本季度线上转化率提升 **1.8 个百分点**，主要来自落地页首屏改版。

- 首屏改版：跳出率从 42% 降到 35%
- 检索词收敛：无效点击减少约 1/4

## 二、数据明细

| 指标 | 上季度 | 本季度 | 变化 |
| --- | --- | --- | --- |
| 访问量 | 128,400 | 141,900 | +10.5% |
| 转化率 | 3.2% | 5.0% | +1.8pt |

## 三、下一步

```python
rate = conv / visits
```
"""


class TestSaveDocumentSchema(unittest.TestCase):
    def test_the_schema_costs_what_it_was_measured_at(self):
        """504 tokens, down from 591 with the behavioural guidance in it.

        Pinned because the schema is paid on every single request once the box is
        ticked, and the guidance that pushed it to 591 duplicates a line the prompt
        carries anyway.
        """
        self.assertEqual(_tokens(SAVE_DOCUMENT_TOOL), 504)

    def test_the_enum_is_exactly_the_five_formats_the_browser_can_render(self):
        props = SAVE_DOCUMENT_TOOL["function"]["parameters"]["properties"]
        self.assertEqual(props["format"]["enum"], list(ARTIFACT_FORMATS))
        self.assertEqual(ARTIFACT_FORMATS, ("md", "html", "csv", "pdf", "docx"))

    def test_content_says_markdown_and_not_one_word_per_format(self):
        """All five formats take Markdown source, so the schema must not branch on format.

        That single semantic is what makes this feature free: every export path in
        static/app.js already consumes Markdown, so no second renderer exists on either
        side. A per-format description here would be the first sign of someone building one.
        """
        props = SAVE_DOCUMENT_TOOL["function"]["parameters"]["properties"]
        self.assertIn("Markdown", props["content"]["description"])
        for fmt in ARTIFACT_FORMATS:
            self.assertNotIn(fmt, props["content"]["description"])

    def test_title_is_optional_because_pdf_metadata_is_not_worth_a_refusal(self):
        params = SAVE_DOCUMENT_TOOL["function"]["parameters"]
        self.assertEqual(params["required"], ["filename", "format", "content"])
        self.assertIn("title", params["properties"])

    def test_filename_asks_for_a_name_and_not_a_path(self):
        props = SAVE_DOCUMENT_TOOL["function"]["parameters"]["properties"]
        self.assertIn("路径", props["filename"]["description"])

    def test_the_literal_in_main_duplicates_the_tuple_in_tools(self):
        """The copy cannot drift: a Literal cannot be built from a tuple at runtime.

        Without this the two lists are the same five strings in two files with nothing
        tying them together, and the failure mode is a 422 on saving a session the user
        has already watched a file arrive for.
        """
        allowed = StoredArtifact.model_fields["format"].annotation.__args__
        self.assertEqual(tuple(allowed), ARTIFACT_FORMATS)


class TestSafeArtifactName(unittest.TestCase):
    def test_traversal_loses_its_separators(self):
        out = safe_artifact_name("../../etc/passwd", "md")
        for sep in ("\\", "/"):
            self.assertNotIn(sep, out)
        self.assertNotIn("..", out)
        self.assertTrue(out.endswith(".md"))

    def test_every_illegal_windows_character_is_replaced(self):
        out = safe_artifact_name('a/b\\c:d*e?f"g<h>i|j', "md")
        for ch in '\\/:*?"<>|':
            self.assertNotIn(ch, out)

    def test_a_control_character_is_replaced(self):
        self.assertNotIn("\x00", safe_artifact_name("a\x00b", "md"))
        self.assertNotIn("\n", safe_artifact_name("a\nb", "md"))

    def test_the_sample_from_the_real_session_index_survives(self):
        """> and * are both illegal on Windows, and this title is in a real archive."""
        out = safe_artifact_name("> **【角色设定】**：一个测试", "pdf")
        for ch in '\\/:*?"<>|':
            self.assertNotIn(ch, out)
        self.assertTrue(out.endswith(".pdf"))
        self.assertLessEqual(len(out), MAX_ARTIFACT_NAME + len(".pdf"))

    def test_a_leading_dot_is_stripped_because_windows_strips_it_silently(self):
        self.assertEqual(safe_artifact_name(".env", "md"), "env.md")

    def test_trailing_spaces_are_stripped(self):
        self.assertEqual(safe_artifact_name("报告   ", "md"), "报告.md")
        self.assertEqual(safe_artifact_name("  报告", "md"), "报告.md")

    def test_dots_are_collapsed_before_the_strip_and_that_order_is_the_point(self):
        """'报告...' lands as '报告_.md', not '报告.md': '..' -> '_' runs first.

        Reversing the two would let a name whose dots survive the strip carry a '..'
        back into the result, and killing the separators is what kills traversal. The
        underscore is an odd filename and nothing more.
        """
        self.assertEqual(safe_artifact_name("报告...", "md"), "报告_.md")
        self.assertEqual(safe_artifact_name("a..b", "md"), "a_b.md")
        self.assertNotIn("..", safe_artifact_name("....", "md"))

    def test_a_reserved_device_name_is_prefixed_by_stem_not_by_whole_string(self):
        """con.md is still a device: Windows resolves by the part before the first dot."""
        self.assertEqual(safe_artifact_name("con", "md"), "_con.md")
        self.assertEqual(safe_artifact_name("con.md", "md"), "_con.md")
        self.assertEqual(safe_artifact_name("NUL.pdf", "pdf"), "_NUL.pdf")
        self.assertEqual(safe_artifact_name("com1", "docx"), "_com1.docx")

    def test_an_ordinary_name_containing_a_reserved_word_is_left_alone(self):
        self.assertEqual(safe_artifact_name("console", "md"), "console.md")
        self.assertEqual(safe_artifact_name("配置", "md"), "配置.md")

    def test_the_length_cap_is_applied_to_the_stem_not_to_the_result(self):
        out = safe_artifact_name("名" * 80, "md")
        self.assertEqual(out, "名" * MAX_ARTIFACT_NAME + ".md")

    def test_an_empty_name_falls_back_to_a_timestamp_with_the_right_extension(self):
        for fmt in ARTIFACT_FORMATS:
            with self.subTest(fmt):
                out = safe_artifact_name("", fmt)
                self.assertTrue(out.startswith("文档-"), out)
                self.assertTrue(out.endswith(f".{fmt}"), out)

    def test_a_name_that_sanitises_to_nothing_also_falls_back(self):
        """'***' and '...' do NOT land here — they leave underscores behind, which are
        odd but legal. Only a bare dot and whitespace reduce to nothing."""
        self.assertTrue(safe_artifact_name(".", "md").startswith("文档-"))
        self.assertTrue(safe_artifact_name("   ", "csv").startswith("文档-"))
        # Control characters are replaced before the strip, so these become underscores
        # rather than nothing — the same ordering the dot test above pins.
        self.assertEqual(safe_artifact_name("\t\n", "pdf"), "__.pdf")

    def test_a_mismatched_extension_is_replaced(self):
        """report.txt that is really a PDF fails confusingly, so the bytes decide."""
        self.assertEqual(safe_artifact_name("report.txt", "pdf"), "report.pdf")

    def test_a_matching_extension_is_not_doubled(self):
        """The bug this pins: the strip used to fire only on a mismatch.

        Measured before the fix — '周报.PDF' with fmt='pdf' came back '周报.PDF.pdf',
        because the suffix was appended unconditionally. Matching now also normalises
        the case, which is why the expected value is lowercase.
        """
        self.assertEqual(safe_artifact_name("周报.pdf", "pdf"), "周报.pdf")
        self.assertEqual(safe_artifact_name("周报.PDF", "pdf"), "周报.pdf")
        self.assertEqual(safe_artifact_name("周报", "pdf"), "周报.pdf")

    def test_an_unknown_format_falls_back_to_md_rather_than_inventing_an_extension(self):
        self.assertEqual(safe_artifact_name("报告", "exe"), "报告.md")
        self.assertEqual(safe_artifact_name("报告", ""), "报告.md")

    def test_a_non_string_name_does_not_raise(self):
        self.assertTrue(safe_artifact_name(None, "md").startswith("文档-"))
        self.assertEqual(safe_artifact_name(123, "md"), "123.md")

    def test_the_result_is_idempotent(self):
        """The browser sanitises again, so running it twice must be a no-op."""
        for raw in ("../../etc/passwd", "con.md", "周报.PDF", "> **【角色设定】**", ""):
            once = safe_artifact_name(raw, "pdf")
            with self.subTest(raw):
                self.assertEqual(safe_artifact_name(once, "pdf"), once)


class TestUnwrapMarkdownFence(unittest.TestCase):
    """Measured on Qwen3-8B-Q4_K_M: three of six save_document calls wrapped the whole
    document in an unclosed ```markdown fence, which renders as one grey code block and
    exports to PDF and DOCX as monospace. The file arrives and is unusable."""

    def test_the_measured_shape_an_opening_fence_and_no_closing_one(self):
        """All three real runs left the fence open, which is the worst variant: an unclosed
        fence turns *everything after it* into code, not just a short quoted block."""
        raw = "```markdown\n# 路线图\n\n## 1. 引言\n\n自动化测试很重要。\n"
        self.assertEqual(unwrap_markdown_fence(raw), "# 路线图\n\n## 1. 引言\n\n自动化测试很重要。\n")

    def test_a_closed_fence_loses_both_ends(self):
        raw = "```markdown\n# 标题\n\n正文。\n```"
        self.assertEqual(unwrap_markdown_fence(raw), "# 标题\n\n正文。")

    def test_a_document_that_was_never_fenced_comes_back_byte_identical(self):
        for raw in ("# 标题\n\n正文。", "普通一句话，没有标题。", ""):
            with self.subTest(repr(raw[:20])):
                self.assertEqual(unwrap_markdown_fence(raw), raw)

    def test_a_bare_fence_is_left_alone(self):
        """Deliberate: unwrapping a fence with no info string would corrupt a document that
        legitimately opens with a code block. No model here has been seen to use the bare
        form for the whole document, so the ambiguity is resolved by not touching it."""
        raw = "```\nprint(1)\n```\n"
        self.assertEqual(unwrap_markdown_fence(raw), raw)

    def test_a_code_block_inside_the_document_survives(self):
        """One of the measured runs fenced the document *and* carried a ```python example
        inside it. Only the outer wrapper may go."""
        raw = "```markdown\n# 手册\n\n```python\nprint('hi')\n```\n\n## 结语\n"
        out = unwrap_markdown_fence(raw)
        self.assertTrue(out.startswith("# 手册"))
        self.assertIn("```python\nprint('hi')\n```", out)

    def test_a_document_ending_in_a_code_block_keeps_that_blocks_own_fence(self):
        """The shape that broke the first implementation, which decided by presence rather
        than parity and ate the tail of any report ending in a code block.

        Two markers inside means both are already paired, so the last one belongs to the
        document. One of the measured runs produced exactly this: 下一步 was a ```python
        block and it was the final section.
        """
        raw = "```markdown\n# 手册\n\n## 下一步\n\n```python\nrate = conv / visits\n```\n"
        out = unwrap_markdown_fence(raw)
        self.assertTrue(out.startswith("# 手册"))
        self.assertTrue(out.rstrip().endswith("```python\nrate = conv / visits\n```"))

    def test_an_unpaired_trailing_fence_is_the_wrappers_even_with_a_block_inside(self):
        """The mirror case: three markers, so one is unpaired and can only be the wrapper's
        close. It goes; the internal block stays."""
        raw = "```markdown\n# 手册\n\n```python\nx = 1\n```\n\n## 结语\n\n写完。\n```"
        out = unwrap_markdown_fence(raw)
        self.assertIn("```python\nx = 1\n```", out)
        self.assertTrue(out.endswith("写完。"))

    def test_a_body_that_is_only_a_fence_becomes_empty(self):
        """So _save_document's `not content.strip()` check fires and the model is told the
        document was empty, instead of delivering a file containing nothing."""
        self.assertEqual(unwrap_markdown_fence("```markdown\n```").strip(), "")
        self.assertEqual(unwrap_markdown_fence("```md\n\n").strip(), "")

    def test_the_info_string_tolerates_case_spaces_and_crlf(self):
        for raw in (
            "```MARKDOWN\n# A\n",
            "``` md\n# A\n",
            "```markdown\t\n# A\n",
            "```markdown\r\n# A\r\n",
        ):
            with self.subTest(repr(raw[:16])):
                self.assertTrue(unwrap_markdown_fence(raw).lstrip("\r").startswith("# A"))

    def test_blank_lines_between_the_fence_and_the_document_are_dropped(self):
        self.assertEqual(unwrap_markdown_fence("```markdown\n\n\n# A\n"), "# A\n")

    def test_it_is_idempotent(self):
        """A re-download from the archive runs the stored text through the browser, not
        through here again, but a double strip must still be a no-op."""
        for raw in ("```markdown\n# A\n```", "# A\n", "```\nx\n```"):
            once = unwrap_markdown_fence(raw)
            with self.subTest(repr(raw[:16])):
                self.assertEqual(unwrap_markdown_fence(once), once)


class TestSaveDocumentDispatch(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.memory_file = Path(self._tmp.name) / "memory.json"

    def runner(self):
        # No MCP host and no filesystem tools: save_document is native, so a runner with
        # neither is the honest shape of a request where only 生成文档 is ticked.
        return ToolRunner(_Settings(), None, self.memory_file, _offer(SAVE_DOCUMENT))

    async def test_each_format_emits_exactly_one_artifact_and_one_result(self):
        for fmt in ARTIFACT_FORMATS:
            with self.subTest(fmt):
                events = await _collect(
                    self.runner(),
                    SAVE_DOCUMENT,
                    {"filename": f"报告.{fmt}", "format": fmt, "content": SAMPLE},
                )
                arts = _artifacts(events)
                self.assertEqual(len(arts), 1)
                self.assertEqual(arts[0]["format"], fmt)
                self.assertEqual(arts[0]["filename"], f"报告.{fmt}")
                self.assertEqual(arts[0]["content"], SAMPLE)
                self.assertEqual(arts[0]["chars"], len(SAMPLE))
                self.assertEqual(len([e for e in events if e["type"] == "tool_result"]), 1)

    async def test_the_artifact_event_arrives_before_the_result(self):
        """The browser downloads on arrival, so the file must not wait for the round."""
        events = await _collect(
            self.runner(), SAVE_DOCUMENT, {"filename": "a", "format": "md", "content": SAMPLE}
        )
        kinds = [e["type"] for e in events]
        self.assertEqual(kinds, ["artifact", "tool_result"])

    async def test_a_fenced_body_reaches_the_browser_unwrapped(self):
        """End to end, not just the helper: the event is what the browser renders and what
        StoredArtifact archives, so the strip has to have happened by here.

        chars counts the unwrapped text too -- the chip on the card and the archived copy
        would otherwise both report a length the rendered document does not have.
        """
        fenced = "```markdown\n" + SAMPLE
        events = await _collect(
            self.runner(), SAVE_DOCUMENT, {"filename": "报告", "format": "md", "content": fenced}
        )
        arts = _artifacts(events)
        self.assertEqual(len(arts), 1)
        self.assertEqual(arts[0]["content"], SAMPLE)
        self.assertEqual(arts[0]["chars"], len(SAMPLE))

    async def test_the_result_is_small_enough_not_to_eat_the_tool_budget(self):
        """Arithmetic, not taste: run() charges len(result) against MAX_TOOL_TOTAL_CHARS.

        Echoing the document back would spend a tenth of the request's entire budget on
        text the model wrote one round ago and still has in its own assistant turn.
        """
        events = await _collect(
            self.runner(), SAVE_DOCUMENT, {"filename": "报告", "format": "pdf", "content": SAMPLE}
        )
        result = _result(events)
        self.assertLess(len(result), 200)
        runner = self.runner()
        before = runner._spent
        await _collect(runner, SAVE_DOCUMENT, {"filename": "报告", "format": "pdf", "content": SAMPLE})
        self.assertLess(runner._spent - before, 200)

    async def test_the_result_names_the_file_the_user_received(self):
        events = await _collect(
            self.runner(), SAVE_DOCUMENT, {"filename": "报告", "format": "pdf", "content": SAMPLE}
        )
        self.assertIn("报告.pdf", _result(events))

    async def test_the_title_is_carried_but_capped_where_the_archive_caps_it(self):
        """120 is StoredArtifact.title's max_length: a longer one must not turn
        "save this session" into a 422 after the user already has the file."""
        events = await _collect(
            self.runner(),
            SAVE_DOCUMENT,
            {"filename": "a", "format": "md", "content": SAMPLE, "title": "标" * 300},
        )
        self.assertEqual(len(_artifacts(events)[0]["title"]), 120)

    async def test_a_missing_title_is_empty_rather_than_absent(self):
        events = await _collect(
            self.runner(), SAVE_DOCUMENT, {"filename": "a", "format": "md", "content": SAMPLE}
        )
        self.assertEqual(_artifacts(events)[0]["title"], "")

    async def test_an_uppercase_format_is_accepted(self):
        """Models do this, and refusing it would cost a round to teach them not to."""
        events = await _collect(
            self.runner(), SAVE_DOCUMENT, {"filename": "a", "format": "PDF", "content": SAMPLE}
        )
        self.assertEqual(_artifacts(events)[0]["format"], "pdf")

    async def test_an_unknown_format_is_refused_without_an_artifact(self):
        events = await _collect(
            self.runner(), SAVE_DOCUMENT, {"filename": "a", "format": "exe", "content": SAMPLE}
        )
        self.assertEqual(_artifacts(events), [])
        self.assertIn("exe", _result(events))
        self.assertIn("、".join(ARTIFACT_FORMATS), _result(events))

    async def test_an_empty_content_is_refused_without_an_artifact(self):
        for content in ("", "   ", "\n\t"):
            with self.subTest(repr(content)):
                events = await _collect(
                    self.runner(), SAVE_DOCUMENT, {"filename": "a", "format": "md", "content": content}
                )
                self.assertEqual(_artifacts(events), [])
                self.assertIn("content", _result(events))

    async def test_a_document_over_the_cap_is_refused_without_an_artifact(self):
        """MAX_ARTIFACT_CHARS bounds two things .env cannot: one SSE event, one archive."""
        events = await _collect(
            self.runner(),
            SAVE_DOCUMENT,
            {"filename": "a", "format": "md", "content": "字" * (MAX_ARTIFACT_CHARS + 1)},
        )
        self.assertEqual(_artifacts(events), [])
        result = _result(events)
        self.assertIn(str(MAX_ARTIFACT_CHARS), result)
        self.assertIn(str(MAX_ARTIFACT_CHARS + 1), result)

    async def test_a_document_exactly_at_the_cap_is_accepted(self):
        events = await _collect(
            self.runner(),
            SAVE_DOCUMENT,
            {"filename": "a", "format": "md", "content": "字" * MAX_ARTIFACT_CHARS},
        )
        self.assertEqual(len(_artifacts(events)), 1)

    async def test_a_malicious_filename_is_sanitised_before_it_reaches_the_browser(self):
        events = await _collect(
            self.runner(),
            SAVE_DOCUMENT,
            {"filename": "../../etc/passwd", "format": "md", "content": SAMPLE},
        )
        name = _artifacts(events)[0]["filename"]
        self.assertNotIn("/", name)
        self.assertNotIn("..", name)

    async def test_a_missing_filename_still_produces_a_downloadable_name(self):
        events = await _collect(
            self.runner(), SAVE_DOCUMENT, {"format": "md", "content": SAMPLE}
        )
        self.assertTrue(_artifacts(events)[0]["filename"].startswith("文档-"))

    async def test_two_documents_in_one_request_both_arrive(self):
        """The prompt says one per call, but the model decides, and MAX_TOOL_ROUNDS=4
        means it can call again. Both must reach the browser."""
        runner = self.runner()
        first = await _collect(runner, SAVE_DOCUMENT, {"filename": "一", "format": "md", "content": SAMPLE})
        second = await _collect(runner, SAVE_DOCUMENT, {"filename": "二", "format": "csv", "content": SAMPLE})
        self.assertEqual(_artifacts(first)[0]["filename"], "一.md")
        self.assertEqual(_artifacts(second)[0]["filename"], "二.csv")

    async def test_no_status_line_is_emitted(self):
        """Unlike _remember there is nothing to await, and a 「正在生成文档」 line would
        still be sitting in the bubble after the download finished."""
        events = await _collect(
            self.runner(), SAVE_DOCUMENT, {"filename": "a", "format": "md", "content": SAMPLE}
        )
        self.assertEqual([e["type"] for e in events if e["type"] == "status"], [])

    async def test_save_document_needs_no_mcp_host(self):
        """It is not a filesystem tool, so a request with 文件 unticked must still work."""
        runner = ToolRunner(_Settings(), None, self.memory_file, _offer(SAVE_DOCUMENT))
        events = await _collect(runner, SAVE_DOCUMENT, {"filename": "a", "format": "md", "content": SAMPLE})
        self.assertEqual(len(_artifacts(events)), 1)


class TestBuildToolsDocGen(unittest.TestCase):
    def test_the_box_off_costs_nothing_and_offers_nothing(self):
        tools = build_tools(search=True)
        self.assertEqual(len(tools), 2)
        self.assertEqual(_tokens(tools), 807)
        self.assertNotIn(SAVE_DOCUMENT, [t["function"]["name"] for t in tools])

    def test_the_box_on_adds_one_tool_at_the_measured_price(self):
        tools = build_tools(search=True, doc_gen=True)
        self.assertEqual(len(tools), 3)
        self.assertEqual(_tokens(tools), 1312)

    def test_everything_native_at_once(self):
        tools = build_tools(search=True, memory=True, think=True, doc_gen=True)
        self.assertEqual(len(tools), 6)
        self.assertEqual(_tokens(tools), 2269)

    def test_it_sits_after_think_and_before_the_filesystem_layer(self):
        """A stable prefix: the filesystem layer's size varies with what the MCP server
        reports, and llama-server caches the prompt prefix."""
        # annotations.readOnlyHint has to be there and true: pick_fs_tools fails closed,
        # so an entry without it counts as a writer and fs_write=False drops it.
        fs_raw = [
            {
                "name": "list_directory",
                "description": "",
                "inputSchema": {"type": "object", "properties": {}},
                "annotations": {"readOnlyHint": True},
            }
        ]
        names = [
            t["function"]["name"]
            for t in build_tools(
                search=True, memory=True, think=True, doc_gen=True, fs_read=True, fs_raw=fs_raw
            )
        ]
        self.assertIn("list_directory", names)
        self.assertLess(names.index("think"), names.index(SAVE_DOCUMENT))
        self.assertLess(names.index(SAVE_DOCUMENT), names.index("list_directory"))

    def test_the_budget_the_request_actually_pays(self):
        """budget_safety is the number fit_budget uses, so it is the one worth pinning.

        2484. It has moved twice, both times because a real-model run showed the doc_gen
        line was not doing its job: 2389 as first committed, 2420 once the export sentence
        left the base prompt and the line forbade the substitutions the model was making,
        2484 once a second round showed it also had to forbid asking for confirmation and
        telling the user to operate the tool themselves. export_hint (57) is dropped in
        the same request every time, because main.py appends one or the other, never both.
        """
        settings = Settings(_env_file=None)
        prompt = effective_prompt(settings.system_prompt, {"doc_gen"})
        tools = build_tools(search=True, doc_gen=True)
        self.assertEqual(
            budget_safety(prompt, json.dumps(tools, ensure_ascii=False), 8), 2484
        )


class TestEffectivePromptDocGen(unittest.TestCase):
    def test_the_line_is_absent_when_the_box_is_off(self):
        self.assertNotIn("save_document", effective_prompt(DEFAULT_SYSTEM_PROMPT, set()))

    def test_the_line_is_present_when_it_is_on(self):
        out = effective_prompt(DEFAULT_SYSTEM_PROMPT, {"doc_gen"})
        self.assertIn("save_document", out)
        self.assertIn(str(DOC_GEN_CHAR_HINT), out)

    def test_the_line_costs_what_it_was_measured_at(self):
        """299 for the bare entry and 299 for the delta through effective_prompt.

        The two used to differ by one, estimate_tokens' trailing +1 counted once for the
        joined prompt instead of twice. They agree now by coincidence rather than design:
        the rewritten line is exactly 333 characters and DEFAULT_SYSTEM_PROMPT is too, so
        the +1 cancels. Both figures are written down so the next re-measure does not read
        a change in the gap as an error in the estimator.

        It was 146/147 as first committed, then 234/235. Each step bought clauses a
        real-model run had shown the model ignoring, and the two tests below pin them.
        """
        self.assertEqual(estimate_tokens(PROMPT_LINES["doc_gen"]), 299)
        self.assertEqual(len(PROMPT_LINES["doc_gen"]), 333)
        self.assertEqual(len(DEFAULT_SYSTEM_PROMPT), 333)
        base = estimate_tokens(DEFAULT_SYSTEM_PROMPT)
        self.assertEqual(
            estimate_tokens(effective_prompt(DEFAULT_SYSTEM_PROMPT, {"doc_gen"})) - base, 299
        )

    def test_the_default_prompt_itself_was_not_touched(self):
        """doc_gen is appended per request, so a user's saved SYSTEM_PROMPT keeps working.

        The export sentence used to be the counter-example — written into the default, so
        it silently stopped reaching anyone who had overridden it. It has since moved into
        PROMPT_LINES for the opposite reason (it contradicted this line), which means
        nothing in the default prompt describes a per-request capability any more.
        """
        self.assertNotIn("save_document", DEFAULT_SYSTEM_PROMPT)
        self.assertNotIn("生成文档", DEFAULT_SYSTEM_PROMPT)

    def test_the_line_tells_the_model_not_to_repeat_the_document(self):
        """Load-bearing: without it the model writes the document twice, once as an
        argument and once as prose, which doubles the cost and guarantees an overrun."""
        self.assertIn("重复", PROMPT_LINES["doc_gen"])

    def test_the_line_says_one_document_per_call(self):
        self.assertIn("一份", PROMPT_LINES["doc_gen"])

    def test_the_line_forbids_the_two_substitutions_the_model_actually_made(self):
        """Measured over eight real runs on gemma-4-E4B-it, not imagined.

        Twice it declared the requested length 「超出了单次对话回复和模型输出的稳定控制
        范围」 and handed over an outline instead of a document; once it told the user
        「由于我无法直接生成文件并让您下载」 and pointed at the export button with
        save_document in its own tool list. Naming the tool is not enough to stop either,
        so the line asserts one call holds a whole document and forbids both excuses.
        """
        line = PROMPT_LINES["doc_gen"]
        self.assertIn("装得下整篇", line)
        self.assertIn("大纲", line)
        self.assertIn("无法生成文件", line)
        self.assertIn("不要拒绝", line)

    def test_the_line_closes_the_confirmation_gate_the_second_round_found(self):
        """Two more failure sentences, from the round that ran after the export-hint move.

        Neither is a length complaint, so the test above does not cover them:
          - it wrote the whole draft as prose and then asked 「告诉我是否需要我调用
            save_document 工具帮您下载成文件」, gating a call the user had already asked
            for -- and isSendable would have kept that draft on the next request's wire,
            so the model would be re-read its own hedge and could gate again;
          - it inverted the roles: 「您需要使用 save_document 工具来下载成文件」, as though
            the tool were a control on the page instead of one of its own.
        Both leave the user with prose and no file, which is the whole feature failing.
        """
        line = PROMPT_LINES["doc_gen"]
        self.assertIn("不是让用户去用的按钮", line)
        self.assertIn("不要先给草稿再问要不要保存", line)
        self.assertIn("不要反问", line)

    def test_the_line_requires_a_closing_sentence(self):
        """The one run that did call the tool delivered 2,131 intact characters and then
        generated a single token of prose, leaving a blank bubble beside the card.

        isSendable keeps a content-less turn off the next request's wire, so the model
        would not see its own confirmation either. Permitting a short sentence is what the
        earlier wording did, and the model read it as permission to say nothing — so this
        asks for one and forbids the empty bubble by name.
        """
        line = PROMPT_LINES["doc_gen"]
        self.assertIn("一句话", line)
        self.assertIn("不要留空", line)

    # Generation tokens read out of runtime/llama-server.log, against the characters in the
    # artifact that run delivered. Both gemma-4-E4B-it, save_document actually called,
    # MAX_TOKENS=2048. Literals rather than a fixture: they are observations of a model at a
    # moment in time, nothing in this repo can reproduce them, and re-deriving them from
    # estimate_tokens would only re-measure the estimator.
    MEASURED_DELIVERIES = ((529, 805), (1484, 2745))
    GEN_LIMIT = 2048

    def test_the_character_hint_never_exceeds_a_measured_delivery(self):
        """The strongest thing that can be said offline: 2,745 characters arrived intact,
        so a hint of 2,400 is not asking for anything that has failed.

        This is the assertion the previous version of this test got wrong. It recorded a
        run that lost its round at a *request* for 2,600 characters and treated 2,600 as
        the ceiling, which dropped the hint to 1,800 -- but the model overshoots what it is
        asked for (asked for 1,000, it wrote 2,745), so a failed request says nothing about
        the length that fits. Only delivered documents bound the hint.
        """
        largest = max(chars for _, chars in self.MEASURED_DELIVERIES)
        self.assertLessEqual(DOC_GEN_CHAR_HINT, largest)

    def test_the_character_hint_leaves_headroom_without_wasting_the_window(self):
        """Both directions at once, from the measured densities rather than from ratios.

        805/529 = 1.52 and 2745/1484 = 1.85 characters per generated token. Taking the
        pessimistic one over the whole budget gives ~3,100 characters, so 2,400 sits at 77%
        -- inside the ceiling with room for a table-and-code document, which escapes worse
        than the headings-and-lists text these two runs measured, but not so far inside that
        every document leaves a quarter of the window unused.
        """
        density = min(chars / tokens for tokens, chars in self.MEASURED_DELIVERIES)
        ceiling = self.GEN_LIMIT * density
        self.assertLess(DOC_GEN_CHAR_HINT, ceiling)
        self.assertGreater(DOC_GEN_CHAR_HINT, ceiling * 0.6)


class TestStoredArtifact(unittest.TestCase):
    def test_a_bad_format_is_rejected_rather_than_reaching_the_browser_dispatch(self):
        with self.assertRaises(ValidationError):
            StoredArtifact(filename="a.exe", format="exe", content="x")

    def test_every_real_format_is_accepted(self):
        for fmt in ARTIFACT_FORMATS:
            with self.subTest(fmt):
                self.assertEqual(StoredArtifact(format=fmt).format, fmt)

    def test_content_over_the_cap_is_rejected(self):
        with self.assertRaises(ValidationError):
            StoredArtifact(content="字" * (MAX_ARTIFACT_CHARS + 1))

    def test_content_exactly_at_the_cap_is_accepted(self):
        self.assertEqual(
            len(StoredArtifact(content="字" * MAX_ARTIFACT_CHARS).content), MAX_ARTIFACT_CHARS
        )

    def test_an_over_long_filename_or_title_is_rejected(self):
        with self.assertRaises(ValidationError):
            StoredArtifact(filename="名" * 121)
        with self.assertRaises(ValidationError):
            StoredArtifact(title="标" * 121)

    def test_a_message_from_an_archive_predating_this_field_still_loads(self):
        """The backwards-compatibility nail: every existing session on disk lacks the key."""
        legacy = {"role": "assistant", "content": "好的", "reasoning": "", "sources": []}
        msg = StoredMessage.model_validate(legacy)
        self.assertEqual(msg.artifacts, [])

    def test_a_message_round_trips_its_artifacts(self):
        msg = StoredMessage(
            role="assistant",
            content="已生成报告。",
            artifacts=[StoredArtifact(filename="报告.pdf", format="pdf", title="报告", content=SAMPLE)],
        )
        again = StoredMessage.model_validate(json.loads(msg.model_dump_json()))
        self.assertEqual(again.artifacts[0].content, SAMPLE)
        self.assertEqual(again.artifacts[0].format, "pdf")

    def test_more_than_eight_artifacts_on_one_message_is_rejected(self):
        with self.assertRaises(ValidationError):
            StoredMessage(role="assistant", content="x", artifacts=[StoredArtifact()] * 9)

    def test_the_worst_case_archive_growth_is_bounded(self):
        """8 x MAX_ARTIFACT_CHARS = 320 KB per message, against base64 images that are
        why history.py splits index.json from the bodies in the first place."""
        msg = StoredMessage(
            role="assistant",
            content="x",
            artifacts=[StoredArtifact(content="字" * MAX_ARTIFACT_CHARS) for _ in range(8)],
        )
        self.assertLess(len(msg.model_dump_json()), 8 * MAX_ARTIFACT_CHARS * 3 + 1000)


class TestTruncation(unittest.TestCase):
    """The two bugs, pinned. Both predate save_document; it makes them everyday."""

    def test_a_truncated_argument_is_distinguishable_from_no_argument(self):
        """Bug 1. Both used to return {}, so half a document and an empty call looked
        identical and the tool ran with nothing."""
        self.assertIsNone(_parse_args('{"content": "# 报告\\n\\n本季度'))
        self.assertIsNone(_parse_args('{"filename": "a.md", "format":'))
        self.assertEqual(_parse_args("{}"), {})
        self.assertEqual(_parse_args(""), {})
        self.assertEqual(_parse_args("   "), {})

    def test_a_valid_argument_still_parses(self):
        self.assertEqual(_parse_args('{"filename": "a", "format": "md"}'), {"filename": "a", "format": "md"})

    def test_a_non_dict_argument_keeps_its_old_behaviour(self):
        """{} rather than None: this is not truncation, and callers already cope."""
        self.assertEqual(_parse_args("[1,2]"), {})
        self.assertEqual(_parse_args('"a string"'), {})
        self.assertEqual(_parse_args("null"), {})

    def test_length_with_a_buffered_call_and_no_text_is_an_error(self):
        """Bug 2. This combination used to reach the browser as a completely blank reply:
        the break discarded the half-built call and no delta had been emitted either."""
        note = _truncation_note("length", 1, False, 2048)
        self.assertEqual(note[0], "error")
        self.assertIn("2048", note[1])
        self.assertIn("上限", note[1])
        self.assertIn("截断", note[1])

    def test_that_error_tells_the_model_retrying_is_futile(self):
        """Not a kindness but arithmetic: max_tokens is identical on every round of the
        same request, so an argument that did not fit once provably does not fit again.
        Retrying burns all MAX_TOOL_ROUNDS and ends in a vaguer message."""
        note = _truncation_note("length", 1, False, 2048)
        self.assertIn("重试", note[1])
        self.assertIn("拆小", note[1])

    def test_length_with_text_keeps_the_answer_and_only_warns(self):
        note = _truncation_note("length", 0, True, 2048)
        self.assertEqual(note[0], "status")
        self.assertIn("2048", note[1])

    def test_length_with_both_text_and_a_call_keeps_the_answer(self):
        """The prose is worth more than the half-built call beside it."""
        self.assertEqual(_truncation_note("length", 2, True, 4096)[0], "status")

    def test_length_with_nothing_at_all_still_says_something(self):
        note = _truncation_note("length", 0, False, 2048)
        self.assertEqual(note[0], "error")
        self.assertIn("2048", note[1])

    def test_the_thinking_limit_is_the_one_quoted(self):
        """深度思考 uses MAX_THINKING_TOKENS, so the number in the message must follow."""
        self.assertIn("4096", _truncation_note("length", 1, False, 4096)[1])

    def test_every_other_finish_reason_behaves_exactly_as_before(self):
        for reason in ("stop", "tool_calls", None, "", "content_filter"):
            for n_calls in (0, 1, 3):
                for has_text in (False, True):
                    with self.subTest(f"{reason!r} calls={n_calls} text={has_text}"):
                        self.assertIsNone(_truncation_note(reason, n_calls, has_text, 2048))

    def test_search_tools_are_the_names_that_keep_the_empty_dict_degradation(self):
        """The other half of bug 1's fix: {} is still right for search, where it falls
        through to _fallback_query and searches the user's own question instead of
        failing. Fixing truncation must not regress that."""
        from app.llm import _SEARCH_TOOLS

        self.assertEqual(_SEARCH_TOOLS, frozenset({"web_search", "fetch_url"}))
        self.assertNotIn(SAVE_DOCUMENT, _SEARCH_TOOLS)


if __name__ == "__main__":
    unittest.main()
