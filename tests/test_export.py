"""Export writers: block validation, the DOCX writer, and the PDF packer.

Standard library only, by decision — pytest was explicitly rejected as a new
dependency. Run from the project root:

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m unittest discover -s tests -v

The DOCX oracle is documents._parse_docx — the very reader the upload path uses. A
file that round-trips through it is well-formed OOXML carrying the right text,
which is the strongest check available on a machine with no Word. It is not a soft
check either: hand-written OOXML was caught by it twice while app/export.py was
being designed (a duplicated w:val attribute, then a malformed <w:top .../> tag),
which is the whole reason the writer generates its XML programmatically instead of
concatenating strings.

The PDF tests need pymupdf; nothing else does, because app/export.py imports it
lazily inside make_pdf. TestModuleHygiene pins that property in a subprocess so it
holds no matter what the other classes have already loaded.
"""

from __future__ import annotations

import importlib.util
import io
import re
import subprocess
import sys
import threading
import unittest
import zipfile

from app.config import DEFAULT_SYSTEM_PROMPT, ROOT, Settings
from app.documents import (
    DocumentError,
    _PDF_LOCK,
    _parse_docx,
    estimate_tokens,
)
from app.export import (
    DOCX_MEDIA,
    MAX_EXPORT_BLOCKS,
    MAX_EXPORT_HTML_CHARS,
    MAX_EXPORT_PAGES,
    PDF_MEDIA,
    ExportError,
    build,
    check_blocks,
    make_docx,
    make_pdf,
)
from app.history import TITLE_CHARS
from app.tools import PROMPT_LINES, effective_prompt

HAS_PYMUPDF = importlib.util.find_spec("pymupdf") is not None

if HAS_PYMUPDF:
    import pymupdf

    from app import export as _export_mod

# Every element renderMarkdown can emit (app.js:43-121), one block each. The
# grammar is closed, so this is a complete list rather than a sample: top level
# <pre><code class>, <p>, <h1>-<h6>, a bare void <hr>, <blockquote><p>, <ul|ol><li>,
# <table><tr><th|td>; inline <code>, <a href target rel>, <strong>, <em>, <del>.
FULL_GRAMMAR = [
    "<pre><code class=\"language-python\">def f(x):\n    return x</code></pre>",
    "<p>正文 with <code>inline</code>, <strong>bold</strong>, <em>em</em>, <del>del</del>.</p>",
    "<h1>H1</h1>", "<h2>H2</h2>", "<h3>H3</h3>",
    "<h4>H4</h4>", "<h5>H5</h5>", "<h6>H6</h6>",
    "<hr>",
    "<blockquote><p>引用</p></blockquote>",
    "<ul><li>一</li><li>二</li></ul>",
    "<ol><li>第一</li><li>第二</li></ol>",
    "<table><tr><th>列A</th><th>列B</th></tr><tr><td>值1</td><td>值2</td></tr></table>",
    # What actually arrives, as opposed to what renderMarkdown emits: see
    # test_the_browsers_implied_tbody_is_accepted.
    "<table><tbody><tr><th>列A</th></tr><tr><td>值1</td></tr></tbody></table>",
    '<p><a href="https://example.com/x" target="_blank" rel="noopener noreferrer">链接</a></p>',
    "<p>裸地址 https://example.com/y</p>",
]


def _fresh_probe(statement: str) -> str:
    """Run `statement` in a child interpreter with a clean sys.modules.

    In-process these assertions would be order-dependent: any PDF test that ran
    first would already have imported pymupdf, and the check would then pass or
    fail on alphabetical luck rather than on the module's actual behaviour.
    """
    done = subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT), check=True,
    )
    return done.stdout.strip()


def _docx_parts(data: bytes) -> dict[str, str]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return {name: zf.read(name).decode("utf-8") for name in zf.namelist()}


def _round_trip(blocks) -> str:
    """Write a DOCX and read it back with the upload path's own reader."""
    return _parse_docx("t.docx", make_docx(check_blocks(blocks)).data).text


# ---------------------------------------------------------------------------
# Module hygiene
# ---------------------------------------------------------------------------


class TestModuleHygiene(unittest.TestCase):
    def test_importing_export_does_not_pull_in_pymupdf(self):
        """What keeps the DOCX and validation tests runnable with pymupdf absent.

        The lazy import inside make_pdf is not about the 64 ms — it is about this.
        """
        self.assertEqual(
            _fresh_probe("import sys, app.export; print('pymupdf' in sys.modules)"),
            "False",
        )

    def test_export_imports_nothing_else_from_the_package(self):
        """Pinned so the DOCX round-trip test cannot become a circular import.

        tests import both sides (app.export writes, app.documents._parse_docx
        reads), and documents.py imports nothing from the package either, so the
        pair stays acyclic in both directions.
        """
        self.assertEqual(
            _fresh_probe(
                "import sys, app.export;"
                "print(sorted(m for m in sys.modules"
                " if m.startswith('app.') and m != 'app.export'))"
            ),
            "[]",
        )


# ---------------------------------------------------------------------------
# Block validation
# ---------------------------------------------------------------------------


class TestBlockValidation(unittest.TestCase):
    def test_empty_list_rejected(self):
        with self.assertRaises(ExportError):
            check_blocks([])

    def test_blank_block_rejected(self):
        for blank in ("", "   ", "\n\t "):
            with self.assertRaises(ExportError, msg=repr(blank)):
                check_blocks([blank])

    def test_text_with_no_element_rejected(self):
        """renderMarkdown escapes raw HTML (app.js:51), so bare text is never a
        block; if one arrives it came from a hand-made API call."""
        with self.assertRaises(ExportError):
            check_blocks(["just some text"])

    def test_unknown_root_tags_rejected(self):
        """Pins the whitelist. renderMarkdown cannot produce any of these, so
        seeing one means /api/export was called directly."""
        for tag in ("div", "script", "iframe", "img", "span", "body", "style", "form", "object"):
            with self.assertRaises(ExportError, msg=tag):
                check_blocks([f"<{tag}>x</{tag}>"])

    def test_unknown_descendant_tags_rejected(self):
        for payload in ('<p>a<iframe src="https://e.com"></iframe></p>',
                        '<p>a<img src="https://e.com/a.png"></p>',
                        "<ul><li>a<script>x()</script></li></ul>",
                        '<p><object data="https://e.com/a.swf"></object></p>'):
            with self.assertRaises(ExportError, msg=payload):
                check_blocks([payload])

    def test_non_http_href_rejected(self):
        """Same rule as StoredSource._check_url in main.py and safeLink in app.js:
        re-checked server-side because /api/export is directly callable."""
        for href in ("javascript:alert(1)", "data:text/html,<script>x</script>",
                     "file:///C:/Windows/win.ini", "/relative/path", "#anchor", ""):
            with self.assertRaises(ExportError, msg=href):
                check_blocks([f'<p><a href="{href}">x</a></p>'])

    def test_http_and_https_are_both_accepted(self):
        for href in ("http://example.com/x", "https://example.com/x", "HTTPS://EXAMPLE.COM/x"):
            self.assertEqual(check_blocks([f'<p><a href="{href}">x</a></p>'])[0],
                             f'<p><a href="{href}">x</a></p>')

    def test_event_handler_attributes_are_rejected_not_stripped(self):
        """A deliberate deviation from the plan, which said "stripped".

        Stripping means re-serialising HTML, and neither consumer can be hurt by
        the attribute: MuPDF's HTML engine executes no script at all (pinned by
        TestPdfWriter.test_script_tags_produce_no_text), and the DOCX writer reads
        no attribute but href. Building a serialiser to remove something that
        cannot fire is a bad trade — and a block carrying onclick is one
        renderMarkdown could never have produced, so refusing is the honest answer.
        """
        for payload in ('<p onclick="x()">hi</p>', '<p onmouseover="x()">hi</p>',
                        '<p style="color:red">hi</p>', '<p data-x="1">hi</p>'):
            with self.assertRaises(ExportError, msg=payload):
                check_blocks([payload])

    def test_valid_blocks_come_back_unchanged(self):
        """check_blocks validates; it does not rewrite. The browser's outerHTML is
        what reaches the writers, byte for byte."""
        self.assertEqual(check_blocks(FULL_GRAMMAR), list(FULL_GRAMMAR))

    def test_the_full_rendermarkdown_grammar_is_accepted(self):
        """The first test to fail if the whitelist is ever narrowed by accident."""
        self.assertEqual(len(check_blocks(FULL_GRAMMAR)), len(FULL_GRAMMAR))

    def test_the_browsers_implied_tbody_is_accepted(self):
        """Regression nail for a bug no hand-written block could have caught.

        renderMarkdown emits <table><tr>…, but blocksFromMarkdown() in app.js assigns
        that HTML to element.innerHTML and the browser's parser always wraps the rows
        in an implied <tbody>. Measured in the page: the DOM's entire table vocabulary
        is {table, tbody, td, th, tr}, and <thead> never appears. Without tbody in
        _DESC_TAGS, check_blocks answered "第 7 块含不支持的元素 <tbody>" and PDF and
        DOCX export failed on any answer containing a table. Found only by feeding
        the browser's real captured request body to the validator — every fixture in
        this file was written from renderMarkdown's output, which is the one form
        that never arrives.
        """
        dom_table = ("<table><tbody><tr><th>列A</th><th>列B</th></tr>"
                     "<tr><td>值1</td><td>值2</td></tr></tbody></table>")
        self.assertEqual(check_blocks([dom_table]), [dom_table])
        # The DOCX writer has to reach the cells through the extra element. A gate
        # that accepted a tag the writer then dropped would be the same bug wearing
        # a green test. The PDF side is covered in TestPdfWriter, which is the class
        # that carries the pymupdf guard this one deliberately does not.
        self.assertEqual(_round_trip([dom_table]), "| 列A | 列B |\n| 值1 | 值2 |")

    def test_block_count_cap(self):
        with self.assertRaises(ExportError):
            check_blocks(["<p>x</p>"] * (MAX_EXPORT_BLOCKS + 1))
        self.assertEqual(len(check_blocks(["<p>x</p>"] * MAX_EXPORT_BLOCKS)), MAX_EXPORT_BLOCKS)

    def test_total_character_cap(self):
        over = ["<p>" + "字" * 4000 + "</p>"] * (MAX_EXPORT_HTML_CHARS // 4007 + 1)
        self.assertGreater(sum(len(b) for b in over), MAX_EXPORT_HTML_CHARS)
        with self.assertRaises(ExportError):
            check_blocks(over)

    def test_the_rejection_says_which_block(self):
        """A 400 that does not name the offending block is not actionable."""
        with self.assertRaises(ExportError) as ctx:
            check_blocks(["<p>fine</p>", "<p>also fine</p>", "<div>not fine</div>"])
        self.assertIn("第 3 块", str(ctx.exception))


# ---------------------------------------------------------------------------
# DOCX writer
# ---------------------------------------------------------------------------


class TestDocxWriter(unittest.TestCase):
    def test_zip_is_valid_and_has_exactly_five_parts(self):
        """Pins the part list. Drop [Content_Types].xml and Word reports "the file
        is corrupt" without saying why; add docProps/core.xml and there is a sixth
        part the round-trip oracle cannot vouch for and no reader needs."""
        data = make_docx(check_blocks(["<p>x</p>"])).data
        self.assertTrue(data.startswith(b"PK\x03\x04"))
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            self.assertIsNone(zf.testzip())
            self.assertEqual(
                sorted(zf.namelist()),
                sorted(["[Content_Types].xml", "_rels/.rels", "word/document.xml",
                        "word/_rels/document.xml.rels", "word/styles.xml"]),
            )

    def test_media_type_and_extension(self):
        out = make_docx(check_blocks(["<p>x</p>"]))
        self.assertEqual(out.media_type, DOCX_MEDIA)
        self.assertEqual(out.ext, "docx")
        self.assertEqual(out.pages, 0)

    def test_round_trips_through_the_existing_reader(self):
        text = _round_trip(FULL_GRAMMAR)
        for needle in ("正文", "inline", "bold", "H1", "H6", "引用", "链接", "列A", "值1"):
            self.assertIn(needle, text)

    def test_table_becomes_pipe_rows(self):
        """Proves a real w:tbl/w:tr/w:tc structure rather than flattened
        paragraphs: documents._table_lines only emits `| a | b |` for a w:tbl."""
        text = _round_trip(["<table><tr><th>列A</th><th>列B</th></tr>"
                            "<tr><td>值1</td><td>value2</td></tr></table>"])
        self.assertIn("| 列A | 列B |", text)
        self.assertIn("| 值1 | value2 |", text)

    def test_cjk_survives(self):
        """Catches a missing w:eastAsia in w:rFonts and a wrong XML encoding —
        both make Word substitute a font or mojibake the text, and neither is
        visible from the byte length alone."""
        src = "中文正文：简体中文、繁體中文、日本語かな、한국어"
        self.assertIn(src, _round_trip([f"<p>{src}</p>"]))
        self.assertIn('w:eastAsia="Microsoft YaHei"', _docx_parts(make_docx(["<p>x</p>"]).data)["word/styles.xml"])

    def test_literal_bullets_are_in_the_text(self):
        """The evidence behind choosing literal characters over w:numPr.

        A numPr bullet is synthesised by Word at render time and never exists in
        w:t at all, so with no Word on this machine no stdlib test could ever
        confirm it. A literal one round-trips through _para_text and is asserted
        here. The honest cost: these are not real Word lists, so the list gallery
        cannot restyle them and a screen reader will not announce "bullet, 1 of 3".
        """
        text = _round_trip(["<ul><li>无序一</li><li>无序二</li></ul>",
                            "<ol><li>有序一</li><li>有序二</li></ol>"])
        self.assertIn("•", text)
        self.assertIn("无序一", text)
        self.assertIn("1.", text)
        self.assertIn("有序二", text)

    def test_ordered_numbering_restarts_per_list(self):
        """A Python counter rather than a w:abstractNum per list, which is what
        restarting would cost in real numbering."""
        text = _round_trip(["<ol><li>a</li><li>b</li></ol>", "<p>中间</p>",
                            "<ol><li>c</li><li>d</li></ol>"])
        numbers = re.findall(r"^(\d+)\.", text, re.M)
        self.assertEqual(numbers, ["1", "2", "1", "2"])

    def test_inline_styles_do_not_leak_across_runs(self):
        """Catches the _style flags not being popped on the end tag.

        <strong>a</strong>b<em>c</em>d is four runs: a bold, b plain, c italic,
        d plain. Asserting the whole table at once rather than spot-checking one
        pair, because leaking in either direction is the same bug.
        """
        blocks = ["<p><strong>a</strong>b<em>c</em>d</p>"]
        body = _docx_parts(make_docx(blocks).data)["word/document.xml"]
        runs = re.findall(r"<w:r>.*?</w:r>", body, re.S)
        self.assertEqual([re.search(r"<w:t[^>]*>([^<]*)</w:t>", r).group(1) for r in runs],
                         ["a", "b", "c", "d"])
        for run, expected in zip(runs, ("<w:b/>", "", "<w:i/>", "")):
            for flag in ("<w:b/>", "<w:i/>", "<w:strike/>"):
                self.assertEqual(flag in run, flag == expected, run)
        # the reader still sees the text in order
        self.assertIn("abcd", _round_trip(blocks))

    def test_header_cells_are_bold_without_breaking_run_property_order(self):
        """A regression pin for a bug this writer actually had.

        The first version spliced <w:b/> into already-serialised header runs,
        which put it BEFORE <w:rFonts> and violated CT_RPr's schema-defined child
        order (rFonts, b, i, strike, color, u, shd). Word rejects that. Header
        boldness is now a depth counter consulted while the run is built.
        """
        body = _docx_parts(make_docx(
            ["<table><tr><th><code>x</code> 头</th></tr><tr><td>值</td></tr></table>"]
        ).data)["word/document.xml"]
        for rpr in re.findall(r"<w:rPr>(.*?)</w:rPr>", body, re.S):
            order = [t for t in ("w:rFonts", "w:b", "w:i", "w:strike", "w:color", "w:u", "w:shd")
                     if f"<{t}" in rpr]
            positions = [rpr.index(f"<{t}") for t in order]
            self.assertEqual(positions, sorted(positions), f"CT_RPr order violated: {rpr}")
        self.assertIn("<w:b/>", body)

    def test_xml_special_characters_are_escaped(self):
        """Also a well-formedness check: ElementTree.fromstring raises on a bare
        &, so the round-trip succeeding proves the escaping happened."""
        src = "特殊字符 & < > \" ' 结束"
        self.assertIn(src, _round_trip([f"<p>{src}</p>"]))

    def test_pre_newlines_become_breaks(self):
        """One w:p with w:br between lines, not one w:p per line — that is what
        keeps the shading contiguous. _para_text turns w:br back into \\n."""
        text = _round_trip(["<pre><code>第一行\n第二行\n    缩进保留</code></pre>"])
        self.assertIn("第一行\n第二行\n    缩进保留", text)

    def test_hyperlinks_become_real_external_relationships(self):
        """PDF loses link targets (insert_htmlbox emits 0 link annotations), so
        DOCX is the format to recommend when a link has to keep working."""
        parts = _docx_parts(make_docx(
            ['<p><a href="https://example.com/a">甲</a> 与 <a href="https://example.com/b">乙</a></p>']
        ).data)
        rels = parts["word/_rels/document.xml.rels"]
        self.assertEqual(rels.count('TargetMode="External"'), 2)
        self.assertIn('Target="https://example.com/a"', rels)
        self.assertIn('Target="https://example.com/b"', rels)
        self.assertIn("<w:hyperlink", parts["word/document.xml"])
        # rId1 belongs to styles.xml, so links must start at rId2 and stay unique
        ids = re.findall(r'Id="(rId\d+)"', rels)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids[0], "rId1")

    def test_the_r_namespace_is_declared_on_the_document_element(self):
        """Without xmlns:r on <w:document>, w:hyperlink r:id resolves to nothing.
        The first prototype failed exactly here."""
        xml = _docx_parts(make_docx(["<p>x</p>"]).data)["word/document.xml"]
        # Not split(">", 1)[0]: the first ">" ends the XML declaration, not the
        # document element.
        head = xml[xml.index("<w:document"):].split(">", 1)[0]
        self.assertIn("xmlns:w=", head)
        self.assertIn("xmlns:r=", head)

    def test_every_cell_ends_in_a_paragraph(self):
        """A w:tc whose last child is not a w:p is what makes Word call the file
        corrupt, and it is easy to produce for an empty cell."""
        body = _docx_parts(make_docx(
            ["<table><tr><td>a</td><td></td></tr><tr><td></td><td>d</td></tr></table>"]
        ).data)["word/document.xml"]
        for cell in re.findall(r"<w:tc>.*?</w:tc>", body, re.S):
            self.assertTrue(cell.rstrip().endswith("</w:p></w:tc>"), cell[-60:])
        self.assertIn("| a |  |", _round_trip(
            ["<table><tr><td>a</td><td></td></tr><tr><td></td><td>d</td></tr></table>"]))

    def test_short_rows_are_padded_to_the_column_count(self):
        """A ragged w:tr makes Word repair the file. renderMarkdown cannot emit
        one, but a direct API call can.

        Measured output is two lines of four pipes each — _table_lines synthesises
        no divider row, so the padding shows up as an equal column count rather
        than as a shape.
        """
        text = _round_trip(["<table><tr><th>a</th><th>b</th><th>c</th></tr>"
                            "<tr><td>1</td></tr></table>"])
        rows = text.split("\n")
        self.assertEqual(len(rows), 2)
        self.assertEqual({row.count("|") for row in rows}, {4})
        self.assertEqual(rows[1], "| 1 |  |  |")

    def test_hand_written_xml_errors_are_what_the_reader_catches(self):
        """Records that the oracle is strict, which is the entire reason the
        writer generates XML programmatically instead of concatenating strings.

        Both payloads below are the shape of mistakes made by hand while designing
        this module, and both are rejected by documents._parse_docx rather than
        passing quietly.
        """
        good = make_docx(["<p>x</p>"]).data
        for label, mutate in (
            ("duplicated attribute", lambda s: s.replace("<w:p>", '<w:p w:rsidR="1" w:rsidR="2">', 1)),
            ("mis-nested tag", lambda s: s.replace("</w:p>", "</w:r></w:p>", 1)),
        ):
            parts = _docx_parts(good)
            parts["word/document.xml"] = mutate(parts["word/document.xml"])
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, text in parts.items():
                    zf.writestr(name, text)
            with self.assertRaises(DocumentError, msg=label) as ctx:
                _parse_docx("bad.docx", buf.getvalue())
            self.assertIn("结构损坏", str(ctx.exception))


# ---------------------------------------------------------------------------
# PDF writer
# ---------------------------------------------------------------------------


def _page_scales(blocks) -> list[float]:
    """The scale each page rendered at, recomputed the way make_pdf computes it.

    The scale is not recoverable from the finished PDF — insert_htmlbox returns it
    and pymupdf keeps nothing — so this replays the packing pass with the same
    probe. It is the only way to assert "paginated instead of shrinking" rather
    than "produced text", and that distinction is exactly what made the first PDF
    proof for this feature worthless: every text probe passed on a page of 1.7pt
    type because the glyphs were there, just microscopic.
    """
    def scale_of(html: str) -> float:
        scratch = pymupdf.open()
        try:
            page = scratch.new_page(width=_export_mod._PAGE_W, height=_export_mod._PAGE_H)
            return page.insert_htmlbox(_export_mod._WHERE, html, css=_export_mod._PDF_CSS)[1]
        finally:
            scratch.close()

    pages, _ = _export_mod._pack(list(_export_mod._expand(blocks)), scale_of)
    return [scale_of(html) for html in pages]


def _pdf_text(data: bytes) -> str:
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return "".join(page.get_text() for page in doc)


@unittest.skipUnless(HAS_PYMUPDF, "the PDF writer needs pymupdf; the rest of this file does not")
class TestPdfWriter(unittest.TestCase):
    def test_output_is_a_pdf(self):
        out = make_pdf(check_blocks(["<p>中文正文。</p>"]))
        self.assertEqual(out.media_type, PDF_MEDIA)
        self.assertEqual(out.ext, "pdf")
        self.assertTrue(out.data.startswith(b"%PDF-"))
        with pymupdf.open(stream=out.data, filetype="pdf") as doc:
            self.assertEqual(doc.page_count, 1)
            self.assertEqual(out.pages, 1)
        self.assertFalse(out.truncated)

    def test_cjk_text_is_extractable(self):
        out = make_pdf(check_blocks(["<h2>结论</h2>", "<p>这是一段中文正文，含标点。</p>"]))
        text = _pdf_text(out.data)
        self.assertIn("结论", text)
        self.assertIn("这是一段中文正文", text)

    def test_fonts_are_embedded(self):
        """Without an embedded CJK font the file is a page of boxes on any machine
        that does not have one — measured here as Droid Sans Fallback."""
        out = make_pdf(check_blocks(["<p>中文字形测试。</p>"]))
        with pymupdf.open(stream=out.data, filetype="pdf") as doc:
            names = [entry[3] for entry in doc.get_page_fonts(0)]
        self.assertTrue(names, "no fonts embedded at all")
        self.assertTrue(any("Fallback" in n or "Sans" in n for n in names), names)

    def test_long_content_paginates_instead_of_shrinking(self):
        """The regression pin for the bug that invalidated the original design.

        One insert_htmlbox call neither clips nor paginates — it auto-shrinks, and
        its return value is (spare_height, scale). Measured on this machine: 60
        paragraphs return (0.0, 0.5303), i.e. 5.8pt type, and 200 return
        (0.0, 0.1591), i.e. 1.7pt. Reading only the first element reports success,
        because spare height reads 0.0 both for a page that is exactly full at
        scale 1.0 and for one shrunk to a sixth.
        """
        blocks = check_blocks([f"<p>第 {i} 段中文正文内容。</p>" for i in range(108)])
        out = make_pdf(blocks)
        self.assertGreater(out.pages, 1)
        scales = _page_scales(blocks)
        self.assertEqual(len(scales), out.pages)
        for index, scale in enumerate(scales):
            self.assertGreaterEqual(scale, 1.0, f"page {index + 1} shrank to {scale}")
        text = _pdf_text(out.data)
        self.assertIn("第 0 段", text)
        self.assertIn("第 107 段", text)

    def test_garbage_collection_is_applied(self):
        """Pins tobytes(garbage=4, deflate=True), which is load-bearing.

        Measured on a 6-page document: tobytes() = 18,502,649 bytes, with
        garbage=4 and deflate = 27,206. insert_htmlbox embeds the CJK font once
        per page and plain tobytes() keeps every copy; subset_fonts() alone does
        not collapse them. The single-page sample that first looked fine was 477 KB
        and hid this completely.
        """
        out = make_pdf(check_blocks([f"<p>第 {i} 段中文正文内容。</p>" for i in range(108)]))
        self.assertGreater(out.pages, 1)
        self.assertLess(len(out.data), 200_000, f"{len(out.data)} bytes means the font copies survived")

    def test_oversized_pre_is_split_not_shrunk(self):
        """A 300-line fence measured scale 0.2128 as a single insert_htmlbox call."""
        block = "<pre><code>" + "\n".join(f"line {i}" for i in range(300)) + "</code></pre>"
        blocks = check_blocks([block])
        out = make_pdf(blocks)
        self.assertGreater(out.pages, 1)
        for index, scale in enumerate(_page_scales(blocks)):
            self.assertGreaterEqual(scale, 1.0, f"page {index + 1} shrank to {scale}")
        text = _pdf_text(out.data)
        self.assertIn("line 0", text)
        self.assertIn("line 299", text)

    def test_long_code_lines_wrap_instead_of_shrinking_the_whole_page(self):
        """Pins `overflow-wrap:break-word` in _PDF_CSS.

        MuPDF does not wrap inside <pre>, so a long line overflows the box
        sideways and insert_htmlbox shrinks the ENTIRE page to compensate —
        measured scale 0.2873 for one 300-character code line, identical at every
        line count, which no amount of vertical splitting can fix. With the rule it
        returns to 1.0000. MuPDF's CSS subset is picky about the spelling:
        word-break:break-all, overflow-wrap:anywhere and the legacy alias
        word-wrap:break-word were all measured inert.
        """
        blocks = check_blocks(["<pre><code>" + "\n".join("y" * 300 for _ in range(60)) + "</code></pre>"])
        for index, scale in enumerate(_page_scales(blocks)):
            self.assertGreaterEqual(scale, 1.0, f"page {index + 1} shrank to {scale}")

    def test_a_long_url_in_prose_does_not_shrink_the_page(self):
        """The realistic trigger for the same CSS rule: answers cite URLs, and a
        300-character one in a <p> measured scale 0.2375 without it."""
        long = "https://example.com/" + "a" * 300
        blocks = check_blocks([f'<p>见 <a href="{long}">链接</a> 与裸地址 {long}</p>'] * 40)
        for index, scale in enumerate(_page_scales(blocks)):
            self.assertGreaterEqual(scale, 1.0, f"page {index + 1} shrank to {scale}")

    def test_an_ordinary_table_fills_the_width(self):
        """Pins `table{width:100%}` in _PDF_CSS, and this one is about ordinary
        content rather than exotic content: without it MuPDF sizes an auto table to
        its content, the border box lands a hair over the page, and a plain 30-row
        table of short cells measured scale 0.8634 (a 37-row one, 0.6840). With it
        both return to 1.0000. It also matches the screen, where .md table is
        width:100% (style.css:397).
        """
        block = ("<table><tr><th>列A</th><th>列B</th></tr>"
                 + "".join(f"<tr><td>值{i}</td><td>value{i}</td></tr>" for i in range(30))
                 + "</table>")
        blocks = check_blocks([block])
        self.assertEqual(_page_scales(blocks), [1.0])

    def test_oversized_table_repeats_its_header(self):
        """Page 2 of a long table still has to say what its columns are.

        The fixture has no <tbody> even though every table that actually arrives
        does; see test_the_splitter_is_indifferent_to_the_implied_tbody for why
        that leaves this test speaking for the real shape too.
        """
        block = ("<table><tr><th>列A</th><th>列B</th></tr>"
                 + "".join(f"<tr><td>值{i}</td><td>value{i}</td></tr>" for i in range(250))
                 + "</table>")
        out = make_pdf(check_blocks([block]))
        self.assertGreater(out.pages, 1)
        with pymupdf.open(stream=out.data, filetype="pdf") as doc:
            carrying = sum(1 for page in doc if "列A" in page.get_text())
            text = "".join(page.get_text() for page in doc)
        self.assertEqual(carrying, out.pages, "a page lost the repeated header row")
        self.assertIn("值0", text)
        self.assertIn("值249", text)

    def test_the_splitter_is_indifferent_to_the_implied_tbody(self):
        """_split_table reconstructs each chunk as open_tag + rows + "</table>", so
        the <tbody> the browser always inserts is dropped on the way through. That
        has to be harmless, and it is measured rather than assumed: a 250-row table
        yields the same 7 chunks and the same 9 pages of text with and without it.

        Not compared byte for byte, which cannot work: pymupdf writes a random /ID
        trailer, so rendering the SAME input twice is also byte-unequal — measured,
        54 differing bytes all inside /ID at offset 36010 of a 36,103-byte file.

        Cheap on purpose — the chunk equality is regex-level, so it costs nothing
        next to the multi-second pagination test above, and it is what lets that
        test's tbody-less fixture stand in for the shape real exports actually send.
        """
        rows = "".join(f"<tr><td>{i}</td><td>value{i}</td></tr>" for i in range(250))
        head = "<tr><th>colA</th><th>colB</th></tr>"
        plain = f"<table>{head}{rows}</table>"
        wrapped = f"<table><tbody>{head}{rows}</tbody></table>"
        self.assertEqual(_export_mod._split_table(wrapped, 37),
                         _export_mod._split_table(plain, 37))
        bare, with_body = make_pdf(check_blocks([plain])), make_pdf(check_blocks([wrapped]))
        self.assertEqual(with_body.pages, bare.pages)
        self.assertEqual(_pdf_text(with_body.data), _pdf_text(bare.data))

    def test_wide_table_cells_that_wrap_still_reach_full_scale(self):
        """The case _halve exists for: cells long enough to wrap make a chunk too
        TALL, and halving rows converges. Width is the one thing splitting cannot
        fix, and that residual is documented on _PRE_SPLIT_LINES rather than tested
        here, because it has no good outcome to assert."""
        block = ("<table><tr><th>列A</th><th>列B</th></tr>"
                 + "".join(f"<tr><td>{'这是一段较长的中文说明文字' * 3}</td><td>{'另一段中文说明' * 4}</td></tr>"
                           for _ in range(30))
                 + "</table>")
        blocks = check_blocks([block])
        for index, scale in enumerate(_page_scales(blocks)):
            self.assertGreaterEqual(scale, 1.0, f"page {index + 1} shrank to {scale}")

    def test_script_tags_produce_no_text(self):
        """MuPDF's HTML engine is inert by construction. check_blocks refuses a
        <script> root long before this, so make_pdf is called directly to pin the
        second line of defence — the one that matters if the whitelist is ever
        widened."""
        out = make_pdf(['<p>before</p>', '<script>var x = "SECRET"; document.write(x)</script>', '<p>after</p>'])
        text = _pdf_text(out.data)
        self.assertIn("before", text)
        self.assertIn("after", text)
        self.assertNotIn("SECRET", text)
        self.assertNotIn("document.write", text)

    def test_a_remote_image_is_neither_fetched_nor_rendered(self):
        """No SSRF, and no hang. Measured at 0.002 s against an unroutable host,
        which is the point: it does not go looking. 10.255.255.1 is used rather
        than a real hostname so this test cannot depend on, or touch, a network."""
        out = make_pdf(['<p>x<img src="http://10.255.255.1/a.png"></p>'])
        self.assertTrue(out.data.startswith(b"%PDF-"))
        self.assertEqual(out.pages, 1)

    def test_page_cap_truncates_and_says_so(self):
        """Mirrors documents._parse_pdf: a document over the cap still exports its
        first MAX_EXPORT_PAGES pages and reports the truncation, rather than
        failing. The client words the warning from the flag, because Starlette
        encodes headers as latin-1 and the warning is Chinese.

        65 blocks of a 40-line <pre>, because a 40-line pre is 488pt of the 770pt
        content box: two cannot share a page, so every block forces one. Measured
        65 blocks / 52,445 characters / 0.64 s / pages=60 / truncated=True —
        inside MAX_EXPORT_BLOCKS and MAX_EXPORT_HTML_CHARS, so the real cap is
        what gets exercised rather than an earlier refusal. Filling 61 pages with
        paragraphs instead needs 4000 of them, which check_blocks rejects first.
        """
        blocks = check_blocks([
            "<pre><code>" + "\n".join(f"line {i} of block {n}" for i in range(40)) + "</code></pre>"
            for n in range(65)
        ])
        out = make_pdf(blocks)
        self.assertTrue(out.truncated)
        self.assertEqual(out.pages, MAX_EXPORT_PAGES)
        with pymupdf.open(stream=out.data, filetype="pdf") as doc:
            self.assertEqual(doc.page_count, MAX_EXPORT_PAGES)

    def test_empty_and_whitespace_only_input(self):
        """Must refuse rather than produce a 0-page or blank PDF. check_blocks
        catches the whitespace; make_pdf's own guard catches what slips past it."""
        with self.assertRaises(ExportError):
            make_pdf([])
        with self.assertRaises(ExportError):
            check_blocks(["   "])

    def test_the_title_lands_in_the_metadata(self):
        out = make_pdf(check_blocks(["<p>x</p>"]), title="会话标题")
        with pymupdf.open(stream=out.data, filetype="pdf") as doc:
            self.assertEqual(doc.metadata.get("title"), "会话标题")

    def test_links_survive_as_real_annotations(self):
        """Pins a measurement error, not a limitation.

        The design notes recorded "insert_htmlbox produces 0 link annotations" and
        planned to steer users to DOCX whenever a link had to work. That was wrong
        in the same way the original "PDF is fine" proof was wrong: page.get_links()
        returns [] on the live in-memory page and the annotations only become
        visible after tobytes() plus a reopen. Through the real make_pdf path a
        link is a genuine annotation with the right URI.

        Measured, one annotation per word of the label plus zero-width ones at the
        gaps: 'the docs' yields three rects spanning 57.8-73.1, 73.1-76.1 and
        76.1-99.4, which tile the two words exactly. So the claim worth pinning is
        about scope: the union covers the label and nothing beside it.
        """
        out = make_pdf(check_blocks(
            ['<p>see <a href="https://example.com/y">the docs</a> for more</p>']))
        with pymupdf.open(stream=out.data, filetype="pdf") as doc:
            links = doc[0].get_links()
            words = {w[4]: w[:4] for w in doc[0].get_text("words")}
        self.assertEqual({link["uri"] for link in links}, {"https://example.com/y"})

        def covered(word: str) -> bool:
            x0, _, x1, _ = words[word]
            return any(link["from"].x0 <= x0 and x1 <= link["from"].x1 for link in links)

        for inside in ("the", "docs"):
            self.assertTrue(covered(inside), inside)
        for outside in ("see", "for", "more"):
            self.assertFalse(covered(outside), outside)
        self.assertIn("the docs", _pdf_text(out.data))

    def test_the_render_lock_serialises_concurrent_exports(self):
        """run_in_threadpool makes exports concurrent and MuPDF is a C library, so
        the lock both sidesteps its thread-safety questions and bounds peak RSS to
        one render at a time. Asserted as a property of the module, not exercised
        under load: two 300-block renders in a thread would cost seconds for
        nothing this test can observe.

        Not documents._PDF_LOCK: export.py imports nothing from inside the package,
        so it could not reach it even if sharing were wanted.
        """
        self.assertIsInstance(_export_mod._RENDER_LOCK, type(threading.Lock()))
        self.assertIsNot(_export_mod._RENDER_LOCK, _PDF_LOCK)


# ---------------------------------------------------------------------------
# Format discrimination
# ---------------------------------------------------------------------------


class TestFormatDiscrimination(unittest.TestCase):
    def test_build_dispatches_on_format(self):
        blocks = check_blocks(["<p>中文</p>"])
        pdf = build("pdf", blocks)
        docx = build("docx", blocks)
        self.assertEqual((pdf.media_type, pdf.ext), (PDF_MEDIA, "pdf"))
        self.assertEqual((docx.media_type, docx.ext), (DOCX_MEDIA, "docx"))
        self.assertTrue(pdf.data.startswith(b"%PDF-"))
        self.assertTrue(docx.data.startswith(b"PK\x03\x04"))

    def test_unknown_format_rejected(self):
        with self.assertRaises(ExportError) as ctx:
            build("rtf", ["<p>x</p>"])
        self.assertIn("rtf", str(ctx.exception))

    @unittest.skipUnless(HAS_PYMUPDF, "pdf needs pymupdf")
    def test_the_exported_dataclass_is_frozen(self):
        """The endpoint reads it and nothing else; a mutable result would let a
        caller's edit survive into the next one."""
        out = build("pdf", check_blocks(["<p>x</p>"]))
        with self.assertRaises(Exception):
            out.pages = 99

    def test_client_only_formats_are_not_this_modules_business(self):
        """Pins the division of labour so nobody grows a server-side markdown
        dependency here later.

        md, html and csv never leave the browser: it already holds the markdown
        source and the rendered DOM, so those three cost no round trip at all.
        Only pdf and docx come here, because those two need a real writer.
        """
        for fmt in ("md", "html", "csv", "txt", "markdown"):
            with self.assertRaises(ExportError, msg=fmt):
                build(fmt, ["<p>x</p>"])

    def test_exportin_rejects_unknown_format_at_the_model(self):
        """The 422 happens before the handler runs, which is why the route needs no
        dispatch table of its own."""
        from pydantic import ValidationError

        from app.main import ExportIn
        with self.assertRaises(ValidationError):
            ExportIn(format="rtf", blocks=["<p>x</p>"])
        for fmt in ("md", "html", "csv"):
            with self.assertRaises(ValidationError, msg=fmt):
                ExportIn(format=fmt, blocks=["<p>x</p>"])
        self.assertEqual(ExportIn(format="docx", blocks=["<p>x</p>"]).title, "")


# ---------------------------------------------------------------------------
# The system prompt line
# ---------------------------------------------------------------------------


class TestSystemPromptLine(unittest.TestCase):
    """The 「用户想要文件时…」 sentence, and the layer it lives in.

    It used to be the last line of DEFAULT_SYSTEM_PROMPT, unconditionally. Real-model
    runs showed why that could not survive 生成文档: the model read it first and obeyed
    it, telling the user 「由于我无法直接生成文件并让您下载」 and pointing at the export
    button with save_document sitting in its own tool list. It now lives in
    PROMPT_LINES and is appended only while doc_gen is off, so exactly one instruction
    ever answers "the user wants a file".
    """

    def test_export_sentence_is_present(self):
        """The one line approach A adds, worded exactly as it was.

        It only affects whether the model *mentions* the button — the button works
        regardless, which is the whole advantage of A over hanging a tool off TOOLS.
        Kept verbatim on the move: it worked fine for a model with no save_document to
        prefer, and the assertions below were written against these words.
        """
        line = PROMPT_LINES["export_hint"]
        self.assertIn("导出", line)
        self.assertIn("Markdown", line)
        for fmt in ("MD", "HTML", "CSV", "PDF", "DOCX"):
            self.assertIn(fmt, line)

    def test_the_base_prompt_no_longer_mentions_the_button(self):
        """The contradiction pinned from the side that caused it.

        If this sentence ever finds its way back into DEFAULT_SYSTEM_PROMPT, every
        request with 生成文档 ticked carries two answers to the same question again,
        and the one that appears first is the one gemma obeys.
        """
        self.assertNotIn("导出", DEFAULT_SYSTEM_PROMPT)
        self.assertNotIn("save_document", DEFAULT_SYSTEM_PROMPT)

    def test_doc_gen_and_export_hint_never_reach_the_model_together(self):
        """What the move buys: with 生成文档 on, nothing tells the model to hand the
        user a button instead of calling the tool.

        Asserted on 「并提示他点」 rather than on 导出, because doc_gen's own line says
        不要让用户自己去点导出按钮 — a bare 导出 check would fail on the sentence that
        forbids exactly what this test is about.
        """
        generated = effective_prompt(DEFAULT_SYSTEM_PROMPT, {"doc_gen"})
        self.assertIn("save_document", generated)
        self.assertNotIn("并提示他点", generated)
        exported = effective_prompt(DEFAULT_SYSTEM_PROMPT, {"export_hint"})
        self.assertIn("并提示他点", exported)
        self.assertNotIn("save_document", exported)

    def test_no_prompt_line_carries_its_own_newline(self):
        """effective_prompt joins with "\\n", so a line that ends in one puts a blank
        line in the middle of the system prompt.

        This is the hazard the old test_sentences_are_newline_separated guarded, moved
        to the layer that now does the joining: appending to DEFAULT_SYSTEM_PROMPT used
        to be the edit, and getting the \\n wrong produced one run-on line.
        """
        for key, text in PROMPT_LINES.items():
            self.assertEqual(text, text.strip(), f"{key} carries surrounding whitespace")
        self.assertNotIn("\n\n", effective_prompt(DEFAULT_SYSTEM_PROMPT, set(PROMPT_LINES)))

    def test_every_line_but_the_last_ends_with_a_backslash_n(self):
        """The string's implicit invariant, asserted so the next append does not
        have to rediscover it."""
        lines = DEFAULT_SYSTEM_PROMPT.split("\n")
        self.assertGreater(len(lines), 1)
        self.assertFalse(DEFAULT_SYSTEM_PROMPT.endswith("\n"))
        for line in lines:
            self.assertTrue(line.strip(), "a blank line means a doubled \\n")
            self.assertTrue(line.endswith(("。", "；")) or line.endswith("."),
                            f"line does not end with a full stop: {line!r}")

    def test_still_fits_the_settings_patch_limit(self):
        """settings_store.SettingsPatch caps system_prompt at 4000; over that the
        settings panel cannot save the prompt back at all.

        Back to 333 characters and 297 tokens now that the sentence moved out — exactly
        what it was before approach A, leaving 3667 characters of headroom. The hint is
        appended per request instead, which settings_store never sees and so never has
        to fit inside the patch limit.
        """
        self.assertLessEqual(len(DEFAULT_SYSTEM_PROMPT), 4000)
        self.assertEqual(len(DEFAULT_SYSTEM_PROMPT), 333)

    def test_the_export_hint_costs_57_tokens(self):
        """Approach A's entire runtime price, unchanged by the move: same words, same
        57 tokens, measured here rather than quoted from the design notes.

        What the move changes is who pays it. It is appended only while 生成文档 is off,
        so a request with that box ticked pays for doc_gen's line instead and never for
        both — and every combination with the box unticked costs exactly what it did
        when the sentence was baked into the default prompt.
        """
        self.assertEqual(estimate_tokens(PROMPT_LINES["export_hint"]), 57)

    def test_a_saved_prompt_no_longer_loses_the_sentence(self):
        """The shadowing hazard, inverted by the move.

        Precedence is still runtime/settings_override.json, then .env, then
        DEFAULT_SYSTEM_PROMPT, applied by Settings(**overrides) in get_settings(). When
        the sentence lived in the default, a user who had ever saved a prompt through the
        settings panel kept their own text and never saw it again — editing the default
        could not reach them, which is why the export pane could only warn about it.
        Appending per request does reach them, and their own words are still untouched.
        _env_file=None keeps this off the local .env, so it says the same thing on every
        machine.
        """
        self.assertNotIn("导出", Settings(_env_file=None).system_prompt)
        shadowed = Settings(_env_file=None, system_prompt="你是一个助手。").system_prompt
        self.assertEqual(shadowed, "你是一个助手。")
        self.assertIn("导出", effective_prompt(shadowed, {"export_hint"}))
        self.assertNotIn("导出", shadowed)


# ---------------------------------------------------------------------------
# Filename rules
# ---------------------------------------------------------------------------


class TestFilenameSanitisation(unittest.TestCase):
    """The Python twin of sanitiseFilename() in static/app.js.

    The JS is the authoritative implementation and there is no Node on this
    machine to run it, so what is pinned here is the RULE, not the code: if the
    two ever disagree this is the specification to settle it against.

    It is not a hypothetical. The real archive at runtime/history/index.json
    contains a title starting '> **【角色设定】**：你是一位深谙家庭心理学…' — both >
    and * are illegal in a Windows filename — and two sessions whose titles are
    byte-identical, so collisions are real rather than theoretical.
    """

    FS_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
    FS_RESERVED = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])$", re.I)

    def sanitise(self, name: str) -> str:
        s = self.FS_UNSAFE.sub("_", str(name or ""))
        s = re.sub(r"^[.\s]+", "", s)
        s = re.sub(r"[.\s]+$", "", s)[:TITLE_CHARS]
        if self.FS_RESERVED.match(s.split(".")[0]):
            s = f"_{s}"
        return s

    def base_of(self, field: str, title: str) -> str:
        """exportFilename's fallback chain, which is where non-emptiness lives."""
        return self.sanitise(field) or self.sanitise(title) or "对话"

    def test_titles_from_the_real_archive_are_sanitisable(self):
        shapes = [
            "> **【角色设定】**：你是一位深谙家庭心理学，擅长用温和而坚定的方式帮助家庭成员",
            "对比 MD, DOCX, PDF, HTML, CSV 的差异",
            'C:\\Users\\user\\报告.txt',
            "  前后有空格与点...  ",
            "...",
            "",
            "CON",
            "nul.txt",
            "com1",
            "标题\x00带控制字符\x1f",
            "a" * 200,
        ]
        for raw in shapes:
            out = self.sanitise(raw)
            with self.subTest(raw=raw[:30]):
                self.assertLessEqual(len(out), TITLE_CHARS)
                self.assertNotRegex(out, r'[\\/:*?"<>|\x00-\x1f]')
                self.assertNotRegex(out, r"^[.\s]")
                self.assertNotRegex(out, r"[.\s]$")
                self.assertNotRegex(out.split(".")[0], self.FS_RESERVED)
                # The base a download actually gets. Non-emptiness is asserted here
                # rather than on sanitise() because that is where the guarantee is:
                # sanitise() must return "" for "" so this chain can fall through.
                base = self.base_of(out, raw)
                self.assertTrue(base, "an empty filename downloads as nothing")
                self.assertNotRegex(base, r'[\\/:*?"<>|\x00-\x1f]')

    def test_a_device_name_is_reserved_regardless_of_extension(self):
        """Windows resolves a path component to a device by the part before the first
        dot, so `NUL.txt` is exactly as illegal as `NUL` — and `exportFilename`
        appends an extension, making `NUL` a base name that yields `NUL.pdf`.

        The anchored whole-string pattern this replaced let "nul.txt" through while
        "nul.txt" was already in the shapes list above: assertNotRegex against a
        ^...$ pattern cannot see a stem, so the suite passed on the bug it was
        written to catch.
        """
        for name in ("nul.txt", "CON.md", "com1.csv", "Prn.docx", "lpt3"):
            with self.subTest(name=name):
                self.assertEqual(self.sanitise(name), f"_{name}")
        self.assertEqual(self.sanitise("normal.title"), "normal.title")
        self.assertEqual(self.sanitise("config"), "config")

    def test_empty_input_returns_empty_so_the_fallbacks_can_fire(self):
        """Regression nail. sanitise() used to end in `if not s: s = f"_{s}"`, which
        returns the truthy "_" for empty input, so every `||` downstream was dead.

        Measured in the browser against a page with no session list: exportFilename
        returned "_-2026-09-07-0818.pdf" instead of falling back to the session
        title and then to 对话, and renderExportPane prefilled the 文件名 field with
        "_" instead of "对话". Both looked like a missing title, not like a
        sanitiser bug, which is why the fallback chain is asserted alongside it.
        """
        for empty in ("", "   ", "...", ". . ."):
            with self.subTest(empty=repr(empty)):
                self.assertEqual(self.sanitise(empty), "")
        self.assertEqual(self.base_of("", ""), "对话")
        self.assertEqual(self.base_of("", "真标题"), "真标题")
        self.assertEqual(self.base_of("用户改的名字", "真标题"), "用户改的名字")
        self.assertEqual(self.base_of("...", "..."), "对话")

    def test_the_length_cap_matches_the_history_layer(self):
        """40 is not an arbitrary number: history.py truncates a session title to
        TITLE_CHARS, so a filename built from one can never need more."""
        self.assertEqual(TITLE_CHARS, 40)
        self.assertEqual(len(self.sanitise("长" * 100)), TITLE_CHARS)

    def test_identical_titles_are_separated_by_the_timestamp(self):
        """Two sessions in the real archive share a byte-identical title, so the
        title alone cannot be the filename. exportFilename appends a minute-level
        local timestamp; the same conversation exported twice in the same minute is
        deliberately identically named, because it is identically contented, and a
        genuine collision is the browser's "(1)" to resolve."""
        a = self.sanitise("同一个标题")
        b = self.sanitise("同一个标题")
        self.assertEqual(a, b)
        self.assertNotEqual(f"{a}-2026-09-07-1011.md", f"{a}-2026-09-07-1012.md")
