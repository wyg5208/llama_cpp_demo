"""Turn already-rendered answer HTML into a downloadable file.

Outbound, where documents.py is inbound: nothing is parsed for the model here,
everything is generated for the user, and like documents.py nothing touches the
disk — both writers return bytes.

MD, HTML and CSV never reach this module. The browser already holds the markdown
source and the rendered DOM, so it produces those three itself without a round
trip. Only PDF and DOCX come here, because those two need a real writer.

This module imports nothing from the rest of the package at module level, so the
DOCX writer and the block validation are unit-testable without pymupdf installed
(and without a server), the same way documents.py keeps fit_budget testable.

The DOCX oracle in tests/test_export.py is documents._parse_docx — the very
reader the upload path uses. A file that round-trips through it is well-formed
OOXML carrying the right text, which is the strongest check available on a
machine with no Word.
"""

from __future__ import annotations

import io
import logging
import re
import threading
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from html.parser import HTMLParser

log = logging.getLogger(__name__)


class ExportError(Exception):
    """A rejection whose message is safe to show the user verbatim."""


@dataclass(frozen=True)
class Exported:
    data: bytes
    media_type: str
    ext: str
    pages: int = 0
    truncated: bool = False


# --- limits: module constants, never Settings fields (.env.example says so) ---

# One fit probe costs 17.15 ms measured, and warm-start packing needs about 2.75
# of them per page (measured: 300 blocks -> 16 pages, 44 probes, 0.58 s). 1200
# blocks extrapolates to ~64 pages and ~2.4 s, still inside the worst case this
# app already documents for parsing a 50-page PDF (~20 s). A single answer cannot
# reach it — max_tokens 2048 is ~1400 CJK characters, roughly 60-100 blocks — so
# only a whole-conversation export can.
MAX_EXPORT_BLOCKS = 1200

# 2x documents.DOC_MAX_CHARS. MAX_SESSION_MESSAGES is 400, so 400 turns of ~1000
# characters of rendered HTML lands about here.
MAX_EXPORT_HTML_CHARS = 400_000

# Mirrors documents.DOC_MAX_PAGES = 50 and truncates-with-a-warning the same way
# _parse_pdf does, rather than failing: a 200-page conversation still exports its
# first 60 pages, which beats an error.
MAX_EXPORT_PAGES = 60

PDF_MEDIA = "application/pdf"
DOCX_MEDIA = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


# ---------------------------------------------------------------------------
# Block validation
# ---------------------------------------------------------------------------

# Exactly what renderMarkdown can emit at the top level of its output (app.js
# renderMarkdown): paragraphs, headings, a bare <hr>, <blockquote><p>, lists,
# fenced code and pipe tables. Nothing else — no images, because inlineMarkdown
# has no image branch, so ![alt](src) renders as a literal "!" plus an anchor.
_ROOT_TAGS = frozenset({"p", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol",
                        "blockquote", "pre", "table", "hr"})

# Everything that may appear inside one of those. `p` is legal as a descendant
# too, because blockquote wraps its body in one.
#
# `tbody` is in neither renderMarkdown's output nor the markdown source — it is
# inserted by the browser. blocksFromMarkdown() assigns the rendered HTML to
# element.innerHTML, and the HTML parser always wraps a table's rows in an implied
# <tbody>, so every table block that actually reaches this validator contains one.
# Omitting it rejected every real table export: measured, a two-row table came back
# as "第 7 块含不支持的元素 <tbody>". The DOM's whole table vocabulary is
# {table, tbody, td, th, tr} — the parser never implies a <thead>, so tbody is the
# only addition. Both consumers already ignore it: MuPDF parses it as the table
# section it is, and the DOCX writer keys on table/tr/th/td alone.
_DESC_TAGS = frozenset({"li", "tr", "th", "td", "tbody", "code", "strong", "em",
                        "del", "a"})
_ALL_TAGS = _ROOT_TAGS | _DESC_TAGS

# The only attributes renderMarkdown ever emits: `class` on the <code> inside a
# fence, and href/target/rel on an anchor. Everything else is refused — which is
# what rejects on* handlers, style and src in one rule instead of three.
#
# Refused rather than stripped: stripping means re-serialising HTML, and both
# consumers are already inert to it (MuPDF's HTML engine executes no script at
# all, and the DOCX writer below reads no attribute but href). Building a
# serialiser to remove something that cannot fire is a bad trade, and a block
# carrying onclick is one renderMarkdown could never have produced — so it means
# a hand-made API call, and refusing is the honest answer.
_ALLOWED_ATTRS = frozenset({"href", "class", "target", "rel"})
_HREF_OK = re.compile(r"^https?://", re.IGNORECASE)


class _Validator(HTMLParser):
    """Records what one block is made of. Never raises; the caller decides."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.first: str | None = None
        self.bad_tag: str | None = None
        self.bad_attr: str | None = None
        self.bad_href: str | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if self.first is None:
            self.first = tag
        if tag not in _ALL_TAGS and self.bad_tag is None:
            self.bad_tag = tag
        for key, value in attrs:
            if key not in _ALLOWED_ATTRS:
                if self.bad_attr is None:
                    self.bad_attr = key
            elif key == "href" and not _HREF_OK.match(value or ""):
                # Re-checked here even though safeLink (app.js:16-20) already
                # restricts to http(s) in the browser: /api/export is directly
                # callable. Same reasoning as StoredSource._check_url in main.py.
                if self.bad_href is None:
                    self.bad_href = value or ""


def check_blocks(blocks: Sequence[str]) -> list[str]:
    """Validate rendered blocks. Raises ExportError with a user-safe message."""
    if not blocks:
        raise ExportError("没有可导出的内容。")
    if len(blocks) > MAX_EXPORT_BLOCKS:
        raise ExportError(f"导出内容共 {len(blocks)} 块，超过 {MAX_EXPORT_BLOCKS} 块上限。")
    total = sum(len(b) for b in blocks)
    if total > MAX_EXPORT_HTML_CHARS:
        raise ExportError(
            f"导出内容约 {total // 1000}k 字符，超过 {MAX_EXPORT_HTML_CHARS // 1000}k 上限。"
        )

    for index, block in enumerate(blocks):
        where = f"第 {index + 1} 块"
        if not block.strip():
            raise ExportError(f"{where}是空的。")
        v = _Validator()
        v.feed(block)
        v.close()
        if v.first is None:
            raise ExportError(f"{where}不含任何 HTML 元素，无法导出。")
        if v.first not in _ROOT_TAGS:
            raise ExportError(f"{where}的根元素 <{v.first}> 不是可导出的块级元素。")
        if v.bad_tag is not None:
            raise ExportError(f"{where}含不支持的元素 <{v.bad_tag}>，已拒绝导出。")
        if v.bad_attr is not None:
            raise ExportError(f"{where}含不支持的属性 {v.bad_attr}，已拒绝导出。")
        if v.bad_href is not None:
            raise ExportError(f"{where}的链接不是 http(s)，已拒绝导出。")
    return list(blocks)


# ---------------------------------------------------------------------------
# Shared XML escaping
# ---------------------------------------------------------------------------


def _esc_text(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _esc_attr(text: str) -> str:
    return _esc_text(text).replace('"', "&quot;").replace("'", "&apos;")


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

# A4 in points, and a 36 pt (0.5 inch) margin. Hard-coded because pymupdf is
# imported lazily inside make_pdf, so paper_rect() is not available up here.
_PAGE_W, _PAGE_H = 595.0, 842.0
_WHERE = (36.0, 36.0, _PAGE_W - 36.0, _PAGE_H - 36.0)

# A light re-authoring of the .md rules in static/style.css:373-400. The three
# copies (this, EXPORT_CSS in app.js, and style.css) can drift; accepted, because
# they target different engines — a browser and MuPDF's HTML subset, which
# supports only a fraction of CSS — so sharing them would need a build step this
# app does not have.
#
# Three values are load-bearing. `pre` must be #f6f7f9 and NOT style.css's
# #10131a, which prints as a solid black block. The body font must be stated: omit
# it and MuPDF silently falls back to something that may not cover CJK. And
# overflow-wrap:break-word must be there at all.
#
# Without it MuPDF neither wraps inside <pre> nor breaks an unbreakable token in
# prose, so the line overflows the box sideways and insert_htmlbox shrinks the
# WHOLE page to compensate — measured scale 0.2873 for one 300-character code line
# and 0.2375 for a 400-character token in a <p>, identical at every line count, so
# no amount of vertical splitting in _pack can fix it. A long URL in an answer is
# enough to trigger it. With the rule all three measured cases return to 1.0000,
# and because break-word only fires where a word would otherwise overflow, normal
# prose still wraps at word boundaries exactly as before.
#
# MuPDF's CSS subset is picky about spelling. Measured inert, so do not "tidy" the
# rule into any of them: word-break:break-all, overflow-wrap:anywhere, and the
# legacy alias word-wrap:break-word all leave the scale at 0.2873.
_PDF_CSS = (
    "*{overflow-wrap:break-word}"
    "body{font-family:sans-serif;font-size:11px;color:#1f2328}"
    "p{margin:0 0 8px}"
    "h1{font-size:19px}h2{font-size:17px}h3{font-size:15px}"
    "h4,h5,h6{font-size:13px}"
    "ul,ol{margin:0 0 8px;padding-left:20px}"
    "li{margin:2px 0}"
    "blockquote{margin:0 0 8px;padding:2px 10px;border-left:3px solid #0b57d0;color:#57606a}"
    "pre{background:#f6f7f9;border:1px solid #d0d7de;padding:8px;margin:0 0 8px}"
    "code{font-family:monospace;font-size:10px}"
    # width:100% is load-bearing too, and for ordinary tables rather than exotic
    # ones: without it MuPDF sizes an auto table to its content and a plain 30-row
    # table of short cells lands at scale 0.8634, a 37-row one at 0.6840. With it
    # both return to 1.0000. It also matches the screen, where .md table is
    # width:100% (style.css:397). table-layout:fixed measured identically on every
    # case down to the column x-positions, so the simpler declaration wins.
    "table{border-collapse:collapse;width:100%}"
    "th,td{border:1px solid #999999;padding:3px;text-align:left}"
    "th{background:#eeeeee}"
    "hr{border:none;border-top:1px solid #c9ccd1}"
    "a{color:#0b57d0}"
)

# Pre-split thresholds. Deliberately row/line counts rather than fit probes:
# probing all 1200 blocks would cost 17.15 ms x 1200 = 20 s, while these two rules
# cost nothing and catch the only blocks that can be atomically too tall for a page.
#
# Measured against the 770 pt content area with short cells: a table row is 20.2 pt
# and 37 rows is the page maximum (38 -> scale 0.9752), while a <pre> line is 12.2 pt
# and 63 lines is the maximum (64 -> 0.9894). So 40 body rows plus the repeated
# header -- the obvious round number, and what this used to be -- overshot at 41 and
# rendered every page of a long table at scale 0.93. 30 leaves six rows for cells
# that wrap, and 40 leaves 23 lines for the same reason.
#
# Both are only a first guess: _pack halves any chunk that is still too TALL for a
# page, so wrapping-heavy content converges instead of shrinking. Width is the one
# thing splitting cannot fix, and MuPDF sizes a table column to its longest
# unbreakable token — measured, a cell holding 180 ASCII characters pins the page to
# scale 0.9086 no matter how few rows it has (60 such rows export as 52 pages, 48 of
# them at 0.91). No CSS reaches it: word-break:break-all and overflow-wrap:anywhere
# are both inert in MuPDF's subset, and overflow-wrap:break-word, which does fix
# <pre> and prose, does not influence column min-width. Cells that merely wrap — the
# realistic case, CJK sentences and URLs up to ~90 characters — measure 1.0000.
_PRE_SPLIT_LINES = 40
_TABLE_SPLIT_ROWS = 30

# Serialises rendering. MuPDF is a C library and run_in_threadpool makes exports
# concurrent, so this both sidesteps its thread-safety questions and bounds peak
# RSS to one render at a time. This module's own lock, not documents._PDF_LOCK:
# importing that would break the "no package imports" rule at the top.
_RENDER_LOCK = threading.Lock()


class _PreText(HTMLParser):
    """Pull the text and the <code> attributes back out of one <pre> block."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.code_attrs = ""

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "code" and not self.code_attrs:
            self.code_attrs = " ".join(f'{k}="{_esc_attr(v or "")}"' for k, v in attrs)

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _split_pre(block: str, per: int) -> list[str]:
    # Guarded on exactly one <pre>: with two, the text pulled out would be the
    # concatenation of both and re-emitting it would merge them into one block.
    if block.count("<pre") != 1:
        return [block]
    p = _PreText()
    p.feed(block)
    p.close()
    lines = "".join(p.parts).split("\n")
    if len(lines) <= per:
        return [block]
    attrs = f" {p.code_attrs}" if p.code_attrs else ""
    return [
        "<pre><code%s>%s</code></pre>"
        % (attrs, _esc_text("\n".join(lines[i:i + per])))
        for i in range(0, len(lines), per)
    ]


def _split_table(block: str, per: int) -> list[str]:
    # Guarded on exactly one <table>, which is what makes the non-greedy row
    # regex below safe: rows can only nest inside a nested table, so with one
    # table every <tr> is a sibling. The header row is repeated on every chunk so
    # page 2 of a long table still says what its columns are.
    if block.count("<table") != 1:
        return [block]
    rows = [m.group(0) for m in re.finditer(r"<tr\b[^>]*>.*?</tr>", block, re.S)]
    if len(rows) <= per:
        return [block]
    opener = re.match(r"\s*<table\b[^>]*>", block)
    open_tag = opener.group(0) if opener else "<table>"
    header = rows[0] if "<th" in rows[0] else ""
    body = rows[1:] if header else rows
    return [
        open_tag + "".join(([header] if header else []) + body[i:i + per]) + "</table>"
        for i in range(0, len(body), per)
    ]


def _expand(blocks: Sequence[str]) -> list[str]:
    out: list[str] = []
    for block in blocks:
        if block.lstrip().startswith("<pre"):
            out.extend(_split_pre(block, _PRE_SPLIT_LINES))
        elif block.lstrip().startswith("<table"):
            out.extend(_split_table(block, _TABLE_SPLIT_ROWS))
        else:
            out.append(block)
    return out


def _halve(atom: str) -> list[str]:
    """Split again an atom that is still too tall for a page on its own.

    The static thresholds in _expand are guesses about how tall a line is, and a
    line that wraps is taller than one that does not. This is the correction: it
    re-splits at half the current count, so a chunk made of nothing but long
    wrapping lines converges on a size that fits instead of being shrunk.

    Returns [atom] when nothing more can be done — one line, one row, or a tag
    this module does not split — which is _pack's signal to put the atom alone on
    a page and let make_pdf log the shrink.
    """
    text = atom.lstrip()
    if text.startswith("<pre"):
        p = _PreText()
        p.feed(atom)
        p.close()
        lines = len("".join(p.parts).split("\n"))
        return _split_pre(atom, lines // 2) if lines > 1 else [atom]
    if text.startswith("<table"):
        rows = len(re.findall(r"<tr\b", atom))
        return _split_table(atom, rows // 2) if rows > 1 else [atom]
    return [atom]


def _pack(atoms: list[str], scale_of) -> tuple[list[str], bool]:
    """Group atoms into page-sized HTML strings.

    Warm-start linear rather than binary search: consecutive pages hold similar
    amounts, so seeding each guess from the previous page's count means the
    halving loop almost never runs and the grow loop costs one or two probes.
    Measured 300 blocks -> 16 pages in 44 probes / 0.58 s.

    Normal pages are packed to scale 1.0 and never deliberately shrunk. The one
    exception is an atom too big for a page on its own, and there the two
    directions have to be told apart:

    * Too *tall* — a table whose wrapped cells exceed the page. Halving converges,
      so keep halving.
    * Too *wide* — a table whose columns want more width than the box. The scale
      is then set by width alone and halving rows leaves it untouched, so splitting
      only multiplies pages: measured, 60 such rows became 60 pages at scale 0.9086
      where 6 pages at the same 0.9086 was available. A 9% shrink on a well-filled
      page is readable; one row per page is not.

    Probing the half against the whole separates them for one extra probe, and when
    halving cannot help the atom takes a page alone, shrinks, and make_pdf says so.
    """
    pages: list[str] = []
    i = 0
    guess = 24
    while i < len(atoms):
        if len(pages) >= MAX_EXPORT_PAGES:
            return pages, True
        n = min(guess, len(atoms) - i)
        while n > 1 and scale_of("".join(atoms[i:i + n])) < 1.0:
            n //= 2
        alone = scale_of(atoms[i]) if n == 1 else 1.0
        if n == 1 and alone < 1.0:
            smaller = _halve(atoms[i])
            if len(smaller) > 1 and scale_of(smaller[0]) > alone + 0.001:
                atoms[i:i + 1] = smaller
                # guess drops to 1: the halves are what is now known to fit, and the
                # grow loop packs them back out on the next pass.
                guess = 1
                continue
        else:
            while i + n < len(atoms) and scale_of("".join(atoms[i:i + n + 1])) >= 1.0:
                n += 1
        pages.append("".join(atoms[i:i + n]))
        guess = n
        i += n
    return pages, False


def make_pdf(blocks: Sequence[str], title: str = "") -> Exported:
    # Imported here, never at module level. Measured: import pymupdf alone costs
    # 0.064 s and 45 MB of RSS — not the 1.03 s / 126 MB that documents.py hides,
    # which is pymupdf4llm dragging in numpy, onnxruntime and networkx. Export
    # needs only pymupdf. The real reason to keep it lazy is that it leaves
    # make_docx and check_blocks testable with pymupdf absent; 64 ms on its own
    # would not be worth hiding.
    import pymupdf

    atoms = _expand(blocks)
    if not atoms:
        raise ExportError("没有可导出的内容。")

    def scale_of(html: str) -> float:
        """Render html into a scratch page and return insert_htmlbox's scale factor.

        Its return value is (spare_height, scale), and the second element is the
        load-bearing one: scale < 1.0 means everything was silently reduced to fit
        the rect. Reading only the first element reports success for a page of
        microscopic type — measured, 60 paragraphs return (0.0, 0.5303) and 200
        return (0.0, 0.1591), i.e. 5.8pt and 1.7pt text, and every text probe
        still "passes" because the glyphs are there. The spare height is not just
        uninformative but misleading: it reads 0.0 both for a page that is exactly
        full at scale 1.0 and for one shrunk to a third.

        The float rather than a bool because _pack compares an oversized atom with
        its own half to tell "too tall" from "too wide".

        A scratch document that is never saved, so the per-page font duplication
        below costs nothing here.
        """
        scratch = pymupdf.open()
        try:
            page = scratch.new_page(width=_PAGE_W, height=_PAGE_H)
            return page.insert_htmlbox(_WHERE, html, css=_PDF_CSS)[1]
        finally:
            scratch.close()

    with _RENDER_LOCK:
        pages, truncated = _pack(atoms, scale_of)
        doc = pymupdf.open()
        try:
            if title:
                doc.set_metadata({"title": title, "producer": "llama_cpp_demo"})
            scales = []
            for html in pages:
                page = doc.new_page(width=_PAGE_W, height=_PAGE_H)
                scales.append(page.insert_htmlbox(_WHERE, html, css=_PDF_CSS)[1])
            doc.subset_fonts()
            # garbage=4 is load-bearing, not a tidy-up. Measured on a 6-page
            # document: tobytes() = 18,502,649 bytes, tobytes(garbage=4,
            # deflate=True) = 27,206. insert_htmlbox embeds the CJK font once per
            # page and plain tobytes() keeps every copy; subset_fonts() alone does
            # not collapse them.
            data = doc.tobytes(garbage=4, deflate=True)
        finally:
            doc.close()

    for index, scale in enumerate(scales):
        if scale < 1.0:
            log.warning("export pdf page %d shrank to scale %.3f", index + 1, scale)
    return Exported(data=data, media_type=PDF_MEDIA, ext="pdf",
                    pages=len(pages), truncated=truncated)


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

# A4 in twips, 2 cm margins. TEXT_W is what the table column widths divide up.
_TWIPS_W, _TWIPS_H = 11906, 16838
_MARGIN = 1134
_TEXT_W = _TWIPS_W - 2 * _MARGIN

_XML_DECL = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
_NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
_NS_CT = "http://schemas.openxmlformats.org/package/2006/content-types"
_REL_DOC = f"{_NS_R}/officeDocument"
_REL_STYLES = f"{_NS_R}/styles"


class _BlockWriter(HTMLParser):
    """One top-level HTML block in, a list of <w:p>/<w:tbl> strings out.

    convert_charrefs=True, so entities arrive already decoded (&amp; -> &) and
    every text run has to be re-escaped for XML on the way out. Void elements
    emit handle_starttag with no matching handle_endtag — <hr> is the one
    renderMarkdown produces — so hr is closed in starttag, not endtag.

    Unknown tags are ignored rather than fatal: check_blocks has already refused
    them, and a writer that crashes on stray input is worse than one that keeps
    the text.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.links: list[str] = []
        # Stack of run sinks. The bottom level is always the paragraph being
        # built; <a> and <th>/<td> push a level so their runs can be wrapped or
        # collected on the way back out. Entry = [kind, runs, extra].
        self._stack: list[list] = [["block", [], None]]
        self._ppr = ""
        self._inline = {"strong": 0, "em": 0, "del": 0, "code": 0}
        self._quote = 0
        self._link = 0
        # Depth of <th> nesting. Header cells are bolded through here rather than
        # by rewriting the runs afterwards: CT_RPr has a schema-defined child
        # order (rFonts before b), so splicing <w:b/> into serialised XML puts it
        # in the wrong place for any run that also carries a code font.
        self._header = 0
        self._list: str | None = None
        self._ol = 0
        self._in_pre = False
        self._pre: list[str] = []
        self._rows: list[list] = []
        self._row: list = []

    # -- helpers ----------------------------------------------------------

    @property
    def _sink(self) -> list[str]:
        return self._stack[-1][1]

    def _flags(self) -> dict:
        return {
            "b": self._inline["strong"] or self._header,
            "i": self._inline["em"],
            "strike": self._inline["del"],
            # Not inside <pre>: there the code tag is the block's own container,
            # and the Code paragraph style already carries the font.
            "code": self._inline["code"] and not self._in_pre,
            "color": "0B57D0" if self._link else ("444444" if self._quote else ""),
            "underline": bool(self._link),
        }

    def _open_block(self, ppr: str = "") -> None:
        self._stack[0][1] = []
        self._ppr = ppr

    def _flush_para(self) -> None:
        self.out.append(f"<w:p>{self._ppr}{''.join(self._stack[0][1])}</w:p>")
        self._stack[0][1] = []
        self._ppr = ""

    # -- parser hooks -----------------------------------------------------

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("p",):
            # Inside a list item or a fence there is no separate <p>, and inside a
            # blockquote the <p> IS the block, so it carries the quote's indent.
            if self._in_pre:
                return
            if self._quote:
                self._open_block(
                    '<w:pPr><w:pBdr><w:left w:val="single" w:sz="12" w:space="8"'
                    ' w:color="0B57D0"/></w:pBdr><w:ind w:left="420"/></w:pPr>'
                )
            elif self._list is None:
                self._open_block()
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._open_block(f'<w:pPr><w:pStyle w:val="Heading{tag[1]}"/></w:pPr>')
            return
        if tag == "hr":
            # Void: no endtag will ever arrive, so emit here.
            self.out.append(
                "<w:p><w:pPr><w:pBdr><w:bottom w:val=\"single\" w:sz=\"6\""
                ' w:space="1" w:color="C9CCD1"/></w:pBdr></w:pPr></w:p>'
            )
            return
        if tag in ("ul", "ol"):
            self._list = tag
            self._ol = 0
            return
        if tag == "li":
            # Literal bullet/number rather than w:numPr. numPr's whole purpose is
            # unreachable here (renderMarkdown keeps listTag as one scalar and
            # discards indentation, so there are no nested lists and w:ilvl would
            # always be 0), it would cost two more zip parts, each <ol> would need
            # its own abstractNum to restart at 1 — and decisively, a numPr bullet
            # is synthesised by Word at render time and never exists in w:t, so no
            # stdlib test could ever confirm it. A literal one round-trips through
            # documents._para_text. The honest cost: these are not real Word lists,
            # so list-gallery restyling and screen-reader "bullet, 1 of 3" do not
            # apply. Accepted — the artifact is for reading elsewhere.
            if self._list == "ol":
                self._ol += 1
                prefix = f"{self._ol}.\u00a0\u00a0"
            else:
                prefix = "\u2022\u00a0\u00a0"
            self._open_block('<w:pPr><w:ind w:left="420" w:hanging="240"/></w:pPr>')
            self._sink.append(_run(prefix, self._flags()))
            return
        if tag == "blockquote":
            self._quote += 1
            return
        if tag == "pre":
            self._in_pre = True
            self._pre = []
            return
        if tag == "table":
            self._rows = []
            return
        if tag == "tr":
            self._row = []
            return
        if tag in ("th", "td"):
            if tag == "th":
                self._header += 1
            self._stack.append(["cell", [], tag == "th"])
            return
        if tag == "a":
            url = dict(attrs).get("href") or ""
            # rId1 belongs to styles.xml in document.xml.rels, so links start at 2.
            rid = f"rId{len(self.links) + 2}"
            self.links.append(url)
            self._stack.append(["a", [], rid])
            self._link += 1
            return
        if tag in self._inline:
            self._inline[tag] += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._inline:
            self._inline[tag] = max(0, self._inline[tag] - 1)
            return
        if tag == "a":
            if self._stack[-1][0] != "a":
                return
            _, runs, rid = self._stack.pop()
            self._link = max(0, self._link - 1)
            self._sink.append(f'<w:hyperlink r:id="{rid}">{"".join(runs)}</w:hyperlink>')
            return
        if tag in ("th", "td"):
            if self._stack[-1][0] != "cell":
                return
            _, runs, is_header = self._stack.pop()
            if is_header:
                self._header = max(0, self._header - 1)
            self._row.append((is_header, runs))
            return
        if tag == "tr":
            self._rows.append(self._row)
            self._row = []
            return
        if tag == "table":
            self.out.append(self._build_table())
            self._rows = []
            self._stack[0][1] = []
            self._ppr = ""
            return
        if tag == "pre":
            self._in_pre = False
            self.out.append(self._build_pre())
            return
        if tag == "blockquote":
            self._quote = max(0, self._quote - 1)
            return
        if tag in ("ul", "ol"):
            self._list = None
            return
        if tag == "li" or tag == "p" or tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            # A <p> that only opened a list item or a quote still flushes here,
            # because that is where its runs went.
            if self._stack[0][1] or self._ppr:
                self._flush_para()

    def handle_data(self, data: str) -> None:
        if self._in_pre:
            # Raw: whitespace inside a fence is the content, not formatting.
            self._pre.append(data)
            return
        # Collapse to match the screen: .md p (style.css:375) sets no white-space,
        # so a single newline already renders as a space. A chunk that collapses
        # to one space at the start of a paragraph is dropped, or the bullet
        # prefix would be followed by a doubled gap.
        text = re.sub(r"\s+", " ", data)
        if text == " " and not self._sink:
            return
        if not text:
            return
        self._sink.append(_run(text, self._flags()))

    # -- block builders ---------------------------------------------------

    def _build_pre(self) -> str:
        # One paragraph with w:br between lines, not one paragraph per line: the
        # shading then covers the whole fence as one contiguous block. _para_text
        # turns w:br back into "\n", so the reader still sees the line structure.
        lines = "".join(self._pre).split("\n")
        runs: list[str] = []
        for i, line in enumerate(lines):
            if i:
                runs.append("<w:r><w:br/></w:r>")
            if line:
                runs.append(_run(line, {"code": True}))
        ppr = '<w:pPr><w:pStyle w:val="Code"/></w:pPr>'
        return f"<w:p>{ppr}{''.join(runs)}</w:p>"

    def _build_table(self) -> str:
        rows = [r for r in self._rows if r]
        if not rows:
            return ""
        cols = max(len(r) for r in rows)
        width = _TEXT_W // cols
        grid = "".join(f'<w:gridCol w:w="{width}"/>' for _ in range(cols))
        borders = "".join(
            f'<w:{edge} w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
            for edge in ("top", "left", "bottom", "right", "insideH", "insideV")
        )
        out = [
            "<w:tbl><w:tblPr>",
            '<w:tblW w:w="0" w:type="auto"/>',
            f"<w:tblBorders>{borders}</w:tblBorders>",
            "</w:tblPr>",
            f"<w:tblGrid>{grid}</w:tblGrid>",
        ]
        for row in rows:
            out.append("<w:tr>")
            # Padded to `cols` so a short row cannot produce a ragged grid.
            for index in range(cols):
                is_header, runs = row[index] if index < len(row) else (False, [])
                shd = '<w:shd w:val="clear" w:color="auto" w:fill="EEEEEE"/>' if is_header else ""
                # Every w:tc must end in a w:p, even an empty one, or Word calls
                # the file corrupt.
                body = "".join(runs)
                out.append(
                    "<w:tc><w:tcPr>"
                    f'<w:tcW w:w="{width}" w:type="dxa"/>{shd}'
                    f"</w:tcPr><w:p>{body}</w:p></w:tc>"
                )
            out.append("</w:tr>")
        out.append("</w:tbl>")
        return "".join(out)


def _run(text: str, flags: dict) -> str:
    if not text:
        return ""
    # CT_RPr has a schema-defined child order; Word tolerates some deviation but
    # a valid file is the whole point, so emit in it: rFonts, b, i, strike, color,
    # u, shd.
    rpr: list[str] = []
    if flags.get("code"):
        rpr.append('<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/>')
    if flags.get("b"):
        rpr.append("<w:b/>")
    if flags.get("i"):
        rpr.append("<w:i/>")
    if flags.get("strike"):
        rpr.append("<w:strike/>")
    if flags.get("color"):
        rpr.append(f'<w:color w:val="{flags["color"]}"/>')
    if flags.get("underline"):
        rpr.append('<w:u w:val="single"/>')
    if flags.get("code"):
        rpr.append('<w:shd w:val="clear" w:color="auto" w:fill="F0F0F0"/>')
    props = f"<w:rPr>{''.join(rpr)}</w:rPr>" if rpr else ""
    return f'<w:r>{props}<w:t xml:space="preserve">{_esc_text(text)}</w:t></w:r>'


_DOC_MAIN_CT = "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
_STYLES_CT = "application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"


def _content_types() -> str:
    return (
        _XML_DECL
        + f'<Types xmlns="{_NS_CT}">'
        + '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        + '<Default Extension="xml" ContentType="application/xml"/>'
        + f'<Override PartName="/word/document.xml" ContentType="{_DOC_MAIN_CT}"/>'
        + f'<Override PartName="/word/styles.xml" ContentType="{_STYLES_CT}"/>'
        + "</Types>"
    )


def _root_rels() -> str:
    return (
        _XML_DECL
        + f'<Relationships xmlns="{_NS_PKG_REL}">'
        + f'<Relationship Id="rId1" Type="{_REL_DOC}" Target="word/document.xml"/>'
        + "</Relationships>"
    )


def _document_rels(links: Sequence[str]) -> str:
    out = [_XML_DECL, f'<Relationships xmlns="{_NS_PKG_REL}">',
           f'<Relationship Id="rId1" Type="{_REL_STYLES}" Target="word/styles.xml"/>']
    for index, url in enumerate(links):
        out.append(
            f'<Relationship Id="rId{index + 2}" Type="{_NS_R}/hyperlink" '
            f'Target="{_esc_attr(url)}" TargetMode="External"/>'
        )
    out.append("</Relationships>")
    return "".join(out)


_HEADING_SZ = {1: 36, 2: 32, 3: 28, 4: 26, 5: 24, 6: 22}


def _styles() -> str:
    out = [_XML_DECL, f'<w:styles xmlns:w="{_NS_W}">', "<w:docDefaults>",
           "<w:rPrDefault><w:rPr>",
           # w:eastAsia is what makes CJK render. Without it Word substitutes a
           # fallback and the file looks wrong even though every byte is valid.
           '<w:rFonts w:ascii="Calibri" w:eastAsia="Microsoft YaHei" w:hAnsi="Calibri" w:cs="Calibri"/>',
           '<w:sz w:val="22"/><w:szCs w:val="22"/>',
           "</w:rPr></w:rPrDefault>",
           '<w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="276" w:lineRule="auto"/></w:pPr></w:pPrDefault>',
           "</w:docDefaults>",
           '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>']
    for level, size in _HEADING_SZ.items():
        # outlineLvl is what makes Word's navigation pane and its heading styles
        # recognise these as headings rather than as bold paragraphs.
        out.append(
            f'<w:style w:type="paragraph" w:styleId="Heading{level}">'
            f'<w:name w:val="heading {level}"/><w:basedOn w:val="Normal"/>'
            f'<w:pPr><w:keepNext/><w:outlineLvl w:val="{level - 1}"/></w:pPr>'
            f'<w:rPr><w:b/><w:sz w:val="{size}"/><w:szCs w:val="{size}"/></w:rPr></w:style>'
        )
    out.append(
        '<w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/>'
        '<w:basedOn w:val="Normal"/>'
        # #f6f7f9, matching the light re-authoring of .md pre — never style.css's
        # #10131a, which is a dark-theme value.
        '<w:pPr><w:shd w:val="clear" w:color="auto" w:fill="F6F7F9"/>'
        '<w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr>'
        '<w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/>'
        '<w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr></w:style>'
    )
    out.append("</w:styles>")
    return "".join(out)


def _document_xml(blocks_xml: str) -> str:
    sect = (
        "<w:sectPr>"
        f'<w:pgSz w:w="{_TWIPS_W}" w:h="{_TWIPS_H}"/>'
        f'<w:pgMar w:top="{_MARGIN}" w:right="{_MARGIN}" w:bottom="{_MARGIN}"'
        f' w:left="{_MARGIN}" w:header="720" w:footer="720" w:gutter="0"/>'
        "</w:sectPr>"
    )
    # xmlns:r must be declared here or w:hyperlink r:id cannot resolve.
    return (
        _XML_DECL
        + f'<w:document xmlns:w="{_NS_W}" xmlns:r="{_NS_R}">'
        + f"<w:body>{blocks_xml}{sect}</w:body></w:document>"
    )


def make_docx(blocks: Sequence[str]) -> Exported:
    """Five zip parts, and no docProps/core.xml.

    Five is what round-trips through documents._parse_docx, which is the oracle
    the tests use; a sixth part is one nothing can validate and no reader needs.
    """
    writer = _BlockWriter()
    # All blocks through one parser instance, closed once at the end: the link rId
    # numbering is cross-block state, and close() only flushes buffered text.
    for block in blocks:
        writer.feed(block)
    writer.close()
    body = "".join(part for part in writer.out if part)
    if not body.strip():
        raise ExportError("没有可导出的内容。")

    parts = {
        "[Content_Types].xml": _content_types(),
        "_rels/.rels": _root_rels(),
        "word/document.xml": _document_xml(body),
        "word/_rels/document.xml.rels": _document_rels(writer.links),
        "word/styles.xml": _styles(),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, text in parts.items():
            zf.writestr(name, text.encode("utf-8"))
    return Exported(data=buf.getvalue(), media_type=DOCX_MEDIA, ext="docx")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def build(fmt: str, blocks: Sequence[str], title: str = "") -> Exported:
    """PDF or DOCX only.

    md, html and csv raise here on purpose: they are the browser's job, since it
    already holds the markdown source and the rendered DOM. Refusing them is what
    stops a server-side markdown dependency from growing later.
    """
    if fmt == "pdf":
        return make_pdf(blocks, title)
    if fmt == "docx":
        return make_docx(blocks)
    raise ExportError(f"不支持的导出格式：{fmt}。")
