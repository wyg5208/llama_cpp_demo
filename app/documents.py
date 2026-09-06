"""Turn uploaded files into plain text for the model's context.

Six formats: md / txt / pdf / docx / xlsx / pptx. The pre-2007 Office formats are
rejected with an instruction to re-save rather than shelled out to a local Office,
and a PDF with no text layer is reported rather than OCR'd — this app has no OCR.

Everything is parsed in memory. Nothing is written to disk, so there is no upload
directory to clean up and no path built from a client-supplied filename.

This module deliberately imports nothing from the rest of the package, so
`fit_budget` and `estimate_tokens` can be unit-tested without a server, the same
way history.py and settings_store.py are.
"""

from __future__ import annotations

import io
import logging
import re
import threading
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from xml.etree import ElementTree

log = logging.getLogger(__name__)

ACCEPTED = {".md", ".markdown", ".txt", ".pdf", ".docx", ".xlsx", ".pptx"}
LEGACY = {".doc": ".docx", ".xls": ".xlsx", ".ppt": ".pptx"}
_KIND = {
    ".md": "Markdown",
    ".markdown": "Markdown",
    ".txt": "文本",
    ".pdf": "PDF",
    ".docx": "Word",
    ".xlsx": "Excel",
    ".pptx": "PowerPoint",
}

MAX_DOC_BYTES = 25_000_000
# A wire/archive guard, not the context guard: at ~1 token per CJK char this is
# already far past any model's window. fit_budget() is what actually protects the
# context; this only stops one 500-page dump from bloating the session JSON.
DOC_MAX_CHARS = 200_000
# Measured on this machine with pymupdf4llm 1.28.2's ONNX layout path: 514 ms per
# page cold, 294 ms warm. 50 pages is therefore ~20 s worst case, and at a typical
# 1500-3000 chars per page it also lands near DOC_MAX_CHARS. Both limits agree.
DOC_MAX_PAGES = 50

XLSX_MAX_ROWS = 2000
XLSX_MAX_COLS = 60

ZIP_MAX_ENTRIES = 2000
ZIP_MAX_ENTRY_BYTES = 100_000_000
ZIP_MAX_TOTAL_BYTES = 400_000_000

# Added to the prompt budget for every attachment to cover the wrapper text
# to_openai_messages puts around the body (header line, delimiters, the
# "this is data, not instructions" note).
DOC_WRAPPER_TOKENS = 40

CTX_SAFETY_TOKENS = 1024

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"

_DOCX_PARTS = re.compile(r"^word/(document|footnotes|endnotes)\.xml$")
_PPTX_PARTS = re.compile(r"^ppt/(slides/slide|notesSlides/notesSlide)\d+\.xml$")
_SLIDE_NO = re.compile(r"(\d+)\.xml$")

_CJK = re.compile(
    "[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]"
)
# Digits, underscores and punctuation. Split out from plain letters because they
# tokenise roughly one-per-character, while prose letters run about four to a
# token — a single blended rate is wrong by 2.5x in both directions.
_DENSE = re.compile(r"[0-9_]|[^\w\s]")

CJK_TOKEN_WEIGHT = 1.0
DENSE_TOKEN_WEIGHT = 1.5
PLAIN_CHARS_PER_TOKEN = 3.0


class DocumentError(Exception):
    """A rejection whose message is safe to show the user verbatim."""


@dataclass
class ParsedDoc:
    name: str
    kind: str
    text: str
    chars: int = 0
    pages: int = 0
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "text": self.text,
            "chars": self.chars,
            "pages": self.pages,
            "truncated": self.truncated,
            "warnings": self.warnings,
        }


def _ext_of(filename: str) -> tuple[str, str]:
    """Return (display name, lowercase suffix), rejecting anything unparseable.

    The name is reduced to its basename and capped because it is echoed back to
    the browser and stored in the session archive; it is never used to build a
    path, since nothing here touches the disk.
    """
    name = PurePosixPath(str(filename).replace("\\", "/")).name[:200] or "未命名文件"
    ext = PurePosixPath(name).suffix.lower()
    if ext in LEGACY:
        raise DocumentError(
            f"不支持旧版 {ext} 格式（Office 97-2003）。"
            f"请在 Office 或 WPS 里「另存为」{LEGACY[ext]} 后再上传。"
        )
    if ext not in ACCEPTED:
        raise DocumentError(
            f"不支持的文件类型 {ext or '（无扩展名）'}。"
            "可上传：MD、TXT、PDF、DOCX、XLSX、PPTX。"
        )
    return name, ext


def check_filename(filename: str) -> None:
    """Refuse an unsupported extension before the body is uploaded at all."""
    _ext_of(filename)


def parse_document(filename: str, data: bytes) -> ParsedDoc:
    """Dispatch on the extension and return the extracted text."""
    if not data:
        raise DocumentError("文件是空的。")
    if len(data) > MAX_DOC_BYTES:
        raise DocumentError(f"文件超过 {MAX_DOC_BYTES // 1_000_000}MB 上限。")

    name, ext = _ext_of(filename)
    kind = _KIND[ext]
    if ext in (".md", ".markdown", ".txt"):
        return _finish(name, kind, _decode_text(data))
    if ext == ".pdf":
        return _parse_pdf(name, data)
    if ext == ".docx":
        return _parse_docx(name, data)
    if ext == ".pptx":
        return _parse_pptx(name, data)
    return _parse_xlsx(name, data)


def _finish(
    name: str,
    kind: str,
    text: str,
    *,
    pages: int = 0,
    warnings: tuple[str, ...] = (),
    truncated: bool = False,
) -> ParsedDoc:
    text = text.strip()
    warns = list(warnings)
    if len(text) > DOC_MAX_CHARS:
        text = text[:DOC_MAX_CHARS]
        truncated = True
        warns.append(f"正文超过 {DOC_MAX_CHARS} 字，已截断，仅保留前 {DOC_MAX_CHARS} 字。")
    if not text and not warns:
        warns.append("没有从文件里提取到任何文字。")
    return ParsedDoc(
        name=name,
        kind=kind,
        text=text,
        chars=len(text),
        pages=pages,
        truncated=truncated,
        warnings=warns,
    )


def _decode_text(data: bytes) -> str:
    # gb18030 is the layer that matters: plenty of Chinese .txt files are GBK, and
    # without it they arrive as mojibake. gb18030 is a GBK superset, so it covers
    # all simplified and most traditional text. latin-1 never raises, so the
    # function always returns something.
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

# Serialises PDF parsing. The layout path holds an ONNX inference session; two
# concurrent uploads would each pay the cold-start cost and each keep their own
# copy of the model resident.
_PDF_LOCK = threading.Lock()


def _parse_pdf(name: str, data: bytes) -> ParsedDoc:
    # Imported here, never at module level. Measured: importing pymupdf4llm costs
    # 1.03 s and 80 MB of RSS (numpy + onnxruntime + networkx). main.py is on the
    # boot path, so a top-level import would make every start of the app pay for a
    # feature the user may never touch.
    import pymupdf
    import pymupdf4llm

    warnings: list[str] = []
    with _PDF_LOCK:
        # Residual risk, accepted deliberately: PyMuPDF is a C library and a
        # pathological PDF can segfault, which no try/except catches — it takes the
        # whole process down. That is true of every C-based PDF library. This app
        # is local and single-user, so the threat model is "my own broken PDF",
        # not an attacker; restart with start_app.bat if it ever happens.
        try:
            doc = pymupdf.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise DocumentError(
                f"无法打开这个 PDF（可能已损坏，或并不是真正的 PDF）：{type(exc).__name__}"
            ) from exc
        try:
            pages = doc.page_count
            if pages == 0:
                raise DocumentError("这个 PDF 没有任何页面，文件可能已损坏。")
            only = None
            parsed_pages = pages
            if pages > DOC_MAX_PAGES:
                only = list(range(DOC_MAX_PAGES))
                parsed_pages = DOC_MAX_PAGES
                warnings.append(f"文档共 {pages} 页，只解析了前 {DOC_MAX_PAGES} 页。")
            # use_ocr=False rather than relying on tesseract being absent: it
            # short-circuits select_ocr_function() entirely, so a machine that does
            # have tesseract installed still gets the same no-OCR behaviour.
            text = pymupdf4llm.to_markdown(doc, pages=only, use_ocr=False)
        except DocumentError:
            raise
        except Exception as exc:
            log.warning("pdf parse failed for %s: %s: %s", name, type(exc).__name__, exc)
            raise DocumentError(f"PDF 解析失败：{type(exc).__name__}") from exc
        finally:
            doc.close()

    # A scanned PDF has the right page count but no text layer. Under 20 characters
    # per page means there is nothing to read; saying so beats handing the model an
    # empty attachment and letting it invent an answer.
    if len(text.strip()) < 20 * parsed_pages:
        warnings.append(
            f"这个 PDF 的 {parsed_pages} 页里几乎没有可提取的文字层，很可能是扫描件或图片型 PDF。"
            "本应用不做 OCR；如果需要理解其内容，可以改用图片上传（视觉模型支持看图）。"
        )
    return _finish(name, "PDF", text, pages=pages, warnings=tuple(warnings))


# ---------------------------------------------------------------------------
# OOXML shared plumbing
# ---------------------------------------------------------------------------


def _zip_parts(data: bytes, pattern: re.Pattern[str], label: str) -> dict[str, bytes]:
    """Read only the XML parts `pattern` matches, behind zip-bomb and entity gates."""
    if not data.startswith(b"PK\x03\x04"):
        raise DocumentError(f"这不是一个有效的 {label} 文件（缺少 zip 文件头），可能已损坏或改了扩展名。")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise DocumentError(f"{label} 文件已损坏，无法解压：{exc}") from exc

    with zf:
        infos = zf.infolist()
        # Checked against the declared sizes before a single byte is inflated, so a
        # bomb is rejected for what it claims rather than for what it costs us.
        if len(infos) > ZIP_MAX_ENTRIES:
            raise DocumentError(f"{label} 内含 {len(infos)} 个条目，超过 {ZIP_MAX_ENTRIES} 上限，已拒绝。")
        oversized = [i.filename for i in infos if i.file_size > ZIP_MAX_ENTRY_BYTES]
        if oversized:
            raise DocumentError(f"{label} 内含异常巨大的条目（{oversized[0][:60]}），已拒绝。")
        total = sum(i.file_size for i in infos)
        if total > ZIP_MAX_TOTAL_BYTES:
            raise DocumentError(
                f"{label} 解压后合计约 {total // 1_000_000}MB，超过 {ZIP_MAX_TOTAL_BYTES // 1_000_000}MB 上限，已拒绝。"
            )

        out: dict[str, bytes] = {}
        for info in infos:
            if not pattern.match(info.filename):
                continue
            raw = zf.read(info)
            # ElementTree has no billion-laughs protection, and a genuine OOXML part
            # never carries a DOCTYPE, so sniffing the first kilobytes is a complete
            # answer here and costs no dependency.
            head = raw[:1024]
            if b"<!DOCTYPE" in head or b"<!ENTITY" in head:
                raise DocumentError(
                    f"{label} 的 {info.filename[:60]} 含 XML 实体声明（DOCTYPE/ENTITY），出于安全考虑已拒绝解析。"
                )
            out[info.filename] = raw
    return out


def _xml(raw: bytes, label: str) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        raise DocumentError(f"{label} 内部 XML 结构损坏：{exc}") from exc


# ---------------------------------------------------------------------------
# docx
# ---------------------------------------------------------------------------


def _para_text(p: ElementTree.Element) -> str:
    # Walking every descendant rather than only direct w:t children is what picks
    # up text boxes (w:txbxContent) and other inline containers — strictly more
    # than python-docx's document.paragraphs would have given us.
    out: list[str] = []
    for node in p.iter():
        tag = node.tag
        if tag == f"{_W}t":
            out.append(node.text or "")
        elif tag == f"{_W}tab":
            out.append("\t")
        elif tag in (f"{_W}br", f"{_W}cr"):
            out.append("\n")
    return "".join(out)


def _walk_blocks(parent: ElementTree.Element):
    """Yield the w:p and w:tbl elements of a body, in document order."""
    for child in parent:
        if child.tag in (f"{_W}p", f"{_W}tbl"):
            yield child
        elif child.tag == f"{_W}sdt":
            # Content controls wrap blocks without being one themselves.
            content = child.find(f"{_W}sdtContent")
            if content is not None:
                yield from _walk_blocks(content)


def _table_lines(tbl: ElementTree.Element) -> list[str]:
    lines = []
    for tr in tbl.findall(f"{_W}tr"):
        cells = [_para_text(tc).replace("\n", " ").replace("|", "/").strip() for tc in tr.findall(f"{_W}tc")]
        if any(cells):
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def _parse_docx(name: str, data: bytes) -> ParsedDoc:
    parts = _zip_parts(data, _DOCX_PARTS, "Word 文档")
    if "word/document.xml" not in parts:
        raise DocumentError("这不是一个有效的 .docx 文件：缺少 word/document.xml。")

    root = _xml(parts["word/document.xml"], "Word 文档")
    body = root.find(f"{_W}body")
    out: list[str] = []
    for block in _walk_blocks(body if body is not None else root):
        if block.tag == f"{_W}tbl":
            out.extend(_table_lines(block))
        else:
            line = _para_text(block).strip()
            if line:
                out.append(line)

    for part, label, item_tag in (
        ("word/footnotes.xml", "脚注", f"{_W}footnote"),
        ("word/endnotes.xml", "尾注", f"{_W}endnote"),
    ):
        if part not in parts:
            continue
        items = []
        for item in _xml(parts[part], label):
            if item.tag != item_tag:
                continue
            # ids -1 and 0 are the separator and continuation marks, not notes.
            if item.get(f"{_W}id", "") in ("-1", "0"):
                continue
            text = " ".join(t for t in (_para_text(p).strip() for p in item.iter(f"{_W}p")) if t)
            if text:
                items.append(f"[{item.get(f'{_W}id', '')}] {text}")
        if items:
            out.append("")
            out.append(f"## {label}")
            out.extend(items)

    return _finish(name, "Word", "\n".join(out))


# ---------------------------------------------------------------------------
# pptx
# ---------------------------------------------------------------------------


def _slide_text(raw: bytes, label: str) -> str:
    root = _xml(raw, label)
    lines = []
    # a:p inside a:tbl is included by iter(), so table text is captured too,
    # flattened rather than gridded.
    for p in root.iter(f"{_A}p"):
        text = "".join(t.text or "" for t in p.iter(f"{_A}t")).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _parse_pptx(name: str, data: bytes) -> ParsedDoc:
    parts = _zip_parts(data, _PPTX_PARTS, "PowerPoint 演示文稿")
    slides = sorted(
        (n for n in parts if n.startswith("ppt/slides/")),
        # Numeric, not the zip's lexicographic order: otherwise slide10 comes
        # before slide2 and the deck reads out of order.
        key=lambda n: int(_SLIDE_NO.search(n).group(1)),
    )
    if not slides:
        raise DocumentError("这不是一个有效的 .pptx 文件：没有找到任何幻灯片。")

    out: list[str] = []
    for i, slide in enumerate(slides, 1):
        out.append(f"## 第 {i} 页")
        text = _slide_text(parts[slide], "幻灯片")
        if text:
            out.append(text)
        notes = parts.get(f"ppt/notesSlides/notesSlide{i}.xml")
        if notes:
            note = _slide_text(notes, "备注").strip()
            # The notes master contributes a bare slide-number placeholder.
            if note and not note.isdigit():
                out.append(f"> 备注：{note}")
    return _finish(name, "PowerPoint", "\n".join(out), pages=len(slides))


# ---------------------------------------------------------------------------
# xlsx
# ---------------------------------------------------------------------------


def _cell_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).replace("\n", " ").replace("|", "/").strip()


def _parse_xlsx(name: str, data: bytes) -> ParsedDoc:
    from openpyxl import load_workbook

    try:
        # data_only=True so a formula cell yields its cached result instead of
        # "=SUM(A1:A9)", which is what a reader actually wants. read_only=True
        # streams rows instead of building the whole workbook in memory.
        wv = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as exc:
        raise DocumentError(f"无法打开这个 Excel 文件（可能已损坏）：{type(exc).__name__}") from exc

    # data_only can only report a value Excel already computed and stored. A
    # workbook written by a script has no such cache, and every formula then comes
    # back as an empty cell with nothing to say why — the model would read the hole
    # as real data. A second handle on the formulas, walked in lockstep, detects
    # that; same file and same iteration order, so no coordinate bookkeeping.
    try:
        wf = load_workbook(io.BytesIO(data), read_only=True, data_only=False)
    except Exception:  # noqa: BLE001 - detection is a nicety, not a requirement
        wf = None

    out: list[str] = []
    warnings: list[str] = []
    truncated = False
    stale_formulas = 0
    try:
        sheets = wv.worksheets
        fsheets = wf.worksheets if wf is not None else []
        for idx, ws in enumerate(sheets):
            fws = fsheets[idx] if idx < len(fsheets) else None
            out.append(f"## 工作表：{ws.title}")
            rows = 0
            frow_iter = fws.iter_rows(values_only=True) if fws is not None else None
            for row in ws.iter_rows(values_only=True):
                if rows >= XLSX_MAX_ROWS:
                    truncated = True
                    warnings.append(f"工作表「{ws.title}」超过 {XLSX_MAX_ROWS} 行，其余已省略。")
                    break
                frow = next(frow_iter, ()) if frow_iter is not None else ()
                rows += 1
                if len(row) > XLSX_MAX_COLS:
                    truncated = True
                    row = row[:XLSX_MAX_COLS]
                cells = [_cell_str(v) for v in row]
                for j, cell in enumerate(cells):
                    f = frow[j] if j < len(frow) else None
                    if cell == "" and isinstance(f, str) and f.startswith("="):
                        stale_formulas += 1
                if any(cells):
                    out.append("| " + " | ".join(cells) + " |")
            if not rows:
                out.append("（空表）")
    finally:
        wv.close()
        if wf is not None:
            wf.close()
    if stale_formulas:
        warnings.append(
            f"有 {stale_formulas} 个公式单元格里没有存放计算结果，正文中显示为空——"
            "这个文件很可能由脚本生成而非在 Excel/WPS 里保存过。"
            "用 Excel 打开并保存一次后再上传，即可带上计算结果。"
        )
    if truncated and not any("列" in w for w in warnings):
        warnings.append(f"超过 {XLSX_MAX_COLS} 列的部分已省略。")
    return _finish(name, "Excel", "\n".join(out), warnings=tuple(warnings), truncated=truncated)


# ---------------------------------------------------------------------------
# Context budget
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Rough token count, deliberately biased high.

    Overflowing the window fails the whole request; under-filling it only wastes a
    little context. So every rate here is rounded up rather than fitted.

    Calibrated against llama-server's POST /tokenize on five samples, all with the
    gemma tokenizer at n_ctx 131072. Estimate / exact: Chinese prose 1.57, English
    prose 1.90, Markdown table 1.36, Python code 1.56, mixed PDF-style text 1.41.
    The tightest is the punctuation-heavy Markdown table at 1.36x. A different
    model's tokenizer will not match these ratios exactly, so that margin and
    CTX_SAFETY_TOKENS together are what cover the swap.

    CJK is subtracted before the dense pass: full-width Chinese punctuation sits
    inside the CJK ranges, so counting it in both classes would double-charge.
    """
    if not text:
        return 0
    without_cjk = _CJK.sub("", text)
    cjk = len(text) - len(without_cjk)
    dense = len(_DENSE.findall(without_cjk))
    plain = len(without_cjk) - dense
    return int(cjk * CJK_TOKEN_WEIGHT + dense * DENSE_TOKEN_WEIGHT + plain / PLAIN_CHARS_PER_TOKEN) + 1


def _doc_tokens(name: str, text: str) -> int:
    return estimate_tokens(text) + estimate_tokens(name) + DOC_WRAPPER_TOKENS


def _msg_tokens(msg) -> int:
    total = estimate_tokens(getattr(msg, "content", "") or "")
    for name, text in getattr(msg, "documents", None) or []:
        total += _doc_tokens(name, text)
    return total


def _total(msgs) -> int:
    return sum(_msg_tokens(m) for m in msgs)


def _stub(name: str) -> str:
    return f"【附件「{name}」的正文已省略，以控制上下文长度】"


def fit_budget(messages, n_ctx: int, reserve: int, safety: int = CTX_SAFETY_TOKENS):
    """Trim `messages` until they fit, degrading in three stages.

    `reserve` is the completion budget (max_tokens, or thinking_max_tokens while
    thinking); `safety` covers the system prompt, the tool schemas and the chat
    template, none of which appear in `messages`.

    Returns `(messages, notes)` where notes are user-facing Chinese lines to emit
    as SSE status events — trimming must never be silent. On hard failure returns
    `([], [reason])`: an empty list means even the last message cannot fit.

    Messages are rebuilt with dataclasses.replace, never mutated in place, so the
    caller's history (and the archive it feeds) keeps the full document text.
    """
    budget = n_ctx - reserve - safety
    if budget <= 0:
        return [], [
            f"上下文预算不足：模型上下文 {n_ctx} tokens，扣除本次回复保留的 {reserve} "
            f"与安全边际 {safety} 后没有剩余。请在设置里调小 max_tokens，或换用上下文更大的模型。"
        ]

    msgs = list(messages)
    notes: list[str] = []
    if _total(msgs) <= budget:
        return msgs, notes

    # Stage A: blank out older turns' document bodies, oldest first. The newest
    # turn is never touched here — silently discarding what the user just uploaded
    # is the worst possible outcome.
    doc_indexes = [i for i, m in enumerate(msgs) if getattr(m, "documents", None)]
    newest = doc_indexes[-1] if doc_indexes else None
    blanked: list[str] = []
    for i in doc_indexes:
        if i == newest:
            continue
        if _total(msgs) <= budget:
            break
        msg = msgs[i]
        blanked.extend(name for name, _ in msg.documents)
        msgs[i] = replace(msg, documents=[(n, _stub(n)) for n, _ in msg.documents])
    if blanked:
        shown = "、".join(blanked[:3]) + ("…" if len(blanked) > 3 else "")
        notes.append(
            f"为控制上下文长度，已省略较早上传的 {len(blanked)} 份附件正文（{shown}）；"
            "文件名仍保留在对话中，如需重新分析请再上传一次。"
        )
    if _total(msgs) <= budget:
        return msgs, notes

    # Stage B: truncate the newest turn's attachments to whatever is left.
    if newest is not None:
        msg = msgs[newest]
        remaining = budget - (_total(msgs) - sum(_doc_tokens(n, t) for n, t in msg.documents))
        kept: list[tuple[str, str]] = []
        cut_names: list[str] = []
        for doc_name, text in msg.documents:
            remaining -= estimate_tokens(doc_name) + DOC_WRAPPER_TOKENS
            if remaining <= 0:
                kept.append((doc_name, _stub(doc_name)))
                cut_names.append(doc_name)
                continue
            if estimate_tokens(text) <= remaining:
                kept.append((doc_name, text))
                remaining -= estimate_tokens(text)
                continue
            # One char per token: the estimator already charges ~1 token per CJK
            # char, so this cut is conservative for Chinese and very conservative
            # for Latin text. Under-filling is the safe direction.
            head = text[:remaining]
            notice = f"\n【正文已截断，仅保留前 {len(head)} 字】"
            # The notice costs tokens too. Sizing the cut to `remaining` and then
            # appending it lands just over budget, the total check below fails, and
            # Stage C throws away a whole turn that a slightly shorter cut would
            # have kept. Shrinking strictly guarantees this terminates.
            while head and estimate_tokens(head) + estimate_tokens(notice) > remaining:
                head = head[: len(head) - max(1, len(head) // 10)]
                notice = f"\n【正文已截断，仅保留前 {len(head)} 字】"
            if not head:
                kept.append((doc_name, _stub(doc_name)))
            else:
                kept.append((doc_name, f"{head}{notice}"))
            cut_names.append(doc_name)
            remaining = 0
        msgs[newest] = replace(msg, documents=kept)
        if cut_names:
            notes.append(f"附件 {'、'.join(cut_names[:3])} 的正文超出剩余上下文，已截断。")
    if _total(msgs) <= budget:
        return msgs, notes

    # Stage C: the plain text alone no longer fits. Drop whole messages from the
    # oldest end, but never the last one — that is the question being asked.
    dropped = 0
    while len(msgs) > 1 and _total(msgs) > budget:
        msgs.pop(0)
        dropped += 1
    if dropped:
        notes.append(f"对话过长，已丢弃最早的 {dropped} 条消息以控制上下文长度。")
    if _total(msgs) > budget:
        return [], [
            f"仅最后一条消息就约需 {_total(msgs)} tokens，超出可用预算 {budget}"
            f"（模型上下文 {n_ctx}）。请新建对话，或上传更小的文件。"
        ]
    return msgs, notes
