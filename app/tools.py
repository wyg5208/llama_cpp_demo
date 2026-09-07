"""Tool assembly, tool policy, and the runner for everything that is not web search.

A new module rather than an addition to search.py, because search.py is the home of
one feature and its TOOLS constant is imported directly by tests/test_documents.py.
Assembly is cross-cutting and needs a neutral place.

Three jobs, each answering a problem measured rather than imagined:

1. `build_tools()` is the ONE place a tool set is assembled. main.py calls it once
   per request and hands the same list to both `payload["tools"]` and
   `budget_safety()`. Until now llm.py sent search.TOOLS while main.py charged for
   `json.dumps` of that same constant — two spellings of one fact, kept in step by
   hand. Once the set is dynamic that drift is certain, and it fails dangerous:
   charging for 2 tools while sending 11 lets fit_budget admit history that will not
   fit.

2. The deny list. With this project's directory as an allowed root,
   `read_text_file(".env")` was measured to succeed and return the search API keys.
   The MCP server only knows "inside an allowed root or not"; it has no concept of a
   secret. So the check lives here, in our dispatch layer, and it is this app's list
   — not a protocol guarantee, and not portable to another MCP server.

3. `ToolRunner` meters what comes back. Tool results are appended inside llm.py's
   loop and never pass through fit_budget, which trims only the incoming history.
   Bounded here instead: MAX_TOOL_RESULT_CHARS per result, MAX_TOOL_TOTAL_CHARS per
   request. `directory_tree` over this project's root measured 1,302,893 characters
   — 6.2x a 40960 window on its own — so "unbounded" is not a hypothetical.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterable, Sequence

from .memory_store import MemoryStoreError, add_memory, search_memories
from .search import TOOLS as SEARCH_TOOLS

if TYPE_CHECKING:  # both sit above this module, so the edges are types only
    from .config import Settings
    from .mcp import MCPHost

log = logging.getLogger(__name__)


# --- budgets ------------------------------------------------------------------

# One result. 8,000 characters of path-like text measured 4,950 tokens with this
# repo's own estimate_tokens — 12.1% of a 40960 window, 30.2% of a 16384 one.
MAX_TOOL_RESULT_CHARS = 8_000

# Every result of one request, all rounds together. max_tool_rounds is 4 and a round
# may carry several calls, so without this a single question could pull in 32 clipped
# results (~150k tokens) and none of it would ever reach fit_budget.
MAX_TOOL_TOTAL_CHARS = 24_000

# think spends a tool round per step (llm.py's `for _ in range(max_tool_rounds + 1)`),
# which is deliberate: free thinking is a loop risk. This is the backstop for a model
# that would otherwise think until the rounds run out and never answer.
MAX_THOUGHTS = 16

# recall returns this many notes. Small on purpose — the store holds up to 500 and
# dumping them is the failure memory_store.search_memories exists to avoid.
RECALL_LIMIT = 5

# One save_document call. MAX_TOKENS already bounds what a model can generate, so this
# only fires if the user raises MAX_TOKENS a long way — but an artifact crosses two
# boundaries that need a definite ceiling regardless of .env: it is relayed whole in a
# single SSE event, and it is stored whole in runtime/history/*.json. MAX_ARTIFACT_NAME
# matches app.js's TITLE_CHARS so the two filename sanitizers cannot disagree on length.
MAX_ARTIFACT_CHARS = 40_000
MAX_ARTIFACT_NAME = 40


# --- filesystem tool policy ---------------------------------------------------

# Upstream's 14 tools minus these 4. Each is a trade-off of this app, not a protocol
# statement, which is why this is a name list while the read/write split is not:
#   read_file                  DEPRECATED alias of read_text_file, upstream says so
#   read_media_file            returns base64 — a token bomb in a text prompt, and
#                              this app already has an image upload path
#   list_directory_with_sizes  list_directory + get_file_info covers it
#   list_allowed_directories   the allowed roots go into the system prompt, so 226
#                              tokens to let the model ask is wasted
FS_DROP = frozenset(
    {"read_file", "read_media_file", "list_directory_with_sizes", "list_allowed_directories"}
)

# Directory segments that are runtime data, generated, or third-party. Beyond the
# privacy argument (runtime/history holds every past conversation, and the settings
# override file lives there too), this is what keeps directory_tree from walking
# runtime/llama-vulkan and .venv — the two reasons one tree over this project
# measured 33,762 lines.
SENSITIVE_DIRS = frozenset(
    {"runtime", ".venv", "venv", ".git", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache"}
)

# Glob patterns for the final name. Lowercase; the comparison is case-folded because
# Windows is case-insensitive and the MCP server was measured to accept .ENV exactly
# as it accepts .env.
SENSITIVE_GLOBS = (
    "*.key",
    "*.pem",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    "credentials*",
    "secrets*",
)

# .env is handled by prefix rather than by glob so that .env.local and
# .env.production are covered too — every spelling of this file holds the same keys.
ENV_ALLOW = frozenset({".env.example"})

# Arguments that carry document text rather than a location. Skipped by _paths_in so
# that editing a file which happens to mention an absolute path is not mistaken for
# reaching for that path. Enumerated from the ten tools this app exposes rather than
# guessed: write_file.content, edit_file.edits[].oldText/newText, search_files.pattern,
# recall.query, think.thought, remember.text.
_CONTENT_KEYS = frozenset({"content", "text", "oldtext", "newtext", "pattern", "query"})

# Arguments that are neither a location nor document text: options the server never
# resolves to a file. Also enumerated, not guessed — directory_tree and search_files
# take excludePatterns (["node_modules"] is a legitimate call whose pattern happens to
# name a sensitive directory), list_directory_with_sizes takes sortBy, edit_file takes
# dryRun and edits, read_text_file takes head and tail.
#
# `edits` is the one container: a list of {oldText, newText} objects whose inner keys
# are already in _CONTENT_KEYS. head/tail/dryRun are numbers and booleans in the
# schema so they are never strings at all. Both kinds are listed so that the three
# sets together are exhaustive against the live tools/list — see
# tests/test_mcp.py::test_every_live_parameter_is_classified, which fails if upstream
# adds a parameter nobody has put in a bucket yet.
#
# These two sets are the ONLY exemptions. Anything else is treated as a location.
_OPTION_KEYS = frozenset({"excludepatterns", "sortby", "dryrun", "edits", "head", "tail"})

# The location arguments of the ten tools this app exposes: read_text_file /
# get_file_info / list_directory / directory_tree / write_file / create_directory /
# edit_file / search_files all take `path`, read_multiple_files takes `paths`,
# move_file takes `source` and `destination`.
#
# No longer load-bearing in _paths_in — the fail-closed default above covers these and
# any parameter upstream adds next release. It stays because it is the enumeration the
# tests check the two exemption sets against: if a real location parameter ever landed
# in _CONTENT_KEYS or _OPTION_KEYS, the deny list would silently stop covering it.
_LOCATION_KEYS = frozenset({"path", "paths", "source", "destination"})

# A drive letter, a UNC prefix, or a leading slash. Used to decide whether a location
# is absolute, which is what is_denied refuses to guess about: a relative argument is
# rejected rather than judged, because where it lands depends on a base the model
# cannot see.
#
# That base is NOT the child's cwd, whatever an earlier comment here claimed. Read out
# of the installed server (dist/lib.js, resolveRelativePathAgainstAllowedDirectories):
# a relative argument is resolved against each allowed root in turn, and
# process.cwd() is consulted only when the server has no roots at all — which never
# happens here, since Settings validates MCP_FS_ROOTS as existing directories. So
# read_text_file(".env") reaches the project's real .env, and refusing relatives is the
# only thing standing between the model and the search API keys in it.
_ABS_RE = re.compile(r"^(?:[a-zA-Z]:[\\/]|[\\/]{2}|/)")


def _norm(path: Any) -> str:
    """One canonical spelling: forward slashes, `..` collapsed, case folded.

    Lexical rather than Path.resolve() because a write_file destination does not
    exist yet, and normpath is what makes `app/../.env` compare equal to `.env`
    without touching the disk.
    """
    return os.path.normpath(str(path).replace("\\", "/")).replace("\\", "/").casefold()


def _segments_below(path: Any, roots: Sequence[Path]) -> list[str] | None:
    """The segments of `path` below the allowed root it is inside, else None.

    Root-relative when a root matches, so a root that itself lives under a directory
    called `node_modules` does not deny its whole tree.
    """
    target = _norm(path)
    for root in roots:
        base = _norm(root).rstrip("/")
        if not base:
            continue
        if target == base:
            return []
        prefix = base + "/"
        if target.startswith(prefix):
            return [s for s in target[len(prefix) :].split("/") if s]
    return None


def is_denied(path: Any, roots: Sequence[Path]) -> str | None:
    """A Chinese reason to refuse this path, or None to let it through.

    Absolute paths only: a relative one is refused rather than judged. Where it points
    is not a mystery — the installed server resolves it against each allowed root in
    turn (dist/lib.js, resolveRelativePathAgainstAllowedDirectories) — and that is
    precisely the problem: the default root is the project directory, so ".env",
    "./.env" and "app/../.env" all resolve onto the real .env and come back with the
    search API keys in it. Measured, through the real runner, before this rule existed.
    Refusing also removes any reason to reimplement the server's resolution here and
    keep the two in step: the system prompt asks for absolute paths, so the answer
    hands that instruction back to the model.

    Every segment is checked, not just the last, so `docs/secrets/keys.txt` is caught
    as well as `id_rsa`. Inspecting all of them also means `..` cannot hide a
    sensitive segment — it can only add extra ones, which over-denies rather than
    under-denies. That holds because the absoluteness rule above runs first; without
    it the fallback below would drop `..` and `../history/index.json` would walk
    straight past.

    What this does NOT do: filter the *output* of a tool. directory_tree over an
    allowed parent will still list a file called `.env`, because listing is what it
    does. Only the name leaks; the contents need read_text_file pointed at that name,
    and that call is refused here.
    """
    if not str(path or "").strip():
        return "路径为空，无法校验，已拒绝。"

    if not _ABS_RE.match(str(path)):
        return (
            "路径必须写成绝对路径（例如 D:\\dir\\notes.md）。"
            "相对路径的落点无法可靠校验，已拒绝；请改用绝对路径重试。"
        )

    segments = _segments_below(path, roots)
    if segments is None:
        # Outside every allowed root — the server will reject it anyway, but a path we
        # cannot place is a path we cannot vouch for, so the list still applies.
        segments = [s for s in _norm(path).split("/") if s and s != ".."]

    for segment in segments:
        if segment in SENSITIVE_DIRS:
            return (
                f"路径里的 {segment}/ 是运行时数据或第三方生成物"
                "（会话归档、日志、模型二进制、依赖树），不在可访问范围内。"
                "如需项目文件，请改用 list_directory 逐层查看源代码目录。"
            )
        if segment.startswith(".env") and segment not in ENV_ALLOW:
            return (
                ".env 里存着检索后端的 API 密钥，不允许读取或改写。"
                "需要看配置项的说明请读 .env.example。"
            )
        for pattern in SENSITIVE_GLOBS:
            if fnmatch.fnmatch(segment, pattern):
                return f"{segment} 看起来是密钥或凭据文件，不允许访问。"
    return None


def clip(text: str) -> str:
    """Trim one tool result to MAX_TOOL_RESULT_CHARS, and say where to go instead.

    The note points onward rather than just apologising: a truncated 33,762-line tree
    is a fragment, and the useful response to that is a narrower question.
    """
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    return text[:MAX_TOOL_RESULT_CHARS] + (
        f"\n\n【已截断：结果共 {len(text)} 字符，只保留前 {MAX_TOOL_RESULT_CHARS} 字符。"
        "请缩小范围再试：目录改用 list_directory 一层一层看，"
        "文件改用 read_text_file 的 head / tail 参数分段读。】"
    )


def _paths_in(args: dict) -> list[str]:
    """Every argument that could be a filesystem path, collected fail-closed.

    A string is a candidate path unless its key is exempt, and the exemptions are the
    only enumerated thing here: _CONTENT_KEYS (document text — writing a file whose
    body mentions a path is not reaching for that path) and _OPTION_KEYS (arguments
    the server never resolves to a file, such as excludePatterns).

    Fail-closed rather than fail-open because the alternative was measured to leak.
    Collecting only the keys in _LOCATION_KEYS, plus anything that looked absolute,
    left a hole exactly one upstream rename wide: a location parameter nobody had
    enumerated yet, passed a relative value, was invisible to both rules. And a
    relative value is not harmless — read out of the installed server's dist/lib.js,
    resolveRelativePathAgainstAllowedDirectories resolves one against each allowed
    root in turn, so read_text_file(".env") reaches the project's real .env and comes
    back with the search API keys in it. An unrecognised key now lands in the deny
    list's hands instead of past it.

    Checking every string was the earlier objection to this shape, on the grounds that
    `directory_tree(excludePatterns=["node_modules"])` is a legitimate call whose
    pattern happens to name a sensitive directory. That is what _OPTION_KEYS is for:
    the objection was right about the argument and wrong about the conclusion.
    """
    found: list[str] = []

    def walk(value: Any, key: str = "") -> None:
        lowered = key.casefold()
        if lowered in _CONTENT_KEYS or lowered in _OPTION_KEYS:
            return
        if isinstance(value, str):
            found.append(value)
        elif isinstance(value, dict):
            for name, item in value.items():
                walk(item, str(name))
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item, key)

    walk(args or {})
    return found


# --- native tool schemas ------------------------------------------------------
# Same shape as search.WEB_SEARCH_TOOL, which is what llama-server expects.

REMEMBER = "remember"
RECALL = "recall"
THINK = "think"

REMEMBER_TOOL = {
    "type": "function",
    "function": {
        "name": REMEMBER,
        "description": (
            "把一条信息长期记住，存进本机的记忆文件，以后的对话还能查到。\n"
            "只在这些情况下调用：用户明确说「记住」「以后都这样」，用户纠正了你的做法，"
            "或者用户说出了自己的偏好、身份、项目背景。\n"
            "一次只记一件事，用一句中文写清楚，不要抄整段对话，"
            "也不要记这次回答用完就没用的临时信息。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "要记住的那一句话，中文，不超过 500 字。",
                },
                "kind": {
                    "type": "string",
                    "enum": ["fact", "preference", "todo"],
                    "description": (
                        "fact=客观事实或项目背景，preference=用户偏好，todo=待办事项。"
                        "不确定就用 fact。"
                    ),
                },
            },
            "required": ["text"],
        },
    },
}

RECALL_TOOL = {
    "type": "function",
    "function": {
        "name": RECALL,
        "description": (
            "按关键词查回以前记住的信息。问题涉及用户偏好、之前的约定、项目背景，"
            "或者用户说「我上次」「我记得」时，先查再答。\n"
            "这是关键词匹配，不是语义检索：query 用 1~3 个核心词（例如「导出 格式」）。"
            "查不到就换一种说法再查一次，不要用同一串词反复查。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "1~3 个关键词，中文或英文都可以。",
                }
            },
            "required": ["query"],
        },
    },
}

THINK_TOOL = {
    "type": "function",
    "function": {
        "name": THINK,
        "description": (
            "把一步中间推理写进「思考过程」，用于需要多步规划、比较、拆解或自查的问题。\n"
            "一次只写一步，写完再决定下一步；想清楚了就直接作答，不要用 think 复述答案。\n"
            f"闲聊、翻译、简单问答不要用 think。每次调用会占用一次工具轮次，最多 {MAX_THOUGHTS} 步。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "这一步的想法，中文，一两句话。",
                }
            },
            "required": ["thought"],
        },
    },
}

MEMORY_TOOLS = [REMEMBER_TOOL, RECALL_TOOL]


SAVE_DOCUMENT = "save_document"

# One semantic for all five: `content` is always Markdown source and `format` only
# decides which file the browser downloads. app.js already feeds Markdown to every one
# of the five paths (saveBlob for md, exportHtmlDocument for html, toCsv — which parses
# Markdown tables — for csv, and blocksFromMarkdown -> /api/export for pdf and docx), so
# this needs no new renderer and, more importantly, no server-side one: export.py refuses
# md/html/csv on purpose and that refusal stays intact. A bonus the model does not need
# to know about: an archived artifact is Markdown, so it can be re-downloaded after a
# refresh in a format other than the one that was asked for.
ARTIFACT_FORMATS = ("md", "html", "csv", "pdf", "docx")

SAVE_DOCUMENT_TOOL = {
    "type": "function",
    "function": {
        "name": SAVE_DOCUMENT,
        # Behavioural guidance ("do not repeat the document in your reply") lives in
        # PROMPT_LINES["doc_gen"], not here: this description measured 591 tokens with it
        # and 504 without, and the prompt line carries it on every request anyway.
        "description": (
            "把一整篇文档交给用户下载。用户要文件时（导出、生成报告、做成表格）用它。\n"
            "content 一律是完整的 Markdown 源文本，五种格式都是；format 只决定下载成哪种文件。"
            "csv 需要文档里有 Markdown 表格才有内容可转。\n"
            "一次调用只生成一份，超长会被整份拒绝。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "只写文件名，不要带路径，例如「周报.md」。",
                },
                "format": {
                    "type": "string",
                    "enum": list(ARTIFACT_FORMATS),
                    "description": "下载成哪种文件。内容都是 Markdown，这里只选容器。",
                },
                "title": {
                    "type": "string",
                    "description": "文档标题，用作 pdf 元数据与 html 的 title。可省略。",
                },
                "content": {
                    "type": "string",
                    "description": "完整的 Markdown 源文本，一份到底。",
                },
            },
            "required": ["filename", "format", "content"],
        },
    },
}

# Mirrors app.js's FS_UNSAFE / FS_RESERVED. Two copies of one rule is a drift risk, and
# it is accepted for the same reason EXPORT_CSS's three copies are: sharing them would
# need a build step this app does not have. The browser runs its own sanitiser on the way
# to a.download as well, so neither copy is the only thing standing.
_FS_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_FS_RESERVED = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])$", re.I)


def safe_artifact_name(name: Any, fmt: str) -> str:
    """A model-supplied filename made safe to hand to a browser's download.

    Sanitised here and not only in the browser because /api/chat is reachable without
    one — the same reason StoredSource._check_url holds a URL to its http(s) rule on the
    server — and because the name is archived into runtime/history/*.json, which should
    not store something that needs sanitising every time it is read back.

    The rules are measured, not theoretical: the real session index holds a title
    starting '> **【角色设定】**：…', and both > and * are illegal in a Windows filename.
    """
    extension = fmt if fmt in ARTIFACT_FORMATS else "md"
    s = _FS_UNSAFE.sub("_", str(name or ""))
    # Killing the separators is what kills traversal: "../../etc/passwd" arrives here and
    # leaves as ".._.._etc_passwd", which is an odd filename and nothing more.
    s = s.replace("..", "_")
    # Windows silently strips leading and trailing dots and spaces, so "…" is not the
    # name that lands on disk.
    s = s.strip(". \t\r\n")
    stem, dot, _ = s.rpartition(".")
    if dot:
        # Drop the existing suffix whether or not it matches, because exactly one is
        # appended below. Matching also normalises its case ("周报.PDF" -> "周报.pdf");
        # mismatching matters more — the bytes are decided by `fmt`, and a file called
        # report.txt that is really a PDF fails confusingly. Measured: without this the
        # matching case came back as "周报.PDF.pdf".
        s = stem or s
    s = s[:MAX_ARTIFACT_NAME].strip(". \t\r\n")
    if not s:
        s = datetime.now(timezone.utc).astimezone().strftime("文档-%Y-%m-%d-%H%M")
    # A device name is reserved regardless of extension, so "con.md" is still bad:
    # Windows resolves a path component to a device by the part before the first dot.
    # Hence the stem, not the whole string, against an anchored pattern.
    if _FS_RESERVED.match(s.split(".")[0]):
        s = f"_{s}"
    return f"{s}.{extension}"


# --- MCP schema conversion ----------------------------------------------------


def mcp_to_openai(tool: dict) -> dict:
    """One MCP tools/list entry -> the OpenAI function shape llama-server wants.

    title, outputSchema, execution and annotations are dropped: llama-server reads
    only name/description/parameters, so anything else is tokens paid for nothing.
    annotations is not lost — pick_fs_tools has already tiered on it.
    """
    parameters = tool.get("inputSchema")
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": str(tool.get("name") or ""),
            "description": str(tool.get("description") or ""),
            "parameters": parameters,
        },
    }


def pick_fs_tools(raw: list[dict], allow_write: bool) -> list[dict]:
    """The filesystem tools to expose, tiered by annotations.readOnlyHint.

    Tiered on the annotation rather than on a hardcoded list of names so that an
    upstream release behaves predictably: a new read-only tool joins the read layer,
    and a new writing tool stays behind the write checkbox instead of appearing in
    both or in neither.

    A tool with no annotations at all is treated as writing. That is fail-closed —
    with the write box unticked, an unknown tool is withheld rather than waved
    through.
    """
    picked: list[dict] = []
    for tool in raw:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name") or "")
        if not name or name in FS_DROP:
            continue
        annotations = tool.get("annotations")
        annotations = annotations if isinstance(annotations, dict) else {}
        if not allow_write and annotations.get("readOnlyHint") is not True:
            continue
        picked.append(mcp_to_openai(tool))
    return picked


def fs_tool_names(raw: list[dict], allow_write: bool) -> set[str]:
    """Names of the tools pick_fs_tools would expose — no second source of truth."""
    return {str(t["function"]["name"]) for t in pick_fs_tools(raw, allow_write)}


def build_tools(
    *,
    search: bool = False,
    memory: bool = False,
    think: bool = False,
    doc_gen: bool = False,
    fs_read: bool = False,
    fs_write: bool = False,
    fs_raw: list[dict] | None = None,
) -> list[dict]:
    """Assemble the tool list for one request. The single place this happens.

    fs_write without fs_read yields nothing: reading is the capability, writing is an
    addition to it, and the checkbox in the UI is disabled unless read is ticked.
    main.py clamps the same way, so the two agree by construction rather than by luck.

    doc_gen is ordered before the filesystem layer because the filesystem layer is the
    one whose size varies with what the MCP server reports, and a varying tail keeps the
    prefix of this list stable — which is what llama-server's prompt-prefix cache wants.

    Cost, measured with this repo's estimate_tokens over the JSON of the result:
    search alone 807 tokens, save_document 504 on its own, the read-only filesystem
    layer 2,184 more, the writing layer 1,361 more. On a 16384 window — what any GGUF
    missing from N_CTX_OVERRIDES falls back to — everything at once is over 40% of the
    context before a single message. That is why every capability here has its own
    checkbox and none is on by default.
    """
    tools: list[dict] = []
    if search:
        tools.extend(SEARCH_TOOLS)
    if memory:
        tools.extend(MEMORY_TOOLS)
    if think:
        tools.append(THINK_TOOL)
    if doc_gen:
        tools.append(SAVE_DOCUMENT_TOOL)
    if fs_read:
        tools.extend(pick_fs_tools(fs_raw or [], allow_write=fs_write))
    return tools


# --- system prompt ------------------------------------------------------------

# The character budget doc_gen's line quotes to the model, and the one number here that
# needs its derivation written down. MAX_TOKENS=2048 over a realistic Chinese report
# (headings, bullets, a table, a code fence) measures 2,472 characters at 1.045
# chars/token, and passing that text back as a JSON string argument costs a further 13.6%.
# At face value that says ~1,900 net, so 2,400 is only defensible because estimate_tokens
# is calibrated 1.36-1.90x HIGH against the gemma tokenizer (see its docstring): the
# tightest of those ratios, on exactly the table-heavy text a report contains, buys back
# ~2,900. Quoting the pessimistic 1,900 instead would tell the model to leave a quarter of
# the window unused on every document.
# It stays an approximation on purpose. The real figure doubles when 深度思考 is on (that
# path uses MAX_THINKING_TOKENS=4096), a different model's tokenizer will not match those
# ratios, and an exact per-request number would need a second placeholder threaded through
# effective_prompt — precision the model cannot act on. When it does overrun, llm.py's
# _truncation_note says so explicitly instead of leaving a blank reply.
DOC_GEN_CHAR_HINT = 2_400

# Appended per feature rather than folded into DEFAULT_SYSTEM_PROMPT, because a line
# describing a capability the user has not enabled is tokens paid on every single turn
# for nothing.
PROMPT_LINES = {
    "memory": (
        "你可以用 remember 记住用户希望你长期记住的信息（偏好、事实、待办），"
        "用 recall 按关键词查回。只记用户说要记的、或反复强调的偏好，一条记忆一句话；"
        "涉及用户偏好、此前的约定或项目背景时，先 recall 再作答。"
        "记忆是过去记下的，可能已过时，与用户当前的说法冲突时以用户为准。"
    ),
    "think": (
        f"遇到需要多步推理、比较、规划或自查的问题，用 think 工具把中间步骤写下来，"
        f"一次一步，想清楚再作答；think 会占用一次工具轮次，最多 {MAX_THOUGHTS} 步。"
        "闲聊、翻译、简单问答直接回答，不要用 think。"
    ),
    "doc_gen": (
        "用户要文件时（导出、生成报告、做成表格），用 save_document 一次交出整篇："
        "content 写完整的 Markdown 源文本，filename 只写文件名不带路径，一次调用只生成一份。"
        f"内容大约以 {DOC_GEN_CHAR_HINT} 个中文字符为限，长文档请精炼分节，"
        "写到一半被截断的话整份都作废。"
        "调用之后不要在正文里重复整篇内容，一句话说明生成了什么即可。"
    ),
    # The only line with a placeholder, and the only one effective_prompt formats.
    # The allowed roots are runtime configuration and the model cannot guess an
    # absolute Windows path: without them its first call has to fail against the
    # server's "Access denied - path outside allowed directories" to learn where it
    # is, which is precisely the round trip list_allowed_directories was dropped to
    # save. The absolute-path requirement is load-bearing rather than advisory —
    # is_denied refuses a relative one, because where it points depends on a cwd the
    # model cannot see — but a refusal is returned with that explanation attached, so
    # a model that ignores this line costs itself one tool round, not the request.
    "fs_read": (
        "你可以读取这些目录里的文件（路径必须写成绝对路径，相对路径会被拒绝）：{roots}。"
        "list_directory 看目录、read_text_file 读文件、"
        "search_files 按文件名找、get_file_info 看大小和时间。"
        "目录可能很大，优先 list_directory 一层一层看，不要一上来就要整棵目录树。"
        f"单条工具结果超过 {MAX_TOOL_RESULT_CHARS} 字符会被截断，届时请缩小范围重试。"
        "引用文件内容时说明来自哪个文件。"
    ),
    "fs_write": (
        "你被授权在允许目录内写入：create_directory 建目录、write_file 覆盖写、"
        "edit_file 精确替换、move_file 移动或改名（等价于删除原文件）。"
        "动手前先用 read_text_file 确认现状，一次只改必要的部分，"
        "绝不删除或覆盖用户没有提到的文件，也不要改写 .env。"
    ),
}


def effective_prompt(
    base: str, enabled: Iterable[str], fs_roots: Sequence[Path] = ()
) -> str:
    """The system prompt for this request: `base` plus one line per enabled feature.

    Iterates PROMPT_LINES rather than `enabled`, so the result does not depend on the
    order of a set. That matters twice over: main.py charges this exact string to the
    context budget, and llama-server caches the common prompt prefix — a prompt whose
    lines shuffled between requests would miss that cache every time.

    This appends to whatever prompt is in force, including one the user edited in the
    settings panel. That is not the silent rewrite refused elsewhere: the user's own
    text is untouched, the addition is recomputed per request, it only appears while
    the box is ticked, and it is the minimum the feature needs to work at all — a model
    that has not been told remember exists will not call it.

    `fs_roots` names the directories the filesystem tools may touch. Only fs_read's
    line carries a placeholder and only that line is formatted: .format over all of
    them would turn a literal brace in some future line into a KeyError part way
    through a chat request.
    """
    wanted = set(enabled)
    roots = "、".join(str(r) for r in fs_roots) or "（本请求没有配置允许的目录）"
    lines = [(base or "").rstrip()]
    for key, text in PROMPT_LINES.items():
        if key not in wanted:
            continue
        lines.append(text.format(roots=roots) if key == "fs_read" else text)
    return "\n".join(line for line in lines if line)


# --- execution ----------------------------------------------------------------


class ToolRunner:
    """Runs every tool that is not web search, and meters what comes back.

    Search stays in llm.py on purpose: it owns `sources`, `seen_urls` and the
    `sources` event, which are part of the answer rather than part of the tool result.
    Everything else comes here, so llm.py's dispatch grows one branch instead of four.

    One runner per request — `_spent` and `_thoughts` are per-request counters, and a
    runner shared between two chats would let one question's directory tree silence
    another's.

    Every handler is an async generator whose LAST event is
    `{"type": "tool_result", "content": str}`; it may emit `status` and `reasoning`
    events first, which llm.py forwards untouched and the browser already renders.
    """

    def __init__(
        self,
        settings: Settings,
        mcp: MCPHost | None,
        memory_file: Path,
        tools: Sequence[dict],
        fs_tools: Iterable[str] = (),
        fs_write_tools: Iterable[str] = (),
    ) -> None:
        # settings is here for fs_roots: the deny list needs to know what "inside an
        # allowed root" means, and that comes from .env, never from a request body.
        self._roots = list(settings.fs_roots)
        self._mcp = mcp
        self._memory_file = memory_file
        # Exactly what was offered to the model, so a call to a capability the user
        # unticked is answered with "no such tool" instead of being honoured.
        self._names = [str(t.get("function", {}).get("name", "")) for t in tools]
        self._fs_tools = set(fs_tools)
        self._fs_write_tools = set(fs_write_tools)
        self._spent = 0
        self._thoughts = 0

    async def run(self, name: str, args: dict) -> AsyncIterator[dict]:
        """One tool call -> status/reasoning events, then a single tool_result."""
        if self._spent >= MAX_TOOL_TOTAL_CHARS:
            # Checked before the call, so the true worst case is the cap plus one
            # clipped result: bounded, and much better than an open-ended total.
            yield {
                "type": "tool_result",
                "content": (
                    f"工具结果预算已用完（本次请求累计上限 {MAX_TOOL_TOTAL_CHARS} 字符），"
                    "不再执行新的工具调用。请基于已有信息直接作答。"
                ),
            }
            return
        async for event in self._dispatch(name, args or {}):
            if event.get("type") == "tool_result":
                self._spent += len(str(event.get("content", "")))
            yield event

    def _dispatch(self, name: str, args: dict) -> AsyncIterator[dict]:
        if name not in self._names:
            return self._unknown(name)
        if name in self._fs_tools:
            return self._filesystem(name, args)
        if name == REMEMBER:
            return self._remember(args)
        if name == RECALL:
            return self._recall(args)
        if name == THINK:
            return self._think(args)
        if name == SAVE_DOCUMENT:
            return self._save_document(args)
        # Offered but nothing here runs it. llm.py intercepts search's two tools
        # before they reach the runner, so this is a wiring bug rather than a model
        # mistake — log it, and do not tell the model the tool does not exist while
        # listing it as available.
        log.warning("tool %s was offered to the model but ToolRunner does not implement it", name)
        return self._unhandled(name)

    async def _unhandled(self, name: str) -> AsyncIterator[dict]:
        yield {
            "type": "tool_result",
            "content": f"工具 {name} 这次没有执行成功。请换一种方式，或直接基于已有信息作答。",
        }

    async def _unknown(self, name: str) -> AsyncIterator[dict]:
        offered = "、".join(n for n in self._names if n) or "（本次没有任何可用工具）"
        yield {
            "type": "tool_result",
            "content": (
                f"没有名为 {name} 的工具可用。本次可用的工具是：{offered}。"
                "请改用其中之一，或直接基于已有信息作答。"
            ),
        }

    async def _remember(self, args: dict) -> AsyncIterator[dict]:
        text = str(args.get("text") or "").strip()
        kind = str(args.get("kind") or "").strip()
        yield {"type": "status", "text": f"正在记住：{text[:40] or '（空）'}"}
        try:
            # to_thread rather than a direct call: memory_store is synchronous and
            # takes a threading.Lock, and this runs on the event loop that is also
            # streaming tokens to the browser.
            record = await asyncio.to_thread(add_memory, self._memory_file, text, kind)
        except MemoryStoreError as exc:
            yield {"type": "tool_result", "content": f"没能记住：{exc}"}
            return
        except OSError as exc:
            # memory_store raises on a failed write by design; the model can retry, and
            # an unwritten note must never be reported as saved.
            log.warning("could not write the memory file: %s", exc)
            yield {"type": "tool_result", "content": "没能记住：写记忆文件失败，请稍后再试。"}
            return
        yield {
            "type": "tool_result",
            "content": f"已记住（{record.get('kind', 'fact')}）：{record.get('text', '')}",
        }

    async def _recall(self, args: dict) -> AsyncIterator[dict]:
        query = str(args.get("query") or "").strip()
        yield {"type": "status", "text": f"正在查记忆：{query[:40] or '（空）'}"}
        if not query:
            yield {
                "type": "tool_result",
                "content": "recall 需要一个关键词。请给出 1~3 个核心词，例如「导出 格式」。",
            }
            return
        hits = await asyncio.to_thread(search_memories, self._memory_file, query, RECALL_LIMIT)
        if not hits:
            yield {
                "type": "tool_result",
                "content": f"没有找到与「{query}」相关的记忆。可以换一种说法再查，或直接作答。",
            }
            return
        lines = "\n".join(
            f"- [{hit.get('kind', 'fact')}] {hit.get('text', '')}"
            f"（记于 {str(hit.get('created', ''))[:10]}）"
            for hit in hits
        )
        yield {
            "type": "tool_result",
            "content": (
                f"查到的相关记忆：\n{lines}\n"
                "（这些是以前记下的，可能已过时；与用户当前的说法冲突时以用户为准。）"
            ),
        }

    async def _think(self, args: dict) -> AsyncIterator[dict]:
        thought = str(args.get("thought") or "").strip()
        if not thought:
            yield {"type": "tool_result", "content": "think 需要 thought 参数：写下这一步的想法。"}
            return
        if self._thoughts >= MAX_THOUGHTS:
            yield {
                "type": "tool_result",
                "content": f"分步思考已达上限（{MAX_THOUGHTS} 步），请直接给出答案。",
            }
            return
        self._thoughts += 1
        # A `reasoning` event, not a new event type: the browser already renders those
        # into the 「思考过程」 <details> block and already persists them with the
        # message, so the steps survive a refresh for free.
        yield {"type": "reasoning", "text": f"\n第 {self._thoughts}/{MAX_THOUGHTS} 步：{thought}\n"}
        left = MAX_THOUGHTS - self._thoughts
        yield {
            "type": "tool_result",
            "content": (
                f"已记录第 {self._thoughts} 步。"
                + (f"还可以再想 {left} 步；" if left else "步数已用完；")
                + "想清楚了就直接作答。"
            ),
        }

    async def _save_document(self, args: dict) -> AsyncIterator[dict]:
        content = str(args.get("content") or "")
        fmt = str(args.get("format") or "").strip().lower()
        # 120 because StoredArtifact.title is capped there: an over-long title must not
        # turn "save this session" into a 422 after the user already has the file.
        title = str(args.get("title") or "").strip()[:120]

        if fmt not in ARTIFACT_FORMATS:
            yield {
                "type": "tool_result",
                "content": (
                    f"save_document 的 format 必须是 {'、'.join(ARTIFACT_FORMATS)} 之一，"
                    f"收到的是「{fmt or '（空）'}」。没有生成文件，请修正后重试。"
                ),
            }
            return
        if not content.strip():
            yield {
                "type": "tool_result",
                "content": "save_document 需要 content：一整篇 Markdown 源文本。空文档没有生成。",
            }
            return
        if len(content) > MAX_ARTIFACT_CHARS:
            yield {
                "type": "tool_result",
                "content": (
                    f"文档太长（{len(content)} 字符，上限 {MAX_ARTIFACT_CHARS}），没有生成。"
                    "请精炼内容后重试。"
                ),
            }
            return

        name = safe_artifact_name(args.get("filename"), fmt)
        # No status event here, unlike _remember: there is nothing to await, and the chip
        # the browser renders for this artifact is the feedback — a 「正在生成文档」 line
        # would still be sitting in the bubble after the download finished.
        #
        # An `artifact` event, which llm.py forwards untouched and main.py relays verbatim
        # (its SSE line is `sse(event["type"], event)`, so a new type needs no relay
        # change — the same free ride `reasoning` got). The server renders nothing:
        # export.py refuses md/html/csv on purpose, and all five browser paths already
        # take Markdown source.
        yield {
            "type": "artifact",
            "filename": name,
            "format": fmt,
            "title": title,
            "content": content,
            "chars": len(content),
        }
        # Deliberately tiny, and the reason is arithmetic rather than taste: run() charges
        # every tool_result against MAX_TOOL_TOTAL_CHARS, so echoing the document back
        # would spend a tenth of this request's entire tool budget on text the model wrote
        # one round ago and still has in its own assistant turn. All it needs to know is
        # that the file reached the user.
        yield {
            "type": "tool_result",
            "content": f"已生成《{name}》（{fmt}，{len(content)} 字符），文件已交给用户下载。",
        }

    async def _filesystem(self, name: str, args: dict) -> AsyncIterator[dict]:
        paths = _paths_in(args)
        for candidate in paths:
            reason = is_denied(candidate, self._roots)
            if reason:
                log.warning("denied %s on %r: %s", name, candidate, reason)
                yield {"type": "status", "text": f"已拒绝访问：{candidate[:90]}"}
                yield {
                    "type": "tool_result",
                    "content": f"这个路径不允许访问：{reason}",
                }
                return

        writing = name in self._fs_write_tools
        target = paths[0][:90] if paths else name
        if writing:
            # Every write is logged with its target. The user ticked a box to allow
            # this, and a box is not an audit trail.
            log.warning("mcp write via %s: %s", name, paths or "(no path argument)")
            yield {"type": "status", "text": f"正在写入：{name} → {target}"}
        else:
            yield {"type": "status", "text": f"正在查看：{target}"}

        if self._mcp is None:
            yield {"type": "tool_result", "content": "文件工具不可用：MCP 未启用。"}
            return
        try:
            client = await self._mcp.ensure_started()
            text = await client.call_tool(name, args)
        except Exception as exc:  # MCPError, plus anything a dead child can throw
            # Reported as a result, not raised: one failed tool call should cost the
            # model a wrong guess, not the whole request.
            log.warning("mcp call %s failed: %s", name, exc)
            yield {"type": "tool_result", "content": f"文件工具调用失败：{exc}"}
            return
        yield {"type": "tool_result", "content": clip(text)}
