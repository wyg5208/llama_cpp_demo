"""Web search providers, all reachable from mainland China.

`bing` uses Bing's public RSS output (no API key, personal non-commercial use
per the license text embedded in the feed). `bocha` and `tavily` need keys but
return noticeably better snippets; set them in .env and provider "auto" picks
them up automatically.

cn.bing.com serves the China index regardless of the mkt/cc/ensearch/setlang
parameters, and that index degrades sharply as queries grow: "PEP 703" returns
peps.python.org/pep-0703/ at rank 1, while "PEP 703 free-threaded CPython"
returns a textbook publisher's homepage. search_bing compensates by also
searching the two-token prefix of a long query; the schema below is the second
line of defence. DuckDuckGo, Mojeek, Brave, Qwant, Yahoo, SearXNG and
Marginalia are all unreachable from this network.
"""

from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from urllib.parse import urlsplit

import httpx

from .config import Settings

log = logging.getLogger(__name__)

BING_RSS_URL = "https://cn.bing.com/search"
BOCHA_URL = "https://api.bochaai.com/v1/web-search"
TAVILY_URL = "https://api.tavily.com/search"
# Enough for any article; bounds memory when the model picks a huge page.
MAX_FETCH_BYTES = 2_000_000
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)


class SearchError(RuntimeError):
    pass


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict:
        return asdict(self)


async def _bing_rss(client: httpx.AsyncClient, query: str, count: int) -> list[SearchResult]:
    resp = await client.get(
        BING_RSS_URL,
        params={"q": query, "format": "rss", "count": count},
        headers={"User-Agent": USER_AGENT},
    )
    resp.raise_for_status()
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        raise SearchError(f"Bing returned unparseable XML: {exc}") from exc

    results = []
    for item in root.iter("item"):
        link = (item.findtext("link") or "").strip()
        if not link:
            continue
        results.append(
            SearchResult(
                title=(item.findtext("title") or "").strip(),
                url=link,
                snippet=(item.findtext("description") or "").strip(),
            )
        )
        if len(results) >= count:
            break
    return results


async def search_bing(client: httpx.AsyncClient, query: str, count: int) -> list[SearchResult]:
    """Bing China, with a shortened retry for long queries.

    The China index drops relevance as terms are added: "PEP 703" returns the
    official PEP at rank 1, "PEP 703 free-threaded CPython" returns a textbook
    publisher. Searching the two-token prefix as well and interleaving the two
    result lists keeps the specific pages the long query missed.
    """
    terms = query.split()
    variants = [query] if len(terms) <= 2 else [query, " ".join(terms[:2])]
    batches = await asyncio.gather(*(_bing_rss(client, v, count) for v in variants))

    merged: list[SearchResult] = []
    seen: set[str] = set()
    for rank in range(count):
        for batch in batches:
            if rank < len(batch) and batch[rank].url not in seen:
                seen.add(batch[rank].url)
                merged.append(batch[rank])
            if len(merged) >= count:
                return merged
    return merged


async def search_bocha(
    client: httpx.AsyncClient, query: str, count: int, api_key: str
) -> list[SearchResult]:
    resp = await client.post(
        BOCHA_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"query": query, "count": count, "summary": True, "freshness": "noLimit"},
    )
    resp.raise_for_status()
    payload = resp.json()
    pages = (payload.get("data") or {}).get("webPages") or {}
    return [
        SearchResult(
            title=item.get("name") or "",
            url=item.get("url") or "",
            snippet=item.get("summary") or item.get("snippet") or "",
        )
        for item in pages.get("value") or []
        if item.get("url")
    ][:count]


async def search_tavily(
    client: httpx.AsyncClient, query: str, count: int, api_key: str
) -> list[SearchResult]:
    resp = await client.post(
        TAVILY_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"query": query, "max_results": count, "search_depth": "basic"},
    )
    resp.raise_for_status()
    payload = resp.json()
    return [
        SearchResult(
            title=item.get("title") or "",
            url=item.get("url") or "",
            snippet=item.get("content") or "",
        )
        for item in payload.get("results") or []
        if item.get("url")
    ][:count]


async def web_search(settings: Settings, query: str) -> list[SearchResult]:
    """Run one search, returning [] rather than raising when the provider fails."""
    provider = settings.resolved_search_provider()
    count = settings.search_max_results
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        try:
            if provider == "bocha":
                return await search_bocha(client, query, count, settings.bocha_api_key)
            if provider == "tavily":
                return await search_tavily(client, query, count, settings.tavily_api_key)
            return await search_bing(client, query, count)
        except (httpx.HTTPError, SearchError) as exc:
            log.warning("web_search(%r) via %s failed: %s", query, provider, exc)
            return []


def format_for_model(results: list[SearchResult], start_index: int = 1) -> str:
    if not results:
        return "（未检索到结果，请基于已有知识谨慎回答，并说明未能联网确认。）"
    lines = []
    for offset, item in enumerate(results):
        index = start_index + offset
        lines.append(f"[{index}] {item.title}\nURL: {item.url}\n摘要: {item.snippet}")
    return "\n\n".join(lines)


class _TextExtractor(HTMLParser):
    """Pulls visible text out of HTML, preferring the main content region.

    Pages like python.org's release announcements put well over a thousand
    characters of navigation chrome ahead of the actual text, which would crowd
    out the answer inside the character budget handed to the model.
    """

    SKIP = {"script", "style", "noscript", "svg", "template", "head"}
    CONTENT_TAGS = {"main", "article"}
    CONTENT_IDS = {"content", "main-content", "maincontent"}
    # Only these are tracked on the stack: p/li/td close implicitly in HTML and
    # never emit an end tag, which would leave the stack permanently unbalanced.
    TRACKED = {"div", "section", "main", "article", "body", "header", "footer",
               "nav", "aside"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._all: list[str] = []
        self._main: list[str] = []
        self._skip_depth = 0
        self._stack: list[bool] = []

    def _is_content(self, tag: str, attrs) -> bool:
        if tag in self.CONTENT_TAGS:
            return True
        if tag not in {"div", "section"}:
            return False
        for name, value in attrs:
            if not value:
                continue
            if name == "role" and value.strip().lower() == "main":
                return True
            if name == "id" and value.strip().lower() in self.CONTENT_IDS:
                return True
        return False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.SKIP:
            self._skip_depth += 1
        if tag in self.TRACKED:
            self._stack.append(self._is_content(tag, attrs))

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        if tag in self.TRACKED and self._stack:
            self._stack.pop()

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not data.strip():
            return
        chunk = data.strip()
        self._all.append(chunk)
        if any(self._stack):
            self._main.append(chunk)

    def text(self) -> str:
        # Too little to be the real body means the container guess was wrong.
        body = self._main if len("".join(self._main)) >= 200 else self._all
        return re.sub(r"\n{3,}", "\n\n", "\n".join(body))


async def _read_page(url: str, max_chars: int) -> str | None:
    """Download one page and return its main text, or None if it is unreadable."""
    if not re.match(r"^https?://", url, re.I):
        return None
    try:
        async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
            async with client.stream("GET", url, headers={"User-Agent": USER_AGENT}) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")
                if "html" not in content_type and "text" not in content_type:
                    return None
                raw = bytearray()
                async for chunk in resp.aiter_bytes(65536):
                    raw += chunk
                    if len(raw) >= MAX_FETCH_BYTES:
                        break
                encoding = resp.encoding or "utf-8"
        parser = _TextExtractor()
        parser.feed(bytes(raw).decode(encoding, "replace"))
        text = parser.text()
    except httpx.HTTPError as exc:
        log.warning("fetch_url(%r) failed: %s", url, exc)
        return None

    if not text.strip():
        return None
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…（正文已截断）"
    return text


async def fetch_url(url: str, max_chars: int = 4000) -> str:
    """Read one page as plain text. Returns an explanation instead of raising.

    4000 characters is roughly 4000 tokens for Chinese, already a quarter of the
    default 16k context; the main-content extraction reaches the answer well
    inside that.
    """
    text = await _read_page(url, max_chars)
    if text is None:
        return f"（无法读取 {url[:120]}：链接不可达，或它不是可阅读的文本网页。）"
    return f"以下是 {url} 的正文：\n\n{text}"


async def read_best_result(
    results: list[SearchResult], start_index: int = 1, max_chars: int = 1500
) -> str:
    """Read the body of the most promising result, or "" if none is readable.

    The key-free backend's snippets are a sentence or two, rarely enough to
    answer from, and a small model often gives up rather than asking for the
    page: given the same history three times, Gemma 4 E4B called fetch_url once
    and twice replied that it could not list the features, even though the
    correct release page sat at index 2 every time. So read it for them.
    Bare homepages are skipped — they carry no article text.
    """
    candidates = [
        (offset, r)
        for offset, r in enumerate(results)
        if urlsplit(r.url).path.strip("/")
    ][:2]
    for offset, result in candidates:
        text = await _read_page(result.url, max_chars)
        if text:
            # Naming the citation number keeps the model from attributing the
            # excerpt to whichever source it last saw.
            return (
                f"\n\n---\n【来源 {start_index + offset} 的正文节选】"
                f"{result.title}\n{result.url}\n\n{text}"
            )
    return ""


WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "联网检索实时信息。当问题涉及新闻、时事、价格、版本发布、"
            "统计数据、文档更新或任何你不确定的事实时调用。\n"
            "重要：query 必须是 1~3 个关键词，绝对不要把用户的整句话当作 query。"
            "本检索后端在关键词变长时相关性会急剧下降（例如查 'PEP 703' 能命中官方原文，"
            "查 'PEP 703 free-threaded CPython' 却会返回无关的商业站点）。\n"
            "宁可分多次调用、每次一个短 query，也不要用一个长 query。"
            "同一问题可以调用多次，从不同角度补齐信息后再作答。\n"
            "注意：返回的摘要很短，往往不足以直接回答问题。"
            "如果某条结果的链接看起来正是权威出处（官方文档、发布公告、原始新闻），"
            "应改用 fetch_url 读取该链接的正文，而不是反复用相近的关键词重复检索。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "1~3 个关键词，优先只写核心专有名词或实体名，"
                        "例如 'PEP 703'、'Python 3.13'、'黑神话悟空 销量'。"
                        "中文话题用中文关键词，英文技术名词保留原文。"
                        "不要加 '是什么'、'有哪些'、'new features' 之类的修饰词。"
                    ),
                }
            },
            "required": ["query"],
        },
    },
}

FETCH_URL_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": (
            "读取指定网页的正文并返回纯文本。"
            "用于 web_search 找到了权威链接、但摘要不足以回答问题时。"
            "一次只读一个链接，读完即可作答，不要连续读取大量页面。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "要读取的完整 http/https 链接，通常来自 web_search 的结果。",
                }
            },
            "required": ["url"],
        },
    },
}

TOOLS = [WEB_SEARCH_TOOL, FETCH_URL_TOOL]
