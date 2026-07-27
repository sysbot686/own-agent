from __future__ import annotations

import html
import re
from urllib.parse import unquote, urlencode, urlparse, parse_qs

import httpx

from own_agent.tools.context import ExecutionContext
from own_agent.tools.types import ToolResult, ToolSpec

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
_TIMEOUT = 30.0
_MAX_TEXT = 15000


def _strip_html(raw: str) -> str:
    raw = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"<style[^>]*>.*?</style>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"<nav[^>]*>.*?</nav>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"<footer[^>]*>.*?</footer>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"<header[^>]*>.*?</header>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"<br\s*/?>", "\n", raw, flags=re.IGNORECASE)
    raw = re.sub(r"</(?:p|div|li|tr|h[1-6]|blockquote|pre|section|ol|ul)>", "\n", raw, flags=re.IGNORECASE)
    raw = re.sub(r"<[^>]+>", "", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"(?:\s*\n){3,}", "\n\n", raw)
    return raw.strip()


async def _fetch_http(url: str) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if "text" not in ct and "html" not in ct and "json" not in ct and "xml" not in ct:
                return None
            return resp.text
    except Exception:
        return None


def web_search(query: str, max_results: int = 5, ctx: ExecutionContext | None = None, **kwargs) -> str:
    """Search the web via DuckDuckGo (free, no API key needed)."""
    if not query.strip():
        return "Error: empty query"

    params = {"q": query.strip()}
    url = f"https://html.duckduckgo.com/html/?{urlencode(params)}"

    raw = asyncio_run(_fetch_http(url))
    if raw is None:
        return "Error: failed to fetch search results"

    results = _parse_ddg_results(raw)
    if not results:
        return "Error: no results found or search was blocked"

    out = [f"Web search results for: {query}", "=" * 50]
    for i, (title, snippet, link) in enumerate(results[:max_results], 1):
        out.append(f"\n{i}. {title}")
        out.append(f"   URL: {link}")
        out.append(f"   {snippet[:300]}")
    return "\n".join(out)


def _clean_ddg_url(raw_url: str) -> str:
    """Extract real URL from DuckDuckGo redirect link."""
    raw_url = raw_url.strip()
    if raw_url.startswith("//"):
        raw_url = "https:" + raw_url
    parsed = urlparse(raw_url)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        qs = parse_qs(parsed.query)
        uddg = qs.get("uddg", [None])[0]
        if uddg:
            return unquote(uddg)
    return raw_url


def _is_ad_link(raw_url: str) -> bool:
    return "ad_provider" in raw_url or "ad_type" in raw_url


def _parse_ddg_results(html_text: str) -> list[tuple[str, str, str]]:
    results: list[tuple[str, str, str]] = []
    for m in re.finditer(
        r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        html_text,
        re.DOTALL,
    ):
        link = m.group(1)
        if _is_ad_link(link):
            continue
        title = _strip_html(m.group(2)).strip()

        after = html_text[m.end() : m.end() + 2000]
        snippet_m = re.search(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', after, re.DOTALL)
        snippet = _strip_html(snippet_m.group(1)).strip() if snippet_m else ""

        results.append((title, snippet, _clean_ddg_url(link)))
    return results


def web_fetch(url: str, max_length: int = _MAX_TEXT, ctx: ExecutionContext | None = None, **kwargs) -> str:
    """Fetch a URL and return its text content."""
    if not url.strip():
        return "Error: empty URL"

    raw = asyncio_run(_fetch_http(url))
    if raw is None:
        return "Error: failed to fetch URL"

    text = _strip_html(raw)
    if not text:
        return "Error: page appears to have no text content"

    if len(text) > max_length:
        text = text[:max_length] + "\n\n...(truncated)"

    return text


def asyncio_run(coro):
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    import threading
    barrier = threading.Event()
    result = [None]
    exc_info = [None]

    async def _run():
        try:
            result[0] = await coro
        except BaseException:
            exc_info[0] = __import__("sys").exc_info()
        finally:
            barrier.set()

    def _start():
        import asyncio
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            new_loop.run_until_complete(_run())
        finally:
            new_loop.close()

    t = threading.Thread(target=_start, daemon=True)
    t.start()
    barrier.wait()
    if exc_info[0]:
        raise exc_info[0][1].with_traceback(exc_info[0][2])
    return result[0]


WEB_SEARCH_SPEC = ToolSpec(
    name="web_search",
    description="Search the web for information. Returns titles, snippets, and URLs (free, no API key needed).",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
    categories=("web",),
)

WEB_FETCH_SPEC = ToolSpec(
    name="web_fetch",
    description="Fetch a URL and return its text content (HTML stripped).",
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch.",
            },
            "max_length": {
                "type": "integer",
                "description": "Maximum characters to return (default 15000).",
                "default": _MAX_TEXT,
            },
        },
        "required": ["url"],
    },
    categories=("web",),
)
