"""Unified web search tool with Tavily-first and Brave fallback providers."""

from __future__ import annotations

from html.parser import HTMLParser
import logging
import os
from typing import Any
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


SEARCH_BACKEND_ENV = "SEARCH_BACKEND"
TAVILY_API_KEY_ENV = "TAVILY_API_KEY"
BRAVE_API_KEY_ENV = "BRAVE_SEARCH_API_KEY"
REQUEST_TIMEOUT = (5, 20)
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) IELTS-Agent/1.0",
    "Accept": "text/html,application/json,*/*;q=0.8",
}


class _SearchTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style", "nav", "footer", "aside"}:
            self.skip_depth += 1
        elif self.skip_depth == 0 and tag in {"h1", "h2", "h3", "p", "li", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "footer", "aside"} and self.skip_depth > 0:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.skip_depth == 0 and data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        return "\n".join(part for part in self.parts if part.strip()).strip()


def search(
    query: str,
    search_type: str = "web",
    max_results: int = 5,
    domains: list[str] | None = None,
    recency_days: int | None = None,
    need_extract: bool = False,
) -> dict[str, Any]:
    """Search the web through one stable interface used by all upper layers."""

    normalized_type = search_type if search_type in {"web", "news", "images"} else "web"
    max_results = max(1, min(int(max_results or 5), 20))
    if isinstance(domains, str):
        domains = [domains]
    domains = [domain.strip() for domain in domains or [] if domain and domain.strip()]
    provider_order = _provider_order()
    errors: list[str] = []

    for provider in provider_order:
        try:
            if provider == "tavily" and normalized_type == "images":
                errors.append("tavily: image search is not supported by this provider wrapper")
                continue
            if provider == "tavily":
                result = _search_tavily(
                    query=query,
                    search_type=normalized_type,
                    max_results=max_results,
                    domains=domains,
                    recency_days=recency_days,
                    need_extract=need_extract,
                )
            elif provider == "brave":
                result = _search_brave(
                    query=query,
                    search_type=normalized_type,
                    max_results=max_results,
                    domains=domains,
                    recency_days=recency_days,
                    need_extract=need_extract,
                )
            else:
                continue
            if result["success"]:
                return result
            errors.append(str(result.get("error") or f"{provider} returned no results"))
        except Exception as exc:
            logger.warning("Search provider %s failed: %s", provider, exc)
            errors.append(f"{provider}: {exc}")

    return {
        "query": query,
        "results": [],
        "provider": provider_order[0] if provider_order else "none",
        "success": False,
        "error": "; ".join(errors) or "No search provider configured. Set TAVILY_API_KEY or BRAVE_SEARCH_API_KEY.",
        "raw_count": 0,
    }


def search_web(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Backward-compatible wrapper returning only normalized result items."""

    return search(query=query, max_results=max_results).get("results", [])


def _provider_order() -> list[str]:
    backend = os.getenv(SEARCH_BACKEND_ENV, "auto").strip().lower()
    has_tavily = bool(os.getenv(TAVILY_API_KEY_ENV, "").strip())
    has_brave = bool(os.getenv(BRAVE_API_KEY_ENV, "").strip())

    if backend == "tavily":
        return ["tavily", "brave"] if has_brave else ["tavily"]
    if backend == "brave":
        return ["brave", "tavily"] if has_tavily else ["brave"]
    providers: list[str] = []
    if has_tavily:
        providers.append("tavily")
    if has_brave:
        providers.append("brave")
    return providers


def _search_tavily(
    *,
    query: str,
    search_type: str,
    max_results: int,
    domains: list[str],
    recency_days: int | None,
    need_extract: bool,
) -> dict[str, Any]:
    api_key = os.getenv(TAVILY_API_KEY_ENV, "").strip()
    if not api_key:
        return _empty_result(query, "tavily", "TAVILY_API_KEY is not configured")

    payload: dict[str, Any] = {
        "api_key": api_key,
        "query": _apply_domain_filter(query, domains),
        "topic": "news" if search_type == "news" else "general",
        "search_depth": "advanced",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": need_extract,
    }
    if domains:
        payload["include_domains"] = domains
    if recency_days:
        payload["days"] = int(recency_days)

    response = requests.post("https://api.tavily.com/search", json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    raw_results = data.get("results", []) if isinstance(data, dict) else []
    results = [_normalize_tavily_item(item, rank=index, need_extract=need_extract) for index, item in enumerate(raw_results, start=1)]
    results = [item for item in results if item["url"]]
    return {
        "query": query,
        "results": results[:max_results],
        "provider": "tavily",
        "success": bool(results),
        "error": "" if results else "No Tavily results",
        "raw_count": len(raw_results),
    }


def _search_brave(
    *,
    query: str,
    search_type: str,
    max_results: int,
    domains: list[str],
    recency_days: int | None,
    need_extract: bool,
) -> dict[str, Any]:
    api_key = os.getenv(BRAVE_API_KEY_ENV, "").strip() or os.getenv("BRAVE_API_KEY", "").strip()
    if not api_key:
        return _empty_result(query, "brave", "BRAVE_SEARCH_API_KEY is not configured")

    endpoint = {
        "web": "https://api.search.brave.com/res/v1/web/search",
        "news": "https://api.search.brave.com/res/v1/news/search",
        "images": "https://api.search.brave.com/res/v1/images/search",
    }[search_type]
    params: dict[str, Any] = {
        "q": _apply_domain_filter(query, domains),
        "count": max_results,
        "safesearch": "moderate",
    }
    if recency_days:
        params["freshness"] = f"pd{int(recency_days)}"
    headers = {**REQUEST_HEADERS, "X-Subscription-Token": api_key}

    response = requests.get(endpoint, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    raw_results = _brave_raw_results(data, search_type)
    results = [
        _normalize_brave_item(item, rank=index, need_extract=need_extract)
        for index, item in enumerate(raw_results, start=1)
    ]
    results = [item for item in results if item["url"]]
    return {
        "query": query,
        "results": results[:max_results],
        "provider": "brave",
        "success": bool(results),
        "error": "" if results else "No Brave results",
        "raw_count": len(raw_results),
    }


def _normalize_tavily_item(item: dict[str, Any], *, rank: int, need_extract: bool) -> dict[str, Any]:
    url = str(item.get("url") or "").strip()
    content = str(item.get("raw_content") or item.get("content") or "").strip()
    if need_extract and url and not content:
        content = _extract_url_text(url)
    return {
        "rank": rank,
        "title": str(item.get("title") or "").strip(),
        "url": url,
        "snippet": str(item.get("content") or "").strip(),
        "content": content,
        "domain": _domain(url),
        "published_at": str(item.get("published_date") or "").strip(),
    }


def _normalize_brave_item(item: dict[str, Any], *, rank: int, need_extract: bool) -> dict[str, Any]:
    url = str(item.get("url") or "").strip()
    content = _extract_url_text(url) if need_extract and url else ""
    return {
        "rank": rank,
        "title": str(item.get("title") or "").strip(),
        "url": url,
        "snippet": str(item.get("description") or item.get("snippet") or "").strip(),
        "content": content,
        "domain": _domain(url),
        "published_at": str(item.get("age") or item.get("page_age") or "").strip(),
    }


def _brave_raw_results(data: dict[str, Any], search_type: str) -> list[dict[str, Any]]:
    if search_type == "news":
        return list(data.get("results") or data.get("news", {}).get("results") or [])
    if search_type == "images":
        return list(data.get("results") or data.get("images", {}).get("results") or [])
    return list(data.get("web", {}).get("results") or data.get("results") or [])


def _extract_url_text(url: str) -> str:
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type:
            return ""
        extractor = _SearchTextExtractor()
        extractor.feed(response.text)
        return extractor.text()[:12000]
    except Exception as exc:
        logger.debug("Failed to extract search result content url=%s error=%s", url, exc)
        return ""


def _apply_domain_filter(query: str, domains: list[str]) -> str:
    if not domains:
        return query
    filters = " OR ".join(f"site:{domain}" for domain in domains)
    return f"({filters}) {query}"


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def _empty_result(query: str, provider: str, error: str) -> dict[str, Any]:
    return {"query": query, "results": [], "provider": provider, "success": False, "error": error, "raw_count": 0}
