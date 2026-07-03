"""搜索 provider 具体实现：Tavily / SearXNG / Exa / DuckDuckGo。

每个 provider 只做两件事：判断自身是否就绪、发请求并归一化为 SearchResult。
降级、熔断、指标统计全部交给 router。
"""
from __future__ import annotations

import os

import requests

from .base import SearchProvider, SearchResult


class TavilyProvider(SearchProvider):
    """Tavily Search API —— 额度够时质量最佳（1000/月免费）。"""

    name = "tavily"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("TAVILY_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        from tavily import TavilyClient

        client = TavilyClient(api_key=self.api_key)
        resp = client.search(query, max_results=max_results)
        out: list[SearchResult] = []
        for r in resp.get("results", []):
            out.append(
                SearchResult(
                    title=r.get("title", "无标题"),
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                    score=r.get("score"),
                    engine="tavily",
                    published_date=r.get("published_date"),
                )
            )
        return out


class SearXNGProvider(SearchProvider):
    """自建 SearXNG 元搜索后端 —— 无额度、无速率限制（见 searxng/ 目录）。"""

    name = "searxng"

    def __init__(self, base_url: str, timeout: int = 15) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        return bool(self.base_url)

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        resp = requests.get(
            f"{self.base_url}/search",
            params={"q": query, "format": "json"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        out: list[SearchResult] = []
        for r in data.get("results", [])[:max_results]:
            out.append(
                SearchResult(
                    title=r.get("title", "无标题"),
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                    score=r.get("score"),
                    engine=r.get("engine", "searxng"),
                    published_date=r.get("publishedDate"),
                )
            )
        return out


class ExaProvider(SearchProvider):
    """Exa 神经检索 API —— 1000/月免费，免信用卡。"""

    name = "exa"
    API_URL = "https://api.exa.ai/search"

    def __init__(self, api_key: str | None = None, timeout: int = 15) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("EXA_API_KEY", "")
        self.timeout = timeout

    def is_available(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        resp = requests.post(
            self.API_URL,
            headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
            json={
                "query": query,
                "numResults": max_results,
                "contents": {"text": {"maxCharacters": 500}},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        out: list[SearchResult] = []
        for r in data.get("results", []):
            out.append(
                SearchResult(
                    title=r.get("title") or "无标题",
                    url=r.get("url", ""),
                    snippet=(r.get("text") or "").strip(),
                    score=r.get("score"),
                    engine="exa",
                    published_date=r.get("publishedDate"),
                )
            )
        return out


class DDGProvider(SearchProvider):
    """DuckDuckGo —— 无 key，最后兜底（数据中心 IP 易限流，故置末位）。"""

    name = "ddg"

    def is_available(self) -> bool:
        try:
            import ddgs  # noqa: F401
            return True
        except ImportError:
            try:
                import duckduckgo_search  # noqa: F401
                return True
            except ImportError:
                return False

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS  # type: ignore[import]

        out: list[SearchResult] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                out.append(
                    SearchResult(
                        title=r.get("title", "无标题"),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                        engine="ddg",
                    )
                )
        return out


#: provider 名 → 构造函数（从 SEARCH_CONFIG 装配）
def build_provider(name: str, cfg: dict) -> SearchProvider | None:
    """根据名字和配置构造 provider；未知名返回 None。"""
    if name == "tavily":
        return TavilyProvider()
    if name == "searxng":
        return SearXNGProvider(cfg["searxng_base_url"], cfg["request_timeout"])
    if name == "exa":
        return ExaProvider(cfg["exa_api_key"], cfg["request_timeout"])
    if name == "ddg":
        return DDGProvider()
    return None
