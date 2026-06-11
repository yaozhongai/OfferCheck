"""
web_search / wikipedia_search — 网络搜索工具

移植自 react_exp/tools.py，适配 Nexa Agent 架构。
"""

from __future__ import annotations

import os
import re
import requests

from app.tools import register
from app.utils.logger_config import get_logger

logger = get_logger("tools.web_search")


@register(
    name="web_search",
    description="搜索互联网，返回前 5 条结果（含标题、摘要、链接）。Tavily 优先，DuckDuckGo 兜底",
    signature="web_search(query)",
    examples=["web_search(2025年诺贝尔奖得主)"],
)
def web_search(query: str) -> str:
    query = query.strip()
    if not query:
        return "[错误] web_search: 查询内容不能为空"

    logger.info("web_search query=%s", query[:80])

    # Tavily 优先
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if tavily_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_key)
            response = client.search(query, max_results=5)
            results = response.get("results", [])
            if results:
                lines = [f"Tavily 搜索 '{query}' 结果（共 {len(results)} 条）:\n"]
                for i, r in enumerate(results, 1):
                    title = r.get("title", "无标题")
                    url = r.get("url", "")
                    content = r.get("content", "")[:300]
                    lines.append(f"{i}. {title}\n   链接: {url}\n   摘要: {content}\n")
                return "\n".join(lines)
        except Exception as exc:
            logger.warning("Tavily 失败, 回退 DuckDuckGo: %s", exc)

    # DuckDuckGo 兜底
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(r)
        if not results:
            return f"未找到与 '{query}' 相关的结果。"
        lines = [f"DuckDuckGo 搜索 '{query}' 结果:\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', '')}\n   链接: {r.get('href', '')}\n   摘要: {(r.get('body', '') or '')[:300]}\n")
        return "\n".join(lines)
    except ImportError:
        return "[错误] web_search: 搜索引擎不可用（Tavily 未配置 + duckduckgo_search 未安装）"
    except Exception as exc:
        return f"[错误] web_search 执行失败: {exc}"


@register(
    name="wikipedia_search",
    description="搜索 Wikipedia，返回最相关文章的摘要（最多 800 字符），自动选择中/英文端点",
    signature="wikipedia_search(query)",
    examples=["wikipedia_search(transformer deep learning)"],
)
def wikipedia_search(query: str) -> str:
    query = query.strip()
    if not query:
        return "[错误] wikipedia_search: 查询不能为空"

    logger.info("wikipedia_search query=%s", query[:80])

    has_chinese = bool(re.search(r"[一-鿿]", query))
    lang = "zh" if has_chinese else "en"
    api_url = f"https://{lang}.wikipedia.org/w/api.php"
    headers = {"User-Agent": "NexaAgent-ReAct/1.0"}

    try:
        search_res = requests.get(api_url, params={
            "action": "query", "list": "search", "srsearch": query,
            "format": "json", "srlimit": 1,
        }, headers=headers, timeout=10)
        search_res.raise_for_status()
        hits = search_res.json().get("query", {}).get("search", [])

        if not hits and lang == "zh":
            api_url = "https://en.wikipedia.org/w/api.php"
            search_res = requests.get(api_url, params={
                "action": "query", "list": "search", "srsearch": query,
                "format": "json", "srlimit": 1,
            }, headers=headers, timeout=10)
            search_res.raise_for_status()
            hits = search_res.json().get("query", {}).get("search", [])

        if not hits:
            return f"Wikipedia 未找到与 '{query}' 相关的条目。"

        title = hits[0]["title"]
        content_res = requests.get(api_url, params={
            "action": "query", "prop": "extracts", "exintro": True,
            "explaintext": True, "titles": title, "format": "json",
        }, headers=headers, timeout=10)
        content_res.raise_for_status()
        pages = content_res.json().get("query", {}).get("pages", {})

        for pid, info in pages.items():
            if pid == "-1":
                continue
            extract = (info.get("extract") or "")[:800]
            return f"Wikipedia - {title}\n链接: https://{lang}.wikipedia.org/wiki/{requests.utils.quote(title)}\n\n{extract}"

        return f"Wikipedia 找到条目 '{title}'，无正文摘要。"
    except requests.exceptions.Timeout:
        return "[错误] wikipedia_search: 请求超时"
    except Exception as exc:
        return f"[错误] wikipedia_search 执行失败: {exc}"
