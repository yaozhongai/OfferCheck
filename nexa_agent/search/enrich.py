"""搜索结果增强层 —— 对 top-k 结果抓取正文替换摘要。

动机：DDG 返回的 `content` 摘要偏短，质量不及 Tavily
（Tavily 在云端已抓取并提取正文）。本层复用现有 fetch 链
（Jina Reader → trafilatura）补齐这一差距，使兜底源贴近 Tavily 体验。

特性：
- 并行抓取（ThreadPoolExecutor），总耗时 ≈ 最慢单条，而非累加
- best-effort：单条抓取失败/超时则保留原摘要，绝不影响整体搜索
- 可注入 fetcher，便于单测（不发网络）
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from .base import SearchResult

logger = logging.getLogger("nexa_agent.search")

#: fetcher 签名：url -> 正文纯文本（失败返回空串）
Fetcher = Callable[[str], str]


def _default_fetcher(timeout: int) -> Fetcher:
    """默认 fetcher：复用 tools.extract_url_text（懒导入避免循环依赖）。"""

    def _fetch(url: str) -> str:
        from nexa_agent.tools import extract_url_text

        # 增强层关掉 trafilatura 兜底以提速：Jina 不行就保留原摘要
        text, _engine = extract_url_text(url, timeout=timeout, use_fallback=False)
        return text

    return _fetch


def enrich_results(
    results: list[SearchResult],
    *,
    top_k: int = 3,
    max_chars: int = 1200,
    timeout: int = 12,
    fetcher: Fetcher | None = None,
) -> list[SearchResult]:
    """对前 top_k 条结果抓取正文，替换其 snippet（就地修改并返回）。

    Args:
        results: 搜索结果列表
        top_k: 增强前几条（其余保持原摘要）
        max_chars: 增强后摘要的最大长度
        timeout: 单条抓取超时（秒）
        fetcher: 自定义抓取函数；默认复用 web_fetch 的提取链

    Returns:
        同一个 results 列表（被增强的条目 enriched=True）
    """
    if not results or top_k <= 0:
        return results

    targets = [r for r in results[:top_k] if r.url]
    if not targets:
        return results

    if fetcher is None:
        fetcher = _default_fetcher(timeout)

    enriched_count = 0
    with ThreadPoolExecutor(max_workers=len(targets)) as pool:
        future_to_result = {pool.submit(fetcher, r.url): r for r in targets}
        for future in as_completed(future_to_result):
            result = future_to_result[future]
            try:
                text = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.info("enrich 抓取失败 url=%s: %s（保留原摘要）", result.url, exc)
                continue
            text = (text or "").strip()
            if not text:
                continue
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            result.snippet = text
            result.enriched = True
            enriched_count += 1

    if enriched_count:
        logger.info("enrich 完成：%d/%d 条摘要已用正文增强", enriched_count, len(targets))
    return results
