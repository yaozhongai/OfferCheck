"""增强层 enrich_results 单元测试 —— 注入假 fetcher，不发网络请求。

运行：pytest tests/test_search_enrich.py -v
"""
from __future__ import annotations

from nexa_agent.search import enrich_results
from nexa_agent.search.base import SearchResult


def _results(n: int) -> list[SearchResult]:
    return [
        SearchResult(title=f"t{i}", url=f"http://x/{i}", snippet=f"short-{i}")
        for i in range(n)
    ]


def test_enriches_only_top_k():
    results = _results(5)
    fetcher = lambda url: f"FULLTEXT for {url}"  # noqa: E731
    enrich_results(results, top_k=2, max_chars=1000, fetcher=fetcher)

    assert results[0].enriched and results[1].enriched
    assert not results[2].enriched
    assert results[0].snippet == "FULLTEXT for http://x/0"
    assert results[2].snippet == "short-2"  # 未触达，保留原摘要


def test_truncates_to_max_chars():
    results = _results(1)
    fetcher = lambda url: "A" * 5000  # noqa: E731
    enrich_results(results, top_k=1, max_chars=100, fetcher=fetcher)

    assert results[0].enriched
    assert results[0].snippet == "A" * 100 + "..."


def test_fetch_failure_keeps_original_snippet():
    results = _results(2)

    def fetcher(url):
        if url.endswith("/0"):
            raise RuntimeError("boom")
        return "ok-fulltext"

    enrich_results(results, top_k=2, max_chars=1000, fetcher=fetcher)

    assert results[0].snippet == "short-0"        # 抓取失败保留原值
    assert not results[0].enriched
    assert results[1].snippet == "ok-fulltext"    # 成功的不受影响
    assert results[1].enriched


def test_empty_text_keeps_original():
    results = _results(1)
    enrich_results(results, top_k=1, fetcher=lambda url: "   ")
    assert results[0].snippet == "short-0"
    assert not results[0].enriched


def test_skips_results_without_url():
    results = [SearchResult(title="t", url="", snippet="orig")]
    enrich_results(results, top_k=3, fetcher=lambda url: "should-not-be-used")
    assert results[0].snippet == "orig"
    assert not results[0].enriched


def test_empty_results_is_noop():
    assert enrich_results([], top_k=3, fetcher=lambda url: "x") == []


def test_top_k_zero_is_noop():
    results = _results(3)
    enrich_results(results, top_k=0, fetcher=lambda url: "x")
    assert all(not r.enriched for r in results)
