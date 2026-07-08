"""SearchRouter 单元测试 —— 验证有序降级、健康熔断、指标统计。

用假 provider（不发网络请求）覆盖 router 的核心 Harness 逻辑。
运行：pytest tests/test_search_router.py -v
"""
from __future__ import annotations

import time

import pytest

from nexa_agent.search import SearchRouter, build_router_from_config
from nexa_agent.search.base import SearchProvider, SearchResult


class FakeProvider(SearchProvider):
    """可编程的假 provider：控制可用性、是否抛错、返回结果。"""

    def __init__(self, name, *, available=True, fail=False, results=None):
        self.name = name
        self._available = available
        self._fail = fail
        self._results = results if results is not None else [
            SearchResult(title=f"{name}-result", url=f"http://{name}.test", snippet="s")
        ]
        self.call_count = 0

    def is_available(self) -> bool:
        return self._available

    def search(self, query, max_results):
        self.call_count += 1
        if self._fail:
            raise RuntimeError(f"{self.name} boom")
        return self._results[:max_results]


def test_first_available_provider_wins():
    a, b = FakeProvider("a"), FakeProvider("b")
    router = SearchRouter([a, b])
    results, provider = router.search("q", 5)
    assert provider == "a"
    assert len(results) == 1
    assert b.call_count == 0  # 第一个成功后不应触达第二个


def test_falls_back_on_failure():
    a = FakeProvider("a", fail=True)
    b = FakeProvider("b")
    router = SearchRouter([a, b])
    results, provider = router.search("q", 5)
    assert provider == "b"
    assert a.call_count == 1 and b.call_count == 1


def test_skips_unavailable_without_counting():
    a = FakeProvider("a", available=False)
    b = FakeProvider("b")
    router = SearchRouter([a, b])
    _, provider = router.search("q", 5)
    assert provider == "b"
    assert a.call_count == 0
    assert router.metrics["a"].calls == 0  # 未就绪不计入调用


def test_all_fail_returns_empty():
    a = FakeProvider("a", fail=True)
    b = FakeProvider("b", fail=True)
    router = SearchRouter([a, b])
    results, provider = router.search("q", 5)
    assert results == [] and provider == ""


def test_cooldown_trips_after_threshold_and_skips():
    a = FakeProvider("a", fail=True)
    b = FakeProvider("b")
    router = SearchRouter([a, b], fail_threshold=2, cooldown_sec=60)

    router.search("q", 5)  # a 失败 1
    assert not router.metrics["a"].in_cooldown()
    router.search("q", 5)  # a 失败 2 → 触发冷却
    assert router.metrics["a"].in_cooldown()

    before = a.call_count
    router.search("q", 5)  # a 冷却中应被跳过
    assert a.call_count == before  # 没有再调用 a


def test_cooldown_clears_after_window():
    a = FakeProvider("a", fail=True)
    router = SearchRouter([a], fail_threshold=1, cooldown_sec=60)
    router.search("q", 5)
    assert router.metrics["a"].in_cooldown()
    # 把冷却时间拨到过去，模拟窗口结束
    router.metrics["a"].cooldown_until = time.time() - 1
    assert not router.metrics["a"].in_cooldown()


def test_metrics_track_success_rate_and_latency():
    a = FakeProvider("a")
    router = SearchRouter([a])
    router.search("q", 5)
    router.search("q", 5)
    m = router.metrics["a"]
    assert m.calls == 2 and m.successes == 2
    assert m.success_rate == 1.0
    assert "a:" in router.metrics_summary()


def test_build_router_from_config_respects_order():
    cfg = {
        "provider_order": ["ddg", "searxng"],
        "searxng_base_url": "http://localhost:8888",
        "exa_api_key": "",
        "request_timeout": 5,
        "health_fail_threshold": 3,
        "health_cooldown_sec": 60,
    }
    router = build_router_from_config(cfg)
    assert [p.name for p in router.providers] == ["ddg", "searxng"]


def test_max_results_passed_through():
    many = [SearchResult(title=str(i), url=f"http://{i}") for i in range(10)]
    a = FakeProvider("a", results=many)
    router = SearchRouter([a])
    results, _ = router.search("q", 3)
    assert len(results) == 3
