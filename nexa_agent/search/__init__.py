"""可插拔搜索 provider 层。

用法::

    from nexa_agent.search import get_default_router

    router = get_default_router()
    results, provider = router.search("DeepSeek Harness", max_results=5)

provider 顺序、SearXNG 地址、Exa key、熔断阈值等均由 config.SEARCH_CONFIG 控制。
"""
from __future__ import annotations

from .base import ProviderMetrics, SearchProvider, SearchResult
from .enrich import enrich_results
from .providers import (
    DDGProvider,
    ExaProvider,
    SearXNGProvider,
    TavilyProvider,
    build_provider,
)
from .router import SearchRouter

_default_router: SearchRouter | None = None


def build_router_from_config(cfg: dict) -> SearchRouter:
    """根据 SEARCH_CONFIG 装配 router（按 provider_order 排序）。"""
    providers = []
    for name in cfg["provider_order"]:
        provider = build_provider(name, cfg)
        if provider is not None:
            providers.append(provider)
    return SearchRouter(
        providers,
        fail_threshold=cfg["health_fail_threshold"],
        cooldown_sec=cfg["health_cooldown_sec"],
    )


def get_default_router() -> SearchRouter:
    """进程级单例 router（复用熔断状态与指标）。"""
    global _default_router
    if _default_router is None:
        from ..config import SEARCH_CONFIG

        _default_router = build_router_from_config(SEARCH_CONFIG)
    return _default_router


__all__ = [
    "SearchResult",
    "SearchProvider",
    "ProviderMetrics",
    "SearchRouter",
    "enrich_results",
    "TavilyProvider",
    "SearXNGProvider",
    "ExaProvider",
    "DDGProvider",
    "build_provider",
    "build_router_from_config",
    "get_default_router",
]
