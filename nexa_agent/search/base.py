"""搜索 provider 抽象层 — 基础类型与接口。

设计目标（对应 DeepSeek Harness JD「工具层设计 / 可观测」）：
- SearchResult：归一化结果，屏蔽各家 API 的字段差异
- SearchProvider：薄接口，每个后端只负责"发请求 + 归一化"
- 健康度与降级逻辑交给 router（见 router.py），provider 保持无状态
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """归一化的单条搜索结果（与 Tavily 输出同构）。"""

    title: str
    url: str
    snippet: str = ""
    score: float | None = None
    engine: str = ""               # 来源引擎或 provider 名
    published_date: str | None = None
    enriched: bool = False         # snippet 是否已被增强层（抓取正文）替换


@dataclass
class ProviderMetrics:
    """单 provider 的运行指标 + 熔断状态。由 router 维护。"""

    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_latency: float = 0.0
    consecutive_failures: int = 0
    cooldown_until: float = 0.0    # epoch 秒；> now 表示处于冷却（被跳过）

    @property
    def success_rate(self) -> float:
        return self.successes / self.calls if self.calls else 0.0

    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.successes if self.successes else 0.0

    def in_cooldown(self) -> bool:
        return time.time() < self.cooldown_until

    def record_success(self, latency: float) -> None:
        self.calls += 1
        self.successes += 1
        self.total_latency += latency
        self.consecutive_failures = 0
        self.cooldown_until = 0.0

    def record_failure(self, fail_threshold: int, cooldown_sec: int) -> None:
        self.calls += 1
        self.failures += 1
        self.consecutive_failures += 1
        if self.consecutive_failures >= fail_threshold:
            self.cooldown_until = time.time() + cooldown_sec


class SearchProvider(ABC):
    """搜索后端薄接口。子类只实现 `is_available` 和 `search`。"""

    #: provider 唯一名，与 SEARCH_CONFIG.provider_order 中的标识对应
    name: str = "base"

    def is_available(self) -> bool:
        """配置/依赖是否就绪（不发网络请求）。默认可用。"""
        return True

    @abstractmethod
    def search(self, query: str, max_results: int) -> list[SearchResult]:
        """执行搜索，返回归一化结果列表。

        约定：失败时抛异常（由 router 捕获并降级），不要静默返回空。
        无结果属正常情况，返回空列表即可。
        """
        raise NotImplementedError
