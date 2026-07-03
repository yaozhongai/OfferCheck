"""SearchRouter —— 有序降级 + 健康熔断 + 可观测指标。

这是搜索能力的 Harness 核心：
- 按配置顺序尝试 provider，第一个成功的胜出
- 单 provider 连续失败触发冷却，一段时间内自动跳过
- 每 provider 维护调用/成功率/延迟指标，可导出供日志与面试展示
"""
from __future__ import annotations

import logging
import time

from .base import ProviderMetrics, SearchProvider, SearchResult

logger = logging.getLogger("nexa_agent.search")


class SearchRouter:
    def __init__(
        self,
        providers: list[SearchProvider],
        fail_threshold: int = 3,
        cooldown_sec: int = 120,
    ) -> None:
        self.providers = providers
        self.fail_threshold = fail_threshold
        self.cooldown_sec = cooldown_sec
        self.metrics: dict[str, ProviderMetrics] = {p.name: ProviderMetrics() for p in providers}

    def search(self, query: str, max_results: int) -> tuple[list[SearchResult], str]:
        """按顺序尝试 provider，返回 (结果列表, 命中的 provider 名)。

        全部失败时返回 ([], "")。无结果但成功的 provider 也会被采纳并返回。
        """
        last_error: Exception | None = None

        for provider in self.providers:
            name = provider.name
            metric = self.metrics[name]

            if not provider.is_available():
                logger.debug("search provider %s 未就绪（缺 key/依赖），跳过", name)
                continue
            if metric.in_cooldown():
                remain = int(metric.cooldown_until - time.time())
                logger.info("search provider %s 冷却中（剩 %ds），跳过", name, remain)
                continue

            start = time.time()
            try:
                results = provider.search(query, max_results)
            except Exception as exc:  # noqa: BLE001
                latency = time.time() - start
                metric.record_failure(self.fail_threshold, self.cooldown_sec)
                last_error = exc
                cd = "，进入冷却" if metric.in_cooldown() else ""
                logger.warning(
                    "search provider %s 失败(%.2fs): %s%s", name, latency, exc, cd
                )
                continue

            latency = time.time() - start
            metric.record_success(latency)
            logger.info(
                "search provider %s 命中 results=%d latency=%.2fs", name, len(results), latency
            )
            return results, name

        if last_error is not None:
            logger.error("search 所有 provider 均失败，最后错误: %s", last_error)
        else:
            logger.error("search 无可用 provider（检查配置/依赖）")
        return [], ""

    def metrics_summary(self) -> str:
        """导出指标摘要（日志/调试/面试展示用）。"""
        lines = ["搜索 provider 指标:"]
        for name, m in self.metrics.items():
            cd = " [冷却中]" if m.in_cooldown() else ""
            lines.append(
                f"  {name}: 调用 {m.calls}, 成功率 {m.success_rate:.0%}, "
                f"均延迟 {m.avg_latency:.2f}s, 连续失败 {m.consecutive_failures}{cd}"
            )
        return "\n".join(lines)
