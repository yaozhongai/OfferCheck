"""
Reflexion 长期记忆管理器

实现论文 Reflexion: Language Agents with Verbal Reinforcement Learning 中
的有界滑动窗口记忆机制。支持 FIFO 淘汰、JSON 持久化、记忆检索与注入。

设计原则:
- 有界窗口：max_size = 3（论文推荐 1~3 条）
- FIFO 淘汰：超出容量时淘汰最早的反思
- 持久化可选：支持会话内存储和跨会话文件持久化
- 最小侵入：独立模块，不依赖 react_agent 内部实现
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from nexa_agent.logger import get_logger

logger = get_logger("reflexion_memory")


# ==========================================================================
# 数据结构
# ==========================================================================

@dataclass
class ReflectionEntry:
    """单条反思记忆

    Attributes:
        reflection: 反思文本（2-3 句，含错误定位 + 根因分析 + 纠正策略）
        task: 关联的原始任务描述
        trial_number: 第几次尝试产生的
        timestamp: 生成时间（ISO 8601）
        eval_feedback: 评估器反馈（失败原因）
        trajectory_summary: 轨迹摘要（可选，用于调试追溯）
        lessons: 提炼的标准化教训列表（P0 教训提取，用于晋升追踪）
    """
    reflection: str
    task: str
    trial_number: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    eval_feedback: str = ""
    trajectory_summary: str = ""
    lessons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "reflection": self.reflection,
            "task": self.task,
            "trial_number": self.trial_number,
            "timestamp": self.timestamp,
            "eval_feedback": self.eval_feedback,
            "trajectory_summary": self.trajectory_summary,
            "lessons": self.lessons,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReflectionEntry":
        return cls(
            reflection=data.get("reflection", ""),
            task=data.get("task", ""),
            trial_number=data.get("trial_number", 0),
            timestamp=data.get("timestamp", ""),
            eval_feedback=data.get("eval_feedback", ""),
            trajectory_summary=data.get("trajectory_summary", ""),
            lessons=data.get("lessons", []),
        )


# ==========================================================================
# 记忆管理器
# ==========================================================================

class ReflexionMemory:
    """Reflexion 长期记忆管理器

    有界滑动窗口，FIFO 淘汰策略，可选 JSON 持久化。

    Usage::

        memory = ReflexionMemory(max_size=3)
        memory.add(ReflectionEntry(
            reflection="我在第3步...下次应先搜索再计算。",
            task="2024年全球GDP排名",
            trial_number=1,
            eval_feedback="答案数据来源不可靠",
        ))
        memories = memory.get_memories_for_prompt()  # → list[str]
    """

    def __init__(
        self,
        max_size: int = 3,
        persist_path: Optional[Path] = None,
        eviction_strategy: str = "fifo",
    ):
        """初始化记忆管理器

        Args:
            max_size: 记忆池最大容量（论文推荐 1~3）
            persist_path: JSON 持久化路径，为 None 则仅会话内存储
            eviction_strategy: 淘汰策略，目前仅支持 "fifo"
        """
        if eviction_strategy not in ("fifo",):
            raise ValueError(f"不支持的淘汰策略: {eviction_strategy}，目前仅支持 'fifo'")

        self.max_size = max_size
        self.persist_path = persist_path
        self.eviction_strategy = eviction_strategy
        self._buffer: list[ReflectionEntry] = []

        # 从文件恢复
        if persist_path and persist_path.exists():
            self._load()
            logger.info("从文件加载 %d 条记忆: %s", len(self._buffer), persist_path)
        else:
            logger.info("ReflexionMemory 初始化 max_size=%d strategy=%s persist=%s",
                         max_size, eviction_strategy,
                         str(persist_path) if persist_path else "无")

    # ── 核心操作 ──

    def add(self, entry: ReflectionEntry) -> None:
        """添加一条反思记忆，超限时按策略淘汰

        Args:
            entry: 反思条目
        """
        self._buffer.append(entry)

        evicted = 0
        while len(self._buffer) > self.max_size:
            if self.eviction_strategy == "fifo":
                removed = self._buffer.pop(0)
                logger.debug("FIFO 淘汰: trial=%d task=%s", removed.trial_number, removed.task[:50])
                evicted += 1

        if evicted > 0:
            logger.info("添加记忆后淘汰 %d 条，当前 %d/%d", evicted, len(self._buffer), self.max_size)

        if self.persist_path:
            self._save()

    def get_memories_for_prompt(self) -> list[str]:
        """返回用于注入 Prompt 的反思文本列表

        Returns:
            反思文本列表，可直接传给 react_loop 的 long_term_memory 参数
        """
        return [entry.reflection for entry in self._buffer]

    def get_all_entries(self) -> list[ReflectionEntry]:
        """返回所有记忆条目（用于调试和统计）"""
        return list(self._buffer)

    def clear(self) -> None:
        """清空所有记忆"""
        count = len(self._buffer)
        self._buffer.clear()
        logger.info("清空 %d 条记忆", count)
        if self.persist_path:
            self._save()

    def size(self) -> int:
        """当前记忆条数"""
        return len(self._buffer)

    def is_empty(self) -> bool:
        """是否为空"""
        return len(self._buffer) == 0

    # ── 质量过滤 ──

    def add_with_quality_check(
        self, entry: ReflectionEntry,
        min_length: int = 20,
        max_length: int = 500,
    ) -> bool:
        """带质量检查的添加

        过滤掉过短、过长或与已有记忆高度重复的反思。

        Args:
            entry: 反思条目
            min_length: 反思最短字符数
            max_length: 反思最长字符数

        Returns:
            True 如果添加成功，False 如果被过滤
        """
        reflection = entry.reflection.strip()

        # 长度过滤
        if len(reflection) < min_length:
            logger.warning("反思过短 (%d < %d)，跳过: %s", len(reflection), min_length, reflection[:50])
            return False
        if len(reflection) > max_length:
            logger.warning("反思过长 (%d > %d)，跳过: %s...", len(reflection), max_length, reflection[:50])
            return False

        # 去重检测：与已有记忆的简单重叠度检查
        for existing in self._buffer:
            overlap = _jaccard_similarity(reflection, existing.reflection)
            if overlap > 0.8:
                logger.info("反思与已有记忆高度重复 (overlap=%.2f)，跳过", overlap)
                return False

        self.add(entry)
        return True

    # ── 持久化 ──

    def _save(self) -> None:
        """持久化到 JSON 文件"""
        if not self.persist_path:
            return
        try:
            data = [entry.to_dict() for entry in self._buffer]
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.persist_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.debug("记忆已持久化 %d 条 → %s", len(self._buffer), self.persist_path)
        except Exception as exc:
            logger.error("记忆持久化失败: %s", exc)

    def _load(self) -> None:
        """从 JSON 文件加载"""
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            data = json.loads(self.persist_path.read_text(encoding="utf-8"))
            self._buffer = [ReflectionEntry.from_dict(item) for item in data]
        except Exception as exc:
            logger.error("记忆加载失败: %s，使用空记忆", exc)
            self._buffer = []


# ==========================================================================
# 辅助函数
# ==========================================================================

def _jaccard_similarity(text1: str, text2: str) -> float:
    """计算两段文本的简单 Jaccard 相似度（基于字符 3-gram）

    Args:
        text1, text2: 待比较文本

    Returns:
        相似度 [0, 1]
    """
    def ngrams(text: str, n: int = 3) -> set:
        chars = text.replace(" ", "")
        if len(chars) < n:
            return {chars}
        return {chars[i:i + n] for i in range(len(chars) - n + 1)}

    set1 = ngrams(text1)
    set2 = ngrams(text2)
    if not set1 or not set2:
        return 0.0
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0.0
