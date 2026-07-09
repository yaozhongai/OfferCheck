"""引擎轨迹的同源持久化（评审 3.3：双轨合一）。

**SSE 与持久化同源**：`run_stage/stream` 的 `_event_stream` 每发一条事件，就交给
`TraceRecorder.record()`（内存缓冲，O(1)）；流结束时 `finalize()` 一次性把整条 trace
以 **OTel-GenAI 对齐**的 JSON（spans 数组 + resource 属性）落到
`data/traces/<trace_id>.json`。落库的就是发给浏览器的**同一批事件**——单一 typed
schema（见 `nexa_agent/trace/events.py`）。

设计红线：**绝不拖慢/打断 live SSE 路径**。
  - record / finalize 全 **best-effort**（try/except 吞异常，只记日志）；
  - 事件仅在内存缓冲、流结束时**单次落盘**（不在热路径同步写盘/写库）；
  - 缓冲条数上限，防异常长 run 撑爆内存。

此文件取代 LangGraph 时代的 `server/trace_store/{service,store,sse}.py` +
`nexa_agent/trace/schema.py`（那套 typed schema 与现役事件不符、从未接主链路），
作为唯一的引擎轨迹持久化。
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Optional

from nexa_agent.trace.events import (
    to_otel_attributes, resource_attributes, is_known_event, EngineEventType,
)
from nexa_agent.logger import get_logger

logger = get_logger("trace_recorder")

# 单条 trace 缓冲事件上限（防异常长 run 撑爆内存；超限丢弃并计数）
_MAX_BUFFERED_EVENTS = 2000
# 观察/答案等大字段落盘截断（trace 是可观测产物，不需全文）
_FIELD_CAP = 2000


def new_trace_id() -> str:
    return uuid.uuid4().hex


def default_trace_dir() -> str:
    """trace 落盘目录：紧邻 DB（data/ 已 gitignore），可 NEXA_TRACE_DIR 覆盖。"""
    env = os.environ.get("NEXA_TRACE_DIR")
    if env:
        return env
    try:
        from server.config import get_config
        return os.path.join(os.path.dirname(get_config().db_path), "traces")
    except Exception:  # noqa: BLE001 — 配置不可用时退回临时目录，绝不抛
        return os.path.join(os.path.dirname(__file__), "..", "..", "data", "traces")


class TraceRecorder:
    """一次 run 的轨迹记录器：缓冲事件 → 结束时以 OTel JSON 单次落盘。"""

    def __init__(self, trace_id: str, stage: Optional[str] = None,
                 trace_dir: Optional[str] = None):
        self.trace_id = trace_id
        self.stage = stage
        self.trace_dir = trace_dir or default_trace_dir()
        self._spans: list[dict] = []
        self._seq = 0
        self._dropped = 0
        self._started_at = time.time()
        # 从 usage 事件滚动累计（成本），从 done/final_answer 抓 verdict
        self._input_tokens = 0
        self._output_tokens = 0
        self._verdict = None

    def record(self, event: dict) -> None:
        """缓冲一条事件（best-effort，绝不抛）。__end__ 哨兵不记。"""
        try:
            if not isinstance(event, dict):
                return
            et = event.get("type")
            if et == "__end__":
                return
            if len(self._spans) >= _MAX_BUFFERED_EVENTS:
                self._dropped += 1
                return
            is_known_event(event)  # 未知类型只记 debug
            self._seq += 1
            self._spans.append({
                "seq": self._seq,
                "type": et,
                "timestamp": time.time(),
                "attributes": to_otel_attributes(event),
                "event": _slim(event),
            })
            # 滚动累计 token / 抓 verdict（供 trace 摘要）
            if et == EngineEventType.USAGE.value:
                self._input_tokens += int(event.get("prompt_tokens") or 0)
                self._output_tokens += int(event.get("completion_tokens") or 0)
            if et in (EngineEventType.DONE.value, EngineEventType.FINAL_ANSWER.value) and event.get("verdict"):
                self._verdict = event["verdict"]
        except Exception:  # noqa: BLE001 — 可观测绝不影响主流程
            logger.debug("trace record 异常（已忽略）", exc_info=True)

    def finalize(self, success: Optional[bool] = None,
                 latency_ms: Optional[float] = None) -> Optional[str]:
        """把缓冲的整条 trace 以 OTel JSON 落盘（best-effort）；返回文件路径或 None。"""
        try:
            os.makedirs(self.trace_dir, exist_ok=True)
            doc = {
                "trace_id": self.trace_id,
                "stage": self.stage,
                "resource": resource_attributes(),
                "started_at": self._started_at,
                "finished_at": time.time(),
                "duration_ms": (round(latency_ms) if latency_ms is not None
                                else round((time.time() - self._started_at) * 1000)),
                "success": success,
                "event_count": len(self._spans),
                "dropped_events": self._dropped,
                "usage": {
                    "input_tokens": self._input_tokens,
                    "output_tokens": self._output_tokens,
                    "total_tokens": self._input_tokens + self._output_tokens,
                },
                "verdict": self._verdict,
                "spans": self._spans,
            }
            path = os.path.join(self.trace_dir, f"{self.trace_id}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
            logger.info("trace 落盘 trace_id=%s events=%d tokens=%d path=%s",
                        self.trace_id, len(self._spans),
                        self._input_tokens + self._output_tokens, path)
            return path
        except Exception:  # noqa: BLE001
            logger.warning("trace 落盘失败 trace_id=%s（已忽略）", self.trace_id, exc_info=True)
            return None


def load_trace(trace_id: str, trace_dir: Optional[str] = None) -> Optional[dict]:
    """读回一条持久化的 trace（供查询 API）；不存在/损坏返回 None。"""
    path = os.path.join(trace_dir or default_trace_dir(), f"{os.path.basename(trace_id)}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    except Exception:  # noqa: BLE001
        logger.debug("trace 读取异常 trace_id=%s", trace_id, exc_info=True)
        return None


def _slim(event: dict) -> dict:
    """落盘前对大字段截断（observation / answer / text 等），trace 无需全文。"""
    out = {}
    for k, v in event.items():
        if isinstance(v, str) and len(v) > _FIELD_CAP:
            out[k] = v[:_FIELD_CAP] + f"…(+{len(v) - _FIELD_CAP})"
        else:
            out[k] = v
    return out
