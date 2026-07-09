"""nexa_agent trace — 引擎轨迹事件的单一 typed schema（评审 3.3：双轨合一）。

现役事件类型 + OpenTelemetry GenAI 属性映射见 `events`；同源持久化（落盘/查询）
属服务层，见 `server/trace_store/recorder.py`。
"""

from nexa_agent.trace.events import (  # noqa: F401
    EngineEventType, KNOWN_EVENT_TYPES,
    to_otel_attributes, resource_attributes, is_known_event,
)
