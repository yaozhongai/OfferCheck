# Agent Trace Schema 设计文档

> 版本：V0.3 | 日期：2026-06-10 | 已实现：STM 上下文 + LTM 检索 + Memory Write 事件通过 Trace 展示
> 目标：为 Nexa Agent 的执行轨迹可视化、SSE 流式推送、节点调试和后续 ReAct 轨迹承载提供统一 Schema
> 原则：AgentState 负责状态协议，Agent Trace 负责过程轨迹；两者边界清晰，不互相替代

---

## 1. 模块概述

### 1.1 做什么

`Agent Trace` 用于记录一次请求在 LangGraph pipeline 中的执行过程。

它主要负责：

* 记录每个 LangGraph 节点的开始、完成、失败；
* 记录路由决策、模型调用、检索、校验、记忆写入等关键事件；
* 为前端执行时间线、节点状态卡片、运行指标面板提供结构化数据；
* 为 SSE 流式推送提供统一事件格式；
* 为后续 ReAct 子图中的 Action / Observation 轨迹提供承载基础；
* 为调试、回放、失败定位提供可持久化的运行轨迹。

### 1.2 不做什么

`Agent Trace` 不负责：

* 不决定 LangGraph 节点流转；
* 不修改 `AgentState`；
* 不替代 `AgentState.action_trace`；
* 不保存完整 Chain-of-Thought；
* 不保存完整 prompt；
* 不保存完整模型原始输出；
* 不直接执行业务逻辑；
* 不作为长期记忆或反思记录使用。

### 1.3 与 AgentState 的关系

`AgentState` 是 LangGraph 节点之间共享的状态协议。

`Agent Trace` 是运行过程中的事件流协议。

二者关系如下：

```text
LangGraph Node 执行
        ↓
返回 AgentState partial update
        ↓
同时 emit AgentTraceEvent
        ↓
SSE 推送给前端
        ↓
前端渲染 Timeline / Node Card / Debug Panel
```

`AgentState.action_trace` 只保留面向用户和最终响应的轻量动作摘要。

`AgentTraceEvent` 保留更细粒度的执行事件，用于可视化、调试和回放。

---

## 2. 数据流

### 2.1 整体数据流

```text
FastAPI 接收请求
    ↓
create_initial_state()
    ↓
create_trace_run()
    ↓
compiled_graph.stream()
    ↓
LangGraph Node 执行
    ↓
emit AgentTraceEvent
    ↓
Trace Store 写入
    ↓
SSE 推送事件
    ↓
前端渲染执行轨迹
    ↓
respond / update_memory
    ↓
complete_trace_run()
```

### 2.2 当前 pipeline 节点流

```text
normalize_input
     ↓
load_short_term_context
     ↓
route_task
     │
     ├── VISION_DIRECT: vision_direct → validate_direct
     ├── VISION_SCHEMA: vision_schema → validate_schema
     ├── RAG_QA + 图片: vision_perceive → retrieve → reason → verify
     ├── RAG_QA: retrieve → reason
     ├── TOOL_ACT: tool_act_placeholder
     └── fallback
                                                ↓
                                             respond
                                                ↓
                                          update_memory
                                                ↓
                                               END
```

> **V0 限制**：Trace 事件在 `graph.invoke()` 完成后统一批量发射（非流式）。
> SSE 订阅者在 invoke 完成前无法收到中间事件。
> 实时流式 Trace 将在后续通过 `graph.astream_events()` 实现。

### 2.3 Trace 事件产生位置

| pipeline 阶段 | 产生的 Trace 事件             |
| ----------- | ------------------------ |
| 请求开始        | `trace_started`          |
| 节点开始        | `node_started`           |
| 节点完成        | `node_completed`         |
| 节点失败        | `node_failed`            |
| 路由完成        | `route_decided`          |
| 模型调用开始      | `model_call_started`     |
| 模型调用完成      | `model_call_completed`   |
| 模型调用失败      | `model_call_failed`      |
| 检索完成        | `retrieval_completed`    |
| 校验完成        | `validation_completed`   |
| 记忆写入完成      | `memory_write_completed` |
| 兜底触发        | `fallback_triggered`     |
| 请求完成        | `trace_completed`        |
| 请求失败        | `trace_failed`           |

---

## 3. 表 / 集合设计

V0 阶段建议使用两张表：

```text
agent_trace_runs
agent_trace_events
```

不建议单独存储 `timeline_items`，前端时间线可以由 `agent_trace_events` 派生。

---

### 3.1 `agent_trace_runs` 表

用于记录一次完整请求的 Trace 总览。

| 字段名                    | 类型                 | 必填 | 设计理由                                                      |
| ---------------------- | ------------------ | -- | --------------------------------------------------------- |
| `trace_id`             | String / UUID      | ✅  | 一次执行轨迹的唯一 ID                                              |
| `request_id`           | String / UUID      | ✅  | 对应 AgentState.request_id                                  |
| `session_id`           | String / UUID      | ✅  | 对应多轮会话 ID                                                 |
| `user_id`              | String nullable    |    | 用户 ID，后续用于权限隔离                                            |
| `route_type`           | String nullable    |    | 最终路由类型，如 `VISION_DIRECT` / `RAG_QA`                       |
| `status`               | Enum               | ✅  | Trace 状态：`running` / `completed` / `failed` / `cancelled` |
| `current_node`         | String nullable    |    | 当前执行到的节点，用于前端展示                                           |
| `started_at`           | Timestamp          | ✅  | Trace 开始时间                                                |
| `finished_at`          | Timestamp nullable |    | Trace 结束时间                                                |
| `duration_ms`          | Integer nullable   |    | 总耗时                                                       |
| `event_count`          | Integer            | ✅  | 事件数量统计                                                    |
| `error_count`          | Integer            | ✅  | 错误数量统计                                                    |
| `model_call_count`     | Integer            | ✅  | 模型调用次数                                                    |
| `tool_call_count`      | Integer            | ✅  | 工具调用次数，当前可为 0                                             |
| `final_answer_summary` | Text nullable      |    | 最终回答摘要，不保存完整大文本                                           |
| `created_at`           | Timestamp          | ✅  | 创建时间                                                      |
| `updated_at`           | Timestamp          | ✅  | 更新时间                                                      |

#### 为什么需要 `trace_id`

`request_id` 是 AgentState 的单次请求 ID，`trace_id` 是可观测链路 ID。

V0 阶段可以一对一：

```text
request_id = trace_id
```

但 Schema 层面仍建议保留二者，方便未来支持：

```text
一次请求 → 多个子图 trace
一次主 trace → 多个 sub trace
```

---

### 3.2 `agent_trace_events` 表

用于记录一次请求中的所有执行事件。

| 字段名              | 类型               | 必填 | 设计理由                           |
| ---------------- | ---------------- | -- | ------------------------------ |
| `event_id`       | String / UUID    | ✅  | 单条事件唯一 ID                      |
| `trace_id`       | String / UUID    | ✅  | 关联 `agent_trace_runs.trace_id` |
| `request_id`     | String / UUID    | ✅  | 对应 AgentState.request_id       |
| `session_id`     | String / UUID    | ✅  | 对应会话 ID                        |
| `seq`            | Integer          | ✅  | 单个 trace 内递增序号，用于稳定排序          |
| `event_type`     | Enum             | ✅  | 事件类型                           |
| `event_status`   | Enum             | ✅  | 事件状态                           |
| `event_level`    | Enum             | ✅  | 事件级别                           |
| `visibility`     | Enum             | ✅  | 前端可见级别                         |
| `node_name`      | String nullable  |    | 当前节点名                          |
| `span_id`        | String nullable  |    | 当前事件 span ID                   |
| `parent_span_id` | String nullable  |    | 父 span ID，用于树状轨迹               |
| `title`          | String           | ✅  | 面向前端展示的短标题                     |
| `message`        | Text nullable    |    | 面向前端展示的说明                      |
| `input_summary`  | Text nullable    |    | 输入摘要，不保存完整输入大文本                |
| `output_summary` | Text nullable    |    | 输出摘要，不保存完整原始结果                 |
| `payload`        | JSON nullable    |    | 结构化事件详情                        |
| `duration_ms`    | Integer nullable |    | 当前事件耗时                         |
| `error_type`     | String nullable  |    | 错误类型                           |
| `error_message`  | Text nullable    |    | 错误摘要                           |
| `created_at`     | Timestamp        | ✅  | 事件创建时间                         |

#### 为什么需要 `seq`

前端执行轨迹必须稳定有序。

不能只依赖 `created_at` 排序，因为同一毫秒内可能产生多条事件。

`seq` 必须在同一个 `trace_id` 内单调递增。

---

## 4. Schema 定义

### 4.1 Trace 状态枚举

```python
from enum import Enum


class TraceStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

---

### 4.2 Trace 事件类型

```python
class TraceEventType(str, Enum):
    TRACE_STARTED = "trace_started"
    TRACE_COMPLETED = "trace_completed"
    TRACE_FAILED = "trace_failed"

    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    NODE_SKIPPED = "node_skipped"

    ROUTE_DECIDED = "route_decided"

    MODEL_CALL_STARTED = "model_call_started"
    MODEL_CALL_COMPLETED = "model_call_completed"
    MODEL_CALL_FAILED = "model_call_failed"

    RETRIEVAL_STARTED = "retrieval_started"
    RETRIEVAL_COMPLETED = "retrieval_completed"
    RETRIEVAL_FAILED = "retrieval_failed"

    VALIDATION_STARTED = "validation_started"
    VALIDATION_COMPLETED = "validation_completed"
    VALIDATION_FAILED = "validation_failed"

    MEMORY_READ_STARTED = "memory_read_started"
    MEMORY_READ_COMPLETED = "memory_read_completed"
    MEMORY_WRITE_STARTED = "memory_write_started"
    MEMORY_WRITE_COMPLETED = "memory_write_completed"
    MEMORY_WRITE_SKIPPED = "memory_write_skipped"
    MEMORY_WRITE_FAILED = "memory_write_failed"

    FALLBACK_TRIGGERED = "fallback_triggered"

    HUMAN_CONFIRM_REQUIRED = "human_confirm_required"
    HUMAN_CONFIRM_COMPLETED = "human_confirm_completed"

    TOOL_CALL_PLANNED = "tool_call_planned"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    TOOL_CALL_FAILED = "tool_call_failed"
```

---

### 4.3 事件状态枚举

```python
class TraceEventStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING = "waiting"
```

---

### 4.4 事件级别枚举

```python
class TraceEventLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
```

---

### 4.5 可见性枚举

```python
class TraceVisibility(str, Enum):
    USER = "user"
    DEV = "dev"
    DEBUG = "debug"
```

说明：

| 可见性     | 用途                              |
| ------- | ------------------------------- |
| `user`  | 普通前端可见，例如节点进度、检索完成、回答生成         |
| `dev`   | 开发调试可见，例如模型调用摘要、validation 细节   |
| `debug` | 深度调试可见，例如截断后的 payload、错误 detail |

---

### 4.6 当前 pipeline 节点枚举

```python
class TraceNodeName(str, Enum):
    NORMALIZE_INPUT = "normalize_input"
    LOAD_SHORT_TERM_CONTEXT = "load_short_term_context"
    ROUTE_TASK = "route_task"

    VISION_DIRECT = "vision_direct"
    VISION_SCHEMA = "vision_schema"
    VISION_PERCEIVE = "vision_perceive"

    VALIDATE_DIRECT = "validate_direct"
    VALIDATE_SCHEMA = "validate_schema"

    RETRIEVE = "retrieve"
    REASON = "reason"
    VERIFY = "verify"

    RESPOND = "respond"
    UPDATE_MEMORY = "update_memory"
    FALLBACK = "fallback"

    TOOL_ACT_PLACEHOLDER = "tool_act_placeholder"
```

---

### 4.7 `AgentTraceRun`

```python
from pydantic import BaseModel, Field
from datetime import datetime


class AgentTraceRun(BaseModel):
    trace_id: str
    request_id: str
    session_id: str
    user_id: str | None = None

    route_type: str | None = None
    status: TraceStatus = TraceStatus.RUNNING
    current_node: str | None = None

    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None
    duration_ms: int | None = None

    event_count: int = 0
    error_count: int = 0
    model_call_count: int = 0
    tool_call_count: int = 0

    final_answer_summary: str | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
```

---

### 4.8 `AgentTraceEvent`

```python
from typing import Any


class AgentTraceEvent(BaseModel):
    event_id: str
    trace_id: str
    request_id: str
    session_id: str

    seq: int
    event_type: TraceEventType
    event_status: TraceEventStatus
    event_level: TraceEventLevel = TraceEventLevel.INFO
    visibility: TraceVisibility = TraceVisibility.USER

    node_name: str | None = None

    span_id: str | None = None
    parent_span_id: str | None = None

    title: str
    message: str | None = None

    input_summary: str | None = None
    output_summary: str | None = None

    payload: dict[str, Any] = Field(default_factory=dict)

    duration_ms: int | None = None

    error_type: str | None = None
    error_message: str | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
```

---

### 4.9 `AgentTimelineItem`

`AgentTimelineItem` 是前端展示 DTO，不建议落库。

它由 `AgentTraceEvent` 聚合得到。

```python
class AgentTimelineItem(BaseModel):
    item_id: str
    trace_id: str

    node_name: str

    title: str
    description: str | None = None

    status: TraceEventStatus
    level: TraceEventLevel = TraceEventLevel.INFO

    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int | None = None

    events: list[AgentTraceEvent] = Field(default_factory=list)

    expandable: bool = False
    default_open: bool = False
```

---

## 5. 关键 Payload 约定

`AgentTraceEvent.payload` 是 JSON 字段。

为了避免 Vibe Coding 时乱塞字段，以下事件类型建议使用固定 payload 结构。

---

### 5.1 路由事件 Payload

适用于：

```text
route_decided
```

```python
class RouteTracePayload(BaseModel):
    route_type: str
    confidence: float | None = None
    reason: str | None = None
    matched_rules: list[str] = Field(default_factory=list)

    need_retrieve: bool = False
    need_reason: bool = False
    need_verify: bool = False
    need_memory_write: bool = False

    risk_level: str | None = None
```

示例：

```json
{
  "route_type": "RAG_QA",
  "confidence": 0.86,
  "reason": "用户问题需要结合记忆与上下文进行回答",
  "matched_rules": ["has_text", "need_context"],
  "need_retrieve": true,
  "need_reason": true,
  "need_verify": true,
  "need_memory_write": false,
  "risk_level": "low"
}
```

---

### 5.2 模型调用事件 Payload

适用于：

```text
model_call_started
model_call_completed
model_call_failed
```

```python
class ModelCallTracePayload(BaseModel):
    provider: str | None = None
    model_name: str
    node_name: str

    purpose: str

    input_summary: str | None = None
    output_summary: str | None = None

    latency_ms: int | None = None
    success: bool = True
    error_message: str | None = None

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
```

约束：

* 不保存完整 prompt；
* 不保存完整 raw output；
* 只保存摘要；
* token 信息没有就填 `None`。

---

### 5.3 检索事件 Payload

适用于：

```text
retrieval_started
retrieval_completed
retrieval_failed
memory_read_started
memory_read_completed
```

```python
class RetrievalTracePayload(BaseModel):
    query_summary: str | None = None

    source_type: str
    source_name: str | None = None

    result_count: int = 0
    top_scores: list[float] = Field(default_factory=list)

    used_short_term_memory: bool = False
    used_long_term_memory: bool = False
    used_document_context: bool = False
```

---

### 5.4 校验事件 Payload

适用于：

```text
validation_started
validation_completed
validation_failed
```

```python
class ValidationTracePayload(BaseModel):
    validator_name: str
    passed: bool

    score: float | None = None
    issues: list[str] = Field(default_factory=list)

    revised: bool = False
    revised_answer_summary: str | None = None
```

说明：

当前 `verify` 节点属于回答校验与修正，不单独作为 Reflection Log。

---

### 5.5 记忆写入事件 Payload

适用于：

```text
memory_write_started
memory_write_completed
memory_write_skipped
memory_write_failed
```

```python
class MemoryWriteTracePayload(BaseModel):
    need_memory_write: bool

    target: str | None = None
    written: bool = False
    skipped_reason: str | None = None

    memory_item_count: int = 0
```

---

### 5.6 错误事件 Payload

适用于：

```text
node_failed
trace_failed
model_call_failed
retrieval_failed
validation_failed
memory_write_failed
tool_call_failed
```

```python
class ErrorTracePayload(BaseModel):
    error_type: str
    message: str

    node_name: str | None = None
    recoverable: bool = True
    retryable: bool = False

    detail: dict[str, Any] = Field(default_factory=dict)
```

约束：

* `message` 只能保存错误摘要；
* 不保存完整堆栈；
* 完整堆栈如需保存，只能在 debug 模式下写入本地日志，不能通过 SSE 对外推送。

---

### 5.7 工具调用事件 Payload

适用于后续 ReAct 子图。

```python
class ToolCallTracePayload(BaseModel):
    tool_call_id: str
    tool_name: str

    action_summary: str | None = None
    input_summary: str | None = None
    output_summary: str | None = None

    status: str
    risk_level: str | None = None

    require_human_confirm: bool = False
    confirm_reason: str | None = None

    latency_ms: int | None = None
    error_message: str | None = None
```

说明：

`Agent Trace Schema` 是 ReAct Runtime 的前置基础。

当前 `TOOL_ACT` 仍是占位节点，但后续 ReAct 子图中的工具调用事件必须复用该结构。

---

## 6. 当前 pipeline 节点事件映射

| 节点名 | 主要事件 | 关键 payload |
| ----- | ----- | ----- |
| `normalize_input` | `node_started` / `node_completed` | 输入类型、是否有图片、是否有文件 |
| `load_short_term_context` | `node_started` / `memory_read_completed` / `node_completed` | `RetrievalTracePayload`（读 STM），`output_summary` 含上下文摘要，后端日志同步输出 |
| `route_task` | `node_started` / `route_decided` / `node_completed` | `RouteTracePayload` |
| `vision_direct` | `node_started` / `model_call_started` / `model_call_completed` / `node_completed` | `ModelCallTracePayload` |
| `vision_schema` | `node_started` / `model_call_started` / `model_call_completed` / `node_completed` | `ModelCallTracePayload` |
| `vision_perceive` | `node_started` / `model_call_started` / `model_call_completed` / `node_completed` | `ModelCallTracePayload` |
| `validate_direct` | `node_started` / `validation_completed` / `node_completed` | `ValidationTracePayload` |
| `validate_schema` | `node_started` / `validation_completed` / `node_completed` | `ValidationTracePayload` |
| `retrieve` | `node_started` / `retrieval_completed` / `node_completed` | `RetrievalTracePayload`（读 LTM: MemoryItem → EvidenceItem → retrieved_context） |
| `reason` | `node_started` / `model_call_started` / `model_call_completed` / `node_completed` | `ModelCallTracePayload` |
| `verify` | `node_started` / `model_call_started` / `validation_completed` / `node_completed` | `ValidationTracePayload` |
| `respond` | `node_started` / `node_completed` | final_answer 摘要 |
| `update_memory` | `node_started` / `memory_write_completed` 或 `memory_write_skipped` / `node_completed` | `MemoryWriteTracePayload` |
| `fallback` | `fallback_triggered` / `node_completed` | fallback 原因 |
| `tool_act_placeholder` | `node_started` / `node_completed` | V1 占位说明 |

---

## 7. API 接口设计

### 7.1 创建 Trace

```python
def create_trace_run(
    request_id: str,
    session_id: str,
    user_id: str | None = None,
) -> AgentTraceRun:
    """
    创建一次请求的 TraceRun。
    每个 request_id 默认对应一个 trace_id。
    """
```

---

### 7.2 写入事件

```python
def emit_trace_event(
    trace_id: str,
    request_id: str,
    session_id: str,
    event_type: TraceEventType,
    title: str,
    node_name: str | None = None,
    event_status: TraceEventStatus = TraceEventStatus.SUCCESS,
    event_level: TraceEventLevel = TraceEventLevel.INFO,
    visibility: TraceVisibility = TraceVisibility.USER,
    message: str | None = None,
    input_summary: str | None = None,
    output_summary: str | None = None,
    payload: dict | None = None,
    duration_ms: int | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> AgentTraceEvent:
    """
    写入一条 Trace 事件。
    自动生成 event_id。
    自动递增 seq。
    自动更新 agent_trace_runs 中的 event_count、current_node、updated_at。
    """
```

---

### 7.3 完成 Trace

```python
def complete_trace_run(
    trace_id: str,
    final_answer_summary: str | None = None,
) -> AgentTraceRun:
    """
    将 TraceRun 状态更新为 completed。
    自动计算 duration_ms。
    自动写入 trace_completed 事件。
    """
```

---

### 7.4 标记 Trace 失败

```python
def fail_trace_run(
    trace_id: str,
    error_type: str,
    error_message: str,
    node_name: str | None = None,
    recoverable: bool = True,
) -> AgentTraceRun:
    """
    将 TraceRun 状态更新为 failed。
    自动写入 trace_failed 事件。
    """
```

---

### 7.5 查询事件

```python
def get_trace_events(
    trace_id: str,
    visibility: TraceVisibility | None = None,
    after_seq: int | None = None,
    limit: int = 200,
) -> list[AgentTraceEvent]:
    """
    查询某次请求的 Trace 事件。
    after_seq 用于 SSE 断线重连后续传。
    visibility 用于控制 user / dev / debug 可见范围。
    """
```

---

### 7.6 构建前端时间线

```python
def build_timeline_items(
    trace_id: str,
    visibility: TraceVisibility = TraceVisibility.USER,
) -> list[AgentTimelineItem]:
    """
    将 AgentTraceEvent 聚合为前端 Timeline Item。
    不直接读取 AgentState。
    """
```

---

### 7.7 SSE 订阅接口

```python
async def subscribe_trace_events(
    trace_id: str,
    visibility: TraceVisibility = TraceVisibility.USER,
    after_seq: int | None = None,
):
    """
    订阅某个 trace_id 的事件流。
    用于 FastAPI SSE 推送。
    支持 after_seq，便于前端断线重连。
    """
```

---

## 8. 边界与约束

| 边界情况                               | 处理方式                                                    |
| ---------------------------------- | ------------------------------------------------------- |
| 同一 trace 并发写入事件                    | `seq` 必须基于 trace_id 单调递增，不能只依赖时间戳                       |
| 节点执行失败                             | 必须写入 `node_failed`，并同步更新 `agent_trace_runs.error_count` |
| 模型调用失败                             | 写入 `model_call_failed`，payload 只保存错误摘要                  |
| Trace 存储失败                         | 不应阻断主链路，可降级到内存队列或标准日志                                   |
| SSE 推送失败                           | 不影响 LangGraph 执行，前端可通过 `after_seq` 拉取补偿                 |
| payload 过大                         | 超过 10KB 必须截断，只保留摘要和引用                                   |
| prompt 过长                          | 不保存完整 prompt，只保存 `input_summary`                        |
| 模型输出过长                             | 不保存完整 raw output，只保存 `output_summary`                   |
| 完整 Chain-of-Thought                | 禁止保存                                                    |
| debug 信息包含敏感内容                     | 不允许通过 user / dev 可见事件返回                                 |
| `trace_completed` 时仍有运行中节点         | 视为状态异常，应写入 warning 事件                                   |
| `final_answer` 为空但 trace completed | 写入 warning，说明 respond 节点兜底结果异常                          |
| TOOL_ACT 当前未实现                     | 只记录 `tool_act_placeholder`，不伪造工具调用事件                    |

---

## 9. 前端展示约定

### 9.1 Timeline 展示层级

前端建议按节点聚合展示：

```text
执行轨迹
  ├── 输入标准化
  ├── 任务路由
  ├── 视觉理解 / 结构化提取
  ├── 检索上下文
  ├── 推理生成
  ├── 校验修正
  ├── 兜底响应
  └── 记忆写入
```

### 9.2 默认展示信息

普通用户默认只展示：

* 当前执行到哪个节点；
* 每个节点是否成功；
* 简短说明；
* 耗时；
* 是否触发兜底或校验修正。

### 9.3 开发者展示信息

开发者模式可展示：

* 模型调用摘要；
* 检索结果数量；
* 校验分数；
* validation issues；
* memory write 是否跳过；
* error_type 和 error_message。

### 9.4 Debug 展示信息

Debug 模式可展示：

* 截断后的 payload；
* node input summary；
* node output summary；
* span_id / parent_span_id；
* after_seq 补偿信息。

---

## 10. V0 最小实现范围

V0 阶段必须实现：

```text
AgentTraceRun
AgentTraceEvent
TraceStatus
TraceEventType
TraceEventStatus
TraceEventLevel
TraceVisibility
create_trace_run()
emit_trace_event()
complete_trace_run()
fail_trace_run()
get_trace_events()
subscribe_trace_events()
```

V0 阶段必须支持的事件：

```text
trace_started
trace_completed
trace_failed

node_started
node_completed
node_failed

route_decided

model_call_started
model_call_completed
model_call_failed

retrieval_completed
validation_completed

memory_write_completed
memory_write_skipped

fallback_triggered
```

V0 阶段暂不实现：

```text
完整 trace 回放
复杂 span 树
跨请求 trace 聚合
长期存储归档
ReAct 工具调用细节
人工确认细节
```

---

## 11. 与后续 ReAct Runtime 的关系

`Agent Trace Schema` 是 ReAct Runtime 的前置基础。

后续 ReAct 子图需要复用：

```text
trace_id
event_id
seq
span_id
parent_span_id
node_name
tool_call_id
event_type
payload
```

ReAct 子图事件建议按以下结构扩展：

```text
react_decide
  ↓
tool_call_planned
  ↓
tool_call_started
  ↓
tool_call_completed / tool_call_failed
  ↓
observation_recorded
  ↓
react_continue / react_finish
```

但这些不在当前 V0 强制实现范围内。

---

## 12. Vibe Coding 注意事项

⚠️ 特别注意：

1. 不要把 `AgentTraceEvent` 列表直接塞进 `AgentState`，Trace 由独立 event store 管理。
2. `AgentState.action_trace` 是轻量动作摘要，不等于完整 Agent Trace。
3. 所有事件必须有 `seq`，并且在同一个 `trace_id` 内单调递增。
4. 所有 LangGraph 节点至少要 emit `node_started` 和 `node_completed`；失败时必须 emit `node_failed`。
5. 所有模型调用只能保存摘要，不允许保存完整 prompt 和完整 raw output。
6. 不保存模型内部完整 Chain-of-Thought；必须保存可审计的决策摘要、Action、Observation、工具调用、校验结果和错误信息。
7. SSE 推送失败不能影响主流程执行。
8. payload 超过 10KB 必须截断。
9. 前端 Timeline 必须由 `AgentTraceEvent` 派生，不要直接解析 AgentState。
10. 后续 ReAct 子图必须复用当前 Trace 结构，不要重新设计一套执行日志协议。

---

## 13. 验收标准

1. 任意一次请求都能创建一条 `AgentTraceRun`；
2. 每个 pipeline 节点至少产生开始和完成事件；
3. 路由节点必须产生 `route_decided` 事件；
4. 模型调用节点必须产生 `model_call_started` 和 `model_call_completed` 事件；
5. 检索节点必须产生 `retrieval_completed` 或 `memory_read_completed` 事件；
6. 校验节点必须产生 `validation_completed` 事件；
7. 记忆节点必须产生 `memory_write_completed` 或 `memory_write_skipped` 事件；
8. 失败时必须产生 `node_failed` 或 `trace_failed` 事件；
9. 前端可以通过 SSE 按 `seq` 顺序展示事件；
10. `get_trace_events(after_seq=...)` 可以支持断线续传；
11. 对外返回不包含完整 prompt、完整 raw output、完整 Chain-of-Thought、敏感 debug 信息。

---

## 14. 推荐文件位置

```text
app/
  agent/
    state.py
    graph.py
    nodes/

  trace/
    schema.py
    store.py
    service.py
    sse.py

  api/
    routes/
      agent.py
      trace.py
```

其中：

| 文件                    | 职责                                       |
| --------------------- | ---------------------------------------- |
| `trace/schema.py`     | 定义 Trace 相关枚举和 Pydantic Schema           |
| `trace/store.py`      | 负责事件存储，V0 可用 SQLite / 内存                 |
| `trace/service.py`    | 封装 create / emit / complete / fail 等业务接口 |
| `trace/sse.py`        | 提供 SSE 事件订阅                              |
| `api/routes/trace.py` | 提供 Trace 查询和前端调试接口                       |

---

## 15. 最终原则

```text
AgentState 负责状态协议
Agent Trace 负责执行轨迹
ActionTrace 负责用户可读动作摘要
SSE 负责流式推送
Timeline 负责前端展示
ReAct Runtime 复用 Trace 事件体系
```

`Agent Trace` 不是普通日志系统，也不是新的状态机。

它是 Nexa Agent 在 LangGraph 执行过程中的结构化轨迹协议。
