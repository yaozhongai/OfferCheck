# AgentState Schema 设计文档 V2

> 版本：V0.3 | 日期：2026-06-10 | LTM 检索结果通过 `memory_candidates` / `evidence` 注入，不新增 `long_term_memory` 字段
> 目标：面向 LangGraph 原生重构
> 原则：LangGraph 是唯一图执行引擎，AgentState 是唯一跨节点状态协议

---

## 1. 模块定位

`AgentState` 是 Nexa Agent 在一次请求执行过程中的**全局共享状态 Schema**。

它用于在 LangGraph 各节点之间传递：

* 用户输入；
* 路由结果；
* 检索结果；
* VLM / LLM / Tool 观察结果；
* ReAct 中间步骤；
* 校验结果；
* 执行轨迹；
* 错误信息；
* 最终回答。

在新的架构中，`AgentState` 不再是传统状态机里的 `StateContext`，而是 LangGraph `StateGraph` 的状态协议。

---

## 2. 核心设计原则

### 2.1 LangGraph 是唯一图执行引擎

系统中不再保留独立的：

```text
AgentStateMachine
HandlerRegistry
Handler.execute(ctx) -> next_node
StateContext dataclass
```

所有流程控制都交给：

```text
LangGraph StateGraph
LangGraph node
LangGraph edge
LangGraph conditional edge
LangGraph Command / interrupt
LangGraph subgraph
```

---

### 2.2 AgentState 只保存状态，不做业务逻辑

`AgentState` 不负责：

1. 判断任务类型；
2. 调用 VLM / LLM / Tool；
3. 生成最终回答；
4. 写入长期记忆；
5. 执行 ReAct 决策；
6. 执行人工确认；
7. 操作数据库；
8. 判断下一节点。

这些逻辑应分别放在：

```text
nodes/
routers/
services/
memory/
tools/
llm/
vlm/
storage/
```

---

### 2.3 节点只返回 partial update

LangGraph 节点应该遵循：

```python
def node(state: AgentState) -> dict:
    return {
        "some_field": new_value
    }
```

不建议写成：

```python
def node(state: AgentState) -> AgentState:
    state.some_field = new_value
    return state
```

也不建议写成：

```python
def node(ctx: StateContext) -> str:
    return "next_node"
```

---

### 2.4 路由由 conditional edge 负责

节点只负责更新状态。

例如：

```python
def route_task(state: AgentState) -> dict:
    return {
        "route_result": route_result,
        "status": RunStatus.ROUTED
    }
```

跳转逻辑单独放在 router 函数中：

```python
def route_after_task(state: AgentState) -> str:
    return state["route_result"].route_type.value
```

---

### 2.5 追加型字段必须使用 reducer

以下字段是追加型字段：

```text
messages
action_trace
observations
tool_calls
tool_results
evidence
validation_results
errors
```

它们不应该被后续节点覆盖，而应该通过 reducer 追加。

---

### 2.6 不保存完整 Chain-of-Thought

`AgentState` 允许保存：

```text
route_reason
decision_summary
observation
action_summary
verification_issue
```

不允许保存：

```text
完整 Chain-of-Thought
完整隐藏推理过程
模型内部推理草稿
```

---

## 3. 推荐文件位置

```text
app/
  agent/
    state.py
    graph.py
    routers.py
    nodes/
      normalize.py
      route.py
      vision.py
      retrieve.py
      reason.py
      verify.py
      respond.py
      memory.py
      react.py
```

`state.py` 只负责：

```text
枚举
子 Schema
AgentState TypedDict
初始状态构造函数
对外响应过滤函数
少量 patch helper
```

---

## 4. 数据流

一次请求的推荐数据流：

```text
FastAPI / CLI
  ↓
create_initial_state()
  ↓
compiled_graph.invoke() / compiled_graph.stream()
  ↓
normalize_input
  ↓
load_short_term_context
  ↓
route_task
  ↓
conditional edge
  ↓
vision_direct / vision_schema / retrieve / react_subgraph / fallback
  ↓
reason / verify / respond
  ↓
update_memory
  ↓
to_public_response()
```

Streamlit 不直接构造或执行 AgentState。
Streamlit 只通过 HTTP / SSE 调 FastAPI。

---

## 5. 枚举定义

```python
from enum import Enum


class InputType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"
    FILE = "file"


class RouteType(str, Enum):
    VISION_DIRECT = "vision_direct"
    VISION_SCHEMA = "vision_schema"
    RAG_QA = "rag_qa"
    TOOL_ACT = "tool_act"
    FALLBACK = "fallback"


class RunStatus(str, Enum):
    INIT = "init"
    NORMALIZED = "normalized"
    ROUTED = "routed"
    RUNNING = "running"
    WAITING_USER = "waiting_user"
    WAITING_HUMAN_CONFIRM = "waiting_human_confirm"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ObservationSource(str, Enum):
    USER = "user"
    VLM = "vlm"
    LLM = "llm"
    MEMORY = "memory"
    DOCUMENT = "document"
    TOOL = "tool"
    VERIFIER = "verifier"
    SYSTEM = "system"


class ToolCallStatus(str, Enum):
    PLANNED = "planned"
    VALIDATED = "validated"
    WAITING_CONFIRM = "waiting_confirm"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    REJECTED = "rejected"


class ConfirmDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"
```

---

## 6. Reducer 定义

```python
from typing import TypeVar

T = TypeVar("T")


def append_list(old: list[T] | None, new: list[T] | None) -> list[T]:
    if old is None:
        old = []
    if new is None:
        new = []
    return old + new


def merge_dict(
    old: dict | None,
    new: dict | None,
) -> dict:
    if old is None:
        old = {}
    if new is None:
        new = {}
    return {**old, **new}
```

---

## 7. 子 Schema 定义

### 7.1 输入引用

```python
from typing import Any, Literal
from pydantic import BaseModel, Field


class ImageRef(BaseModel):
    image_id: str
    path: str | None = None
    url: str | None = None
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None
    source: Literal["upload", "clipboard", "api", "local"] = "upload"


class FileRef(BaseModel):
    file_id: str
    filename: str | None = None
    path: str | None = None
    url: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    source: Literal["upload", "api", "local"] = "upload"
```

---

### 7.2 路由结果

```python
class RouteResult(BaseModel):
    route_type: RouteType
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    matched_rules: list[str] = Field(default_factory=list)

    need_retrieve: bool = False
    need_reason: bool = False
    need_verify: bool = False
    need_memory_write: bool = False

    risk_level: RiskLevel = RiskLevel.LOW
```

说明：

* `route_type` 决定主路径；
* `need_verify` 决定是否进入校验节点；
* `risk_level` 决定是否需要人工确认；
* 不再保留 `should_use_react`；
* ReAct 是 `TOOL_ACT` 分支内部机制，不是顶层 RouteType。

---

### 7.3 执行轨迹

```python
class ActionTraceItem(BaseModel):
    step: int
    node: str
    action: str
    status: StepStatus = StepStatus.PENDING

    reason: str = ""
    input_summary: str | None = None
    output_summary: str | None = None

    latency_ms: int | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    error_message: str | None = None
```

---

### 7.4 观察结果

```python
class Observation(BaseModel):
    source: ObservationSource
    content: str

    structured_data: dict[str, Any] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    source_id: str | None = None
    node: str | None = None
```

---

### 7.5 证据项

```python
class EvidenceItem(BaseModel):
    source_type: Literal[
        "image",
        "memory",
        "document",
        "tool",
        "model",
        "user",
    ]
    content: str

    source_id: str | None = None
    title: str | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

---

### 7.6 模型调用记录

```python
class ModelCallRecord(BaseModel):
    provider: str | None = None
    model_name: str
    node: str

    input_summary: str | None = None
    output_summary: str | None = None

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    latency_ms: int | None = None
    success: bool = True
    error_message: str | None = None
```

说明：

* 只保存调用摘要；
* 默认不保存完整 prompt；
* 默认不保存完整 raw output；
* debug 模式下如需保存，放入 `debug_info`，且不得直接对外返回。

---

### 7.7 工具调用记录

```python
class ToolCallRecord(BaseModel):
    tool_call_id: str
    tool_name: str
    tool_input: dict[str, Any]

    status: ToolCallStatus = ToolCallStatus.PLANNED
    risk_level: RiskLevel = RiskLevel.LOW

    require_human_confirm: bool = False
    confirm_reason: str | None = None

    tool_output: dict[str, Any] | None = None
    error_message: str | None = None
    latency_ms: int | None = None
```

---

### 7.8 校验结果

```python
class ValidationResult(BaseModel):
    validator_name: str
    passed: bool

    issues: list[str] = Field(default_factory=list)
    revised_answer: str | None = None

    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
```

---

### 7.9 人工确认请求

```python
class HumanConfirmRequest(BaseModel):
    confirm_id: str
    reason: str
    risk_level: RiskLevel

    tool_call: ToolCallRecord | None = None
    question: str | None = None

    options: list[str] = Field(default_factory=list)
```

---

### 7.10 人工确认结果

```python
class HumanConfirmResult(BaseModel):
    confirm_id: str
    decision: ConfirmDecision
    comment: str | None = None
    edited_payload: dict[str, Any] | None = None
```

---

### 7.11 错误对象

```python
class AgentError(BaseModel):
    error_type: str
    message: str

    node: str | None = None
    recoverable: bool = True
    detail: dict[str, Any] = Field(default_factory=dict)
```

---

## 8. AgentState 主 Schema

```python
from typing import Any
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """
    Nexa Agent 的 LangGraph 全局状态协议。

    注意：
    1. AgentState 是 TypedDict，不是 Pydantic 大对象；
    2. LangGraph node 读取 AgentState，并返回 partial update；
    3. 追加型字段必须通过 reducer 合并；
    4. 不允许在 AgentState 中保存完整 Chain-of-Thought；
    5. 对外返回必须经过 to_public_response() 过滤。
    """

    # 基础标识
    request_id: str
    session_id: str
    user_id: str | None

    # 输入
    user_input: str
    input_type: InputType
    image_refs: list[ImageRef]
    file_refs: list[FileRef]
    input_metadata: Annotated[dict[str, Any], merge_dict]

    # LangGraph 消息
    messages: Annotated[list[BaseMessage], add_messages]

    # 会话上下文
    history_summary: str | None
    short_term_context: list[dict[str, Any]]

    # 路由
    route_result: RouteResult

    # 运行状态
    status: RunStatus
    step_count: int
    max_steps: int

    # 检索与证据
    memory_candidates: Annotated[list[EvidenceItem], append_list]
    retrieved_context: Annotated[list[EvidenceItem], append_list]
    evidence: Annotated[list[EvidenceItem], append_list]

    # 模型 / 感知 / 工具观察
    observations: Annotated[list[Observation], append_list]
    model_calls: Annotated[list[ModelCallRecord], append_list]
    tool_calls: Annotated[list[ToolCallRecord], append_list]
    tool_results: Annotated[list[ToolCallRecord], append_list]

    # ReAct 中间控制
    pending_tool_call: ToolCallRecord | None
    react_decision_summary: str | None
    react_finished: bool

    # 校验
    validation_results: Annotated[list[ValidationResult], append_list]

    # 人工确认 / 用户追问
    need_user_clarification: bool
    clarification_question: str | None

    need_human_confirm: bool
    human_confirm_request: HumanConfirmRequest | None
    human_confirm_result: HumanConfirmResult | None

    # 执行轨迹
    action_trace: Annotated[list[ActionTraceItem], append_list]

    # 错误
    errors: Annotated[list[AgentError], append_list]

    # 输出
    final_answer: str | None
    structured_output: dict[str, Any] | None
    confidence: float | None

    # Debug
    debug: bool
    debug_info: Annotated[dict[str, Any], merge_dict]
```

---

## 9. 字段分组说明

### 9.1 基础标识

| 字段           | 说明              |
| ------------ | --------------- |
| `request_id` | 单次请求 ID         |
| `session_id` | 多轮会话 ID         |
| `user_id`    | 用户 ID，后续用于权限与隔离 |

---

### 9.2 输入字段

| 字段               | 说明                               |
| ---------------- | -------------------------------- |
| `user_input`     | 用户原始文本                           |
| `input_type`     | text / image / multimodal / file |
| `image_refs`     | 图片引用，不直接保存图片二进制                  |
| `file_refs`      | 文件引用，不直接保存文件二进制                  |
| `input_metadata` | 来源、客户端、调试参数等                     |

---

### 9.3 路由字段

| 字段             | 说明                 |
| -------------- | ------------------ |
| `route_result` | TaskRouter 生成的路由结果 |
| `status`       | 当前请求整体状态           |
| `step_count`   | 当前执行步数             |
| `max_steps`    | 防止 ReAct 无限循环      |

---

### 9.4 检索与证据字段

| 字段                  | 说明           |
| ------------------- | ------------ |
| `memory_candidates` | 记忆检索结果       |
| `retrieved_context` | 文档 / 知识库检索结果 |
| `evidence`          | 最终回答依据       |

---

### 9.5 观察与工具字段

| 字段             | 说明                                    |
| -------------- | ------------------------------------- |
| `observations` | VLM / LLM / Memory / Tool 等观察结果 |
| `model_calls`  | 模型调用摘要                                |
| `tool_calls`   | 工具调用计划                                |
| `tool_results` | 工具执行结果                                |

---

### 9.6 ReAct 字段

| 字段                       | 说明            |
| ------------------------ | ------------- |
| `pending_tool_call`      | 当前待执行工具调用     |
| `react_decision_summary` | 当前 ReAct 决策摘要 |
| `react_finished`         | ReAct 是否结束    |

说明：

ReAct 不作为顶层 `RouteType`。
它属于 `TOOL_ACT` 分支内部执行机制。

---

### 9.7 人工确认字段

| 字段                      | 说明       |
| ----------------------- | -------- |
| `need_human_confirm`    | 是否需要人工确认 |
| `human_confirm_request` | 人工确认请求   |
| `human_confirm_result`  | 人工确认结果   |

后续 LangGraph `interrupt()` 可以读取这些字段，实现 HITL。

---

### 9.8 输出字段

| 字段                  | 说明         |
| ------------------- | ---------- |
| `final_answer`      | 最终自然语言回答   |
| `structured_output` | JSON 结构化结果 |
| `confidence`        | 最终置信度      |

---

## 10. 初始状态构造函数

```python
from uuid import uuid4
from langchain_core.messages import HumanMessage


def infer_input_type(
    user_input: str,
    image_refs: list[ImageRef],
    file_refs: list[FileRef],
) -> InputType:
    has_text = bool(user_input.strip())
    has_image = bool(image_refs)
    has_file = bool(file_refs)

    if has_text and has_image:
        return InputType.MULTIMODAL
    if has_image:
        return InputType.IMAGE
    if has_file:
        return InputType.FILE
    return InputType.TEXT


def create_initial_state(
    user_input: str,
    session_id: str,
    request_id: str | None = None,
    user_id: str | None = None,
    image_refs: list[ImageRef] | None = None,
    file_refs: list[FileRef] | None = None,
    input_metadata: dict[str, Any] | None = None,
    debug: bool = False,
    max_steps: int = 6,
) -> AgentState:
    image_refs = image_refs or []
    file_refs = file_refs or []

    if not user_input.strip() and not image_refs and not file_refs:
        raise ValueError("user_input、image_refs、file_refs 不能同时为空")

    input_type = infer_input_type(user_input, image_refs, file_refs)

    return {
        "request_id": request_id or str(uuid4()),
        "session_id": session_id,
        "user_id": user_id,

        "user_input": user_input,
        "input_type": input_type,
        "image_refs": image_refs,
        "file_refs": file_refs,
        "input_metadata": input_metadata or {},

        "messages": [HumanMessage(content=user_input)] if user_input else [],

        "history_summary": None,
        "short_term_context": [],

        "status": RunStatus.INIT,
        "step_count": 0,
        "max_steps": max_steps,

        "memory_candidates": [],
        "retrieved_context": [],
        "evidence": [],

        "observations": [],
        "model_calls": [],
        "tool_calls": [],
        "tool_results": [],

        "pending_tool_call": None,
        "react_decision_summary": None,
        "react_finished": False,

        "validation_results": [],

        "need_user_clarification": False,
        "clarification_question": None,

        "need_human_confirm": False,
        "human_confirm_request": None,
        "human_confirm_result": None,

        "action_trace": [],
        "errors": [],

        "final_answer": None,
        "structured_output": None,
        "confidence": None,

        "debug": debug,
        "debug_info": {},
    }
```

---

## 11. Patch Helper 设计

### 11.1 追加 trace

```python
def trace_patch(
    *,
    step: int,
    node: str,
    action: str,
    status: StepStatus,
    reason: str = "",
    input_summary: str | None = None,
    output_summary: str | None = None,
    latency_ms: int | None = None,
    confidence: float | None = None,
    error_message: str | None = None,
) -> dict:
    return {
        "action_trace": [
            ActionTraceItem(
                step=step,
                node=node,
                action=action,
                status=status,
                reason=reason,
                input_summary=input_summary,
                output_summary=output_summary,
                latency_ms=latency_ms,
                confidence=confidence,
                error_message=error_message,
            )
        ]
    }
```

---

### 11.2 追加 error

```python
def error_patch(
    *,
    error_type: str,
    message: str,
    node: str | None = None,
    recoverable: bool = True,
    detail: dict[str, Any] | None = None,
) -> dict:
    return {
        "errors": [
            AgentError(
                error_type=error_type,
                message=message,
                node=node,
                recoverable=recoverable,
                detail=detail or {},
            )
        ]
    }
```

---

### 11.3 写入最终答案

```python
def final_answer_patch(
    *,
    final_answer: str,
    structured_output: dict[str, Any] | None = None,
    confidence: float | None = None,
) -> dict:
    return {
        "final_answer": final_answer,
        "structured_output": structured_output,
        "confidence": confidence,
        "status": RunStatus.COMPLETED,
    }
```

---

## 12. 对外响应过滤

```python
def to_public_response(state: AgentState) -> dict[str, Any]:
    return {
        "request_id": state.get("request_id"),
        "session_id": state.get("session_id"),
        "status": state.get("status"),

        "answer": state.get("final_answer"),
        "structured_output": state.get("structured_output"),
        "confidence": state.get("confidence"),

        "route": (
            state["route_result"].model_dump()
            if state.get("route_result")
            else None
        ),

        "need_user_clarification": state.get("need_user_clarification", False),
        "clarification_question": state.get("clarification_question"),

        "need_human_confirm": state.get("need_human_confirm", False),
        "human_confirm_request": (
            state["human_confirm_request"].model_dump()
            if state.get("human_confirm_request")
            else None
        ),

        "trace": [
            item.model_dump()
            for item in state.get("action_trace", [])
        ],

        "errors": [
            {
                "error_type": err.error_type,
                "message": err.message,
                "node": err.node,
                "recoverable": err.recoverable,
            }
            for err in state.get("errors", [])
        ],
    }
```

不允许对外返回：

```text
debug_info
完整 prompt
完整 raw model output
内部异常堆栈
完整 Chain-of-Thought
工具密钥
数据库连接信息
```

---

## 13. 节点契约

所有 LangGraph node 必须遵循：

```python
def node_name(state: AgentState) -> dict:
    ...
```

节点只返回状态更新，不返回下一个节点名。

---

### 13.1 route_task 示例

```python
def route_task(state: AgentState) -> dict:
    route_result = task_router.route(
        user_input=state.get("user_input", ""),
        input_type=state.get("input_type", InputType.TEXT),
        image_refs=state.get("image_refs", []),
        file_refs=state.get("file_refs", []),
    )

    return {
        "route_result": route_result,
        "status": RunStatus.ROUTED,
        "action_trace": [
            ActionTraceItem(
                step=state.get("step_count", 0) + 1,
                node="route_task",
                action="route",
                status=StepStatus.SUCCESS,
                reason=route_result.reason,
                confidence=route_result.confidence,
            )
        ],
        "step_count": state.get("step_count", 0) + 1,
    }
```

---

### 13.2 route_after_task 示例

```python
def route_after_task(state: AgentState) -> str:
    route_result = state.get("route_result")

    if route_result is None:
        return "fallback"

    if route_result.route_type == RouteType.VISION_DIRECT:
        return "vision_direct"

    if route_result.route_type == RouteType.VISION_SCHEMA:
        return "vision_schema"

    if route_result.route_type == RouteType.RAG_QA:
        return "retrieve"

    if route_result.route_type == RouteType.TOOL_ACT:
        return "react_subgraph"

    return "fallback"
```

---

## 14. 主图推荐结构

```text
START
  ↓
normalize_input
  ↓
load_short_term_context
  ↓
route_task
  ↓
conditional edge
  ├── vision_direct
  │     ↓
  │   validate_direct
  │     ↓
  │   respond
  │     ↓
  │   update_memory
  │     ↓
  │   END
  │
  ├── vision_schema
  │     ↓
  │   validate_schema
  │     ↓
  │   respond
  │     ↓
  │   update_memory
  │     ↓
  │   END
  │
  ├── retrieve
  │     ↓
  │   reason
  │     ↓
  │   should_verify?
  │     ├── verify
  │     └── respond
  │     ↓
  │   update_memory
  │     ↓
  │   END
  │
  ├── react_subgraph
  │     ↓
  │   respond
  │     ↓
  │   update_memory
  │     ↓
  │   END
  │
  └── fallback
        ↓
      respond
        ↓
      END
```

---

## 15. ReAct 子图状态使用约定

ReAct 子图仍然使用同一个 `AgentState`，但只读写以下字段：

```text
pending_tool_call
react_decision_summary
react_finished
tool_calls
tool_results
observations
action_trace
errors
step_count
need_human_confirm
human_confirm_request
human_confirm_result
final_answer
```

推荐子图结构：

```text
react_decide
  ↓
should_call_tool?
  ├── no  → react_finish
  └── yes → validate_tool_call
              ↓
          need_confirm?
              ├── yes → human_confirm
              └── no  → execute_tool
                         ↓
                      observe_tool_result
                         ↓
                    should_continue?
                         ├── yes → react_decide
                         └── no  → react_finish
```

---

## 16. 字段生命周期

| 字段                   | 生命周期 | 是否建议落库 |
| -------------------- | ---- | ------ |
| `request_id`         | 单次请求 | 是      |
| `session_id`         | 多轮会话 | 是      |
| `user_input`         | 单次请求 | 可选     |
| `image_refs`         | 单次请求 | 是      |
| `file_refs`          | 单次请求 | 是      |
| `route_result`       | 单次请求 | 是      |
| `action_trace`       | 单次请求 | 是      |
| `observations`       | 单次请求 | 可选     |
| `model_calls`        | 单次请求 | 可选     |
| `tool_calls`         | 单次请求 | 是      |
| `tool_results`       | 单次请求 | 是      |
| `evidence`           | 单次请求 | 可选     |
| `validation_results` | 单次请求 | 可选     |
| `final_answer`       | 单次请求 | 是      |
| `errors`             | 单次请求 | 是      |
| `debug_info`         | 单次请求 | 默认不落库  |

---

## 17. 持久化说明

`AgentState` 是运行时状态，**不负责完整执行轨迹持久化**。

执行轨迹的持久化表 `agent_trace_runs` / `agent_trace_events` 由 `AgentTrace_Schema` 统一定义。AgentState 只通过 `action_trace` 字段保留面向用户和最终响应的轻量动作摘要。

- `ActionTraceItem` 是轻量动作摘要，不等同于 `AgentTraceEvent`。
- `AgentTraceEvent` 是细粒度执行事件，用于可视化、调试和回放。
- 不再使用 `agent_runs` / `agent_traces` 表名。

---

## 18. 边界约束

| 场景                                | 处理方式                                        |
| --------------------------------- | ------------------------------------------- |
| 用户输入为空，且无图片、无文件                   | `create_initial_state()` 抛出 `ValueError`    |
| `route_result` 为空却进入后续节点          | 进入 `fallback`                               |
| `step_count >= max_steps`         | 停止 ReAct，进入兜底响应                             |
| VLM / LLM 调用失败                    | 写入 `errors` 和 `action_trace`                |
| 工具调用失败                            | 写入 `tool_results`、`errors` 和 `action_trace` |
| 高风险工具调用                           | 设置 `need_human_confirm = True`              |
| 用户问题不清晰                           | 设置 `need_user_clarification = True`         |
| `final_answer` 为空但状态为 `COMPLETED` | 视为状态非法                                      |
| `structured_output` 不可 JSON 序列化   | 不允许写入                                       |
| `debug_info` 包含敏感信息               | 不允许对外返回                                     |
| 模型输出完整推理链                         | 不允许保存                                       |

---

## 19. V0 验收标准

1. 所有执行路径都使用同一个 `AgentState`；
2. 不再存在 `StateContext dataclass`；
3. 不再存在 `Handler.execute(ctx) -> str`；
4. 不再由 Handler 返回下一个节点；
5. 所有节点都是 LangGraph node；
6. 所有节点返回 partial update；
7. 所有路由由 conditional edge 完成；
8. 追加型字段使用 reducer；
9. FastAPI 是唯一业务入口；
10. Streamlit 只通过 HTTP / SSE 调用 FastAPI；
11. `to_public_response()` 不暴露内部调试信息；
12. 不保存完整 Chain-of-Thought；
13. ReAct 只作为 `TOOL_ACT` 分支的子图存在。

---

## 20. 给 AI 编码助手的约束

1. 不要新增自定义状态机类；
2. 不要新增 Handler 协议；
3. 不要写 `execute(ctx) -> str`；
4. 不要让 node 返回下一个节点名；
5. 不要在 AgentState 中写业务逻辑；
6. 不要随意新增状态字段；
7. 新增字段必须同步更新本文档；
8. 所有追加型字段必须配置 reducer；
9. 所有节点必须返回 dict patch；
10. 所有错误必须写入 `errors`；
11. 所有外部调用必须追加 `action_trace`；
12. 对外响应必须经过 `to_public_response()`；
13. ReAct 循环必须是 LangGraph 子图或显式节点循环，不能隐藏在单个超大 while 节点里。

---

## 21. 最小落地顺序

```text
第 1 步：创建 app/agent/state.py
第 2 步：定义枚举、子 Schema、AgentState
第 3 步：实现 create_initial_state()
第 4 步：实现 to_public_response()
第 5 步：实现 reducers
第 6 步：重写 app/agent/graph.py
第 7 步：将旧 Handler 拆成 LangGraph node function
第 8 步：删除 AgentStateMachine / HandlerRegistry
第 9 步：FastAPI 调用 compiled_graph.invoke / stream
第 10 步：Streamlit 改为 HTTP / SSE 调 FastAPI
第 11 步：接入 react_subgraph
```

---

## 22. 本版相对旧版的关键变化

| 旧版                          | 新版                              |
| --------------------------- | ------------------------------- |
| `AgentState(BaseModel)`     | `AgentState(TypedDict)`         |
| `StateContext dataclass`    | 删除                              |
| `Handler.execute(ctx)`      | 删除                              |
| `execute(ctx) -> next_node` | 删除                              |
| `StateMachine` 决定流转         | LangGraph conditional edge 决定流转 |
| `current_node` 作为状态字段       | 不再作为核心字段                        |
| `should_use_react`          | 删除                              |
| `REACT` 作为 RouteType        | 删除                              |
| ReActExecutor 直接循环          | ReAct subgraph                  |
| append 函数修改 state           | patch helper 返回 dict            |
| 状态字段普通 list                 | 追加型字段使用 reducer                 |
| Streamlit 直接调用 Agent        | Streamlit 调 FastAPI             |

---

## 23. 最终原则

```text
LangGraph 负责图执行
AgentState 负责状态协议
Node 负责业务处理
Router 负责条件跳转
Reducer 负责状态合并
Subgraph 负责复杂循环
FastAPI 负责统一入口
Streamlit 负责 UI 展示
```

AgentState 不是业务对象，也不是状态机控制器。
它只是 LangGraph 节点之间共享的、可审计的、可持久化摘要化的运行时状态协议。
