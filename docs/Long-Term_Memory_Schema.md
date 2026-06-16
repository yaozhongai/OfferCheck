# Long-Term Memory Schema 设计文档

> 版本：V0.1 | 日期：2026-06-10 | **已实现** (`app/memory/ltm_schema.py` + `app/memory/long_term.py`)  
> 目标：为 Nexa Agent 提供跨会话长期记忆能力，约束长期记忆的持久化、偏好 / 事实 / 经验类记忆、检索、更新、遗忘与跨会话复用。  
> 原则：Long-Term Memory 负责跨 session 的用户级沉淀；Short-Term Memory 负责当前 session 上下文；AgentState 负责单次请求状态协议；AgentTrace 负责执行轨迹事件流；Reflection / Memory Gate 负责记忆晋升判断。

---

## 1. 模块概述

### 1.1 做什么

`Long-Term Memory` 是 Nexa Agent 的跨会话记忆持久化模块。

它主要负责：

1. 保存跨会话稳定有效的用户偏好、用户事实、项目事实、经验总结；
2. 为新会话或新请求提供可检索、可过滤、可注入的长期上下文；
3. 支持长期记忆的创建、合并、更新、废弃、遗忘与审计；
4. 支持语义检索、关键词检索、元数据过滤和混合排序；
5. 支持按 `user_id`、`scope`、`memory_type` 隔离不同记忆；
6. 支持将长期记忆结果转换为 `AgentState.memory_candidates` / `EvidenceItem`；
7. 支持用户主动查看、修改、删除自己的长期记忆；
8. 支持跨会话复用，但不直接复用完整历史对话。

---

### 1.2 不做什么

`Long-Term Memory` 不负责：

1. 不决定 LangGraph 节点流转；
2. 不替代 `AgentState`；
3. 不替代 `Short-Term Memory`；
4. 不替代 `AgentTrace`；
5. 不保存完整 Chain-of-Thought；
6. 不保存完整 prompt；
7. 不保存完整 raw model output；
8. 不保存完整对话日志；
9. 不保存图片、文件二进制本体；
10. 不作为 Trace Store 或前端 Timeline 数据源；
11. 不直接执行工具调用；
12. 不直接决定一条信息是否应该被长期保存，最终写入必须经过记忆门控或显式用户指令。

---

## 2. 与现有模块的边界

### 2.1 与 AgentState 的关系

`AgentState` 是单次请求在 LangGraph 节点间传递的运行时状态。

`Long-Term Memory` 是跨会话持久化 Store。

关系如下：

```text
retrieve / memory node
    ↓
Long-Term Memory Service
    ↓ retrieve_memories()
转换为 EvidenceItem
    ↓
AgentState.memory_candidates / evidence
    ↓
reason / respond
```

约束：

1. 不把 `LTMMemoryItem` 原始对象整体塞进 `AgentState`；
2. `AgentState.memory_candidates` 只保存轻量检索结果；
3. `AgentState.evidence` 可引用长期记忆，但只保存 `source_id=memory_id` 和摘要；
4. `AgentState` 不负责长期记忆 CRUD；
5. `update_memory` 节点只调用长期记忆服务，不直接操作数据库；
6. LTM 写入失败不应阻断主回答流程。

---

### 2.2 与 Short-Term Memory 的关系

`Short-Term Memory` 负责当前 session 的上下文缓存。

`Long-Term Memory` 负责跨 session 的稳定沉淀。

关系如下：

```text
当前 session 多轮交互
    ↓
Short-Term Memory 记录上下文
    ↓
Reflection / Memory Gate 判断是否值得长期保存
    ↓
Long-Term Memory 写入偏好 / 事实 / 经验
    ↓
后续新 session 可检索复用
```

约束：

1. LTM 不复用 `stm_sessions`、`stm_turns`、`stm_entries` 表；
2. STM 记录“当前会话上下文”，LTM 记录“跨会话稳定事实”；
3. STM 每轮始终写入，LTM 必须经过门控或用户显式要求；
4. STM 的 `session_id` 可作为 LTM 来源字段，但不是 LTM 主键；
5. STM 中的图片 / 文件引用如需进入 LTM，只保存引用和摘要，不保存二进制。

---

### 2.3 与 AgentTrace 的关系

`AgentTrace` 记录一次请求的执行事件流。

`Long-Term Memory` 记录跨会话记忆本体。

关系如下：

```text
retrieve / update_memory 节点
    ↓
调用 Long-Term Memory Service
    ↓
emit AgentTraceEvent:
  - retrieval_completed
  - memory_write_completed
  - memory_write_skipped
  - memory_write_failed
```

约束：

1. LTM 不创建 `agent_trace_runs`；
2. LTM 不创建 `agent_trace_events`；
3. LTM 不保存 `AgentTraceEvent` 列表；
4. LTM 表中只允许保存 `trace_id` / `request_id` 作为来源关联字段；
5. 前端 Timeline 仍由 AgentTrace 派生，不直接读取 LTM 表；
6. Trace 写入失败不影响 LTM 读写。

---

### 2.4 与 Reflection / Memory Gate 的关系

`Reflection / Memory Gate` 负责判断哪些信息值得进入长期记忆。

`Long-Term Memory` 负责保存、检索、更新、遗忘这些记忆。

关系如下：

```text
AgentState + STM 上下文
    ↓
Memory Gate / Reflection
    ↓
MemoryWriteCandidate[]
    ↓
Long-Term Memory upsert / merge / skip
```

约束：

1. LTM 可以提供基础规则校验，但不独占“是否写入”的判断；
2. 用户显式要求“记住 / 以后都这样 / 忘记”时，可绕过复杂反思流程，但仍需经过安全校验；
3. 反思日志如果后续单独设计，不能复用 LTM 表保存完整反思过程；
4. LTM 只保存最终可审计的记忆内容和来源摘要。

---

## 3. 数据流

### 3.1 长期记忆读取流

```text
FastAPI 接收请求
    ↓
create_initial_state()
    ↓
load_short_term_context
    ↓
route_task
    ↓
retrieve
    ↓
LongTermMemory.retrieve_memories()
    ↓
转换为 EvidenceItem
    ↓
写入 AgentState.memory_candidates
    ↓
reason / respond 使用相关记忆
    ↓
record_memory_use()
```

---

### 3.2 长期记忆写入流

```text
respond 完成
    ↓
update_memory
    ↓
写入 Short-Term Memory
    ↓
检查 route_result.need_memory_write / 用户显式记忆指令 / Reflection 结果
    ↓
生成 MemoryWriteCandidate
    ↓
LongTermMemory.upsert_memory()
    ↓
create / merge / supersede / skip
    ↓
emit memory_write_completed / memory_write_skipped
```

---

### 3.3 长期记忆更新流

```text
新信息进入 Memory Gate
    ↓
按 user_id + memory_type + normalized_key 查询冲突记忆
    ↓
判断是否：
  - create 新记忆
  - update 当前记忆
  - merge 到当前记忆
  - supersede 旧记忆
  - skip 写入
    ↓
写入 ltm_memory_events 审计记录
    ↓
更新 embedding
```

---

### 3.4 长期记忆遗忘流

```text
用户要求忘记 / 系统策略触发
    ↓
resolve target memories
    ↓
创建 ltm_forget_requests
    ↓
soft forget:
      ltm_memory_items.status = forgotten
      ltm_memory_embeddings.status = disabled
    ↓
写入 ltm_memory_events
    ↓
后续检索默认排除 forgotten
```

说明：

1. 实时路径默认执行软遗忘；
2. 物理删除只能由异步清理任务执行；
3. 软遗忘后不得继续进入检索结果；
4. 如需审计，可保留最小 tombstone，不保留原始敏感内容。

---

## 4. 记忆类型定义

### 4.1 Preference：偏好类记忆

用于保存用户长期稳定的选择倾向、表达偏好、工作方式偏好。

示例：

```text
用户偏好北京岗位优先。
用户希望技术文档先给结论再给细节。
用户不希望回答太情绪化。
```

写入条件：

1. 用户显式表达“以后 / 从现在开始 / 我更喜欢 / 不要再”；
2. 多次重复出现的稳定选择；
3. 对后续回答有明确个性化价值。

不应写入：

1. 一次性的临时选择；
2. 当前任务内的短期约束；
3. 无法泛化到未来会话的信息。

---

### 4.2 Fact：事实类记忆

用于保存相对稳定的用户事实、项目事实、环境事实。

示例：

```text
用户正在开发 Nexa Agent。
Nexa Agent 当前使用 FastAPI + Streamlit + LangGraph。
用户的主线方向是 Agent + VLM 应用工程化。
```

写入条件：

1. 事实对后续会话明显有用；
2. 事实具有一定稳定性；
3. 来源可追溯；
4. 可以被后续更新或废弃。

不应写入：

1. 未确认的推测；
2. 一次性任务参数；
3. 过期后会误导后续回答的信息。

---

### 4.3 Experience：经验类记忆

用于保存从交互、项目推进、问题解决中沉淀出的经验总结。

示例：

```text
用户在 Nexa Agent 架构设计中已明确 LangGraph 必须作为唯一图执行引擎。
用户在 Streamlit UI 迭代中更关注上传体验、即时反馈和 Trace 展示可读性。
```

写入条件：

1. 来自一次或多次对话的可复用总结；
2. 对后续项目决策有参考价值；
3. 已经被用户确认或在上下文中高度明确；
4. 不是完整对话复述。

不应写入：

1. 完整聊天记录；
2. 模型内部推理过程；
3. 未经总结的 Trace 事件；
4. 仅对当前请求有效的执行细节。

---

## 5. 枚举定义

```python
from enum import Enum


class LTMMemoryType(str, Enum):
    PREFERENCE = "preference"
    FACT = "fact"
    EXPERIENCE = "experience"


class LTMMemoryScope(str, Enum):
    USER = "user"              # 用户级，跨项目可复用
    PROJECT = "project"        # 项目级，例如 Nexa Agent
    SESSION = "session"        # 从 session 晋升而来，但仍需跨会话复用
    SYSTEM = "system"          # 系统内置长期规则，普通用户不可修改


class LTMMemoryStatus(str, Enum):
    ACTIVE = "active"
    PENDING_REVIEW = "pending_review"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    FORGOTTEN = "forgotten"
    DELETED = "deleted"


class LTMSourceType(str, Enum):
    USER_EXPLICIT = "user_explicit"
    USER_MESSAGE = "user_message"
    ASSISTANT_SUMMARY = "assistant_summary"
    REFLECTION = "reflection"
    SYSTEM_IMPORT = "system_import"
    MANUAL_EDIT = "manual_edit"


class LTMSensitivity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    RESTRICTED = "restricted"


class LTMEventType(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    MERGED = "merged"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    FORGOTTEN = "forgotten"
    RESTORED = "restored"
    DELETED = "deleted"
    USED = "used"


class LTMEmbeddingStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"
    DISABLED = "disabled"


class LTMForgetStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
```

---

## 6. 表 / 集合设计

V0 阶段建议使用以下逻辑集合：

```text
ltm_memory_items
ltm_memory_embeddings
ltm_memory_events
ltm_forget_requests
```

说明：

1. 统一使用 `ltm_` 前缀，避免与 `stm_*`、`agent_trace_*`、`AgentState` 字段冲突；
2. `ltm_memory_items` 是长期记忆主表；
3. `ltm_memory_embeddings` 可由向量数据库实现，也可用关系库记录向量索引元信息；
4. `ltm_memory_events` 记录记忆变更审计，不是 AgentTrace；
5. `ltm_forget_requests` 记录用户或系统发起的遗忘请求。

---

### 6.1 `ltm_memory_items`

长期记忆主表。

| 字段名 | 类型 | 必填 | 设计理由 |
|---|---|---|---|
| `memory_id` | String / UUID | ✅ | 长期记忆唯一 ID |
| `user_id` | String | ✅ | 用户隔离，LTM 必须有用户归属 |
| `scope` | Enum: `user / project / session / system` | ✅ | 控制跨会话复用范围 |
| `project_id` | String nullable |  | 项目级记忆所属项目，例如 `nexa_agent` |
| `memory_type` | Enum: `preference / fact / experience` | ✅ | 区分偏好、事实、经验 |
| `status` | Enum | ✅ | 生命周期状态，默认 `active` |
| `title` | String nullable |  | 面向管理界面的短标题 |
| `content` | Text | ✅ | 可被模型使用的记忆正文，必须是摘要化内容 |
| `normalized_key` | String nullable |  | 归一化键，用于去重和更新，如 `career.city.preference` |
| `subject` | String nullable |  | 事实三元组主语，可选 |
| `predicate` | String nullable |  | 事实三元组谓词，可选 |
| `object_value` | Text nullable |  | 事实三元组宾语或偏好值，可选 |
| `tags` | JSON Array | ✅ | 标签，如 `career`、`project:nexa` |
| `confidence` | Float | ✅ | 0-1，记忆可信度 |
| `importance` | Float | ✅ | 0-1，检索和注入优先级 |
| `sensitivity` | Enum | ✅ | 隐私敏感级别 |
| `source_type` | Enum | ✅ | 来源类型，如用户显式、反思、人工编辑 |
| `source_session_id` | String nullable |  | 来源 session，仅做关联 |
| `source_request_id` | String nullable |  | 来源请求 ID，仅做关联 |
| `source_trace_id` | String nullable |  | 来源 Trace ID，仅做关联 |
| `source_turn_id` | String nullable |  | 来源 STM turn，仅做关联 |
| `source_entry_ids` | JSON Array | ✅ | 来源 STM entry 引用列表 |
| `source_summary` | Text nullable |  | 来源摘要，不保存完整对话 |
| `version` | Integer | ✅ | 版本号，更新时递增 |
| `supersedes_memory_id` | String nullable |  | 当前记忆替代的旧记忆 |
| `superseded_by_memory_id` | String nullable |  | 当前记忆被哪条新记忆替代 |
| `valid_from` | Timestamp nullable |  | 事实或偏好生效时间 |
| `valid_until` | Timestamp nullable |  | 事实或偏好过期时间 |
| `expires_at` | Timestamp nullable |  | 自动过期时间 |
| `last_used_at` | Timestamp nullable |  | 最近一次被检索使用的时间 |
| `use_count` | Integer | ✅ | 被使用次数 |
| `created_at` | Timestamp | ✅ | 创建时间 |
| `updated_at` | Timestamp | ✅ | 更新时间 |
| `metadata` | JSON | ✅ | 扩展字段 |

约束：

1. `user_id + memory_type + normalized_key + status=active` 应尽量唯一；
2. `content` 必须是可审计摘要，禁止保存完整 CoT、完整 prompt、完整 raw output；
3. `source_trace_id` 只是来源关联，不能从 Trace 反向构造长期记忆；
4. `status=forgotten / deleted` 的记录默认不得进入检索结果；
5. `sensitivity=restricted` 的记忆默认不得自动注入模型上下文，除非用户明确请求或策略允许；
6. `version` 更新时必须写入 `ltm_memory_events`。

---

### 6.2 `ltm_memory_embeddings`

长期记忆向量索引表或向量库元信息表。

| 字段名 | 类型 | 必填 | 设计理由 |
|---|---|---|---|
| `embedding_id` | String / UUID | ✅ | 向量记录唯一 ID |
| `memory_id` | String / UUID | ✅ | 关联 `ltm_memory_items.memory_id` |
| `user_id` | String | ✅ | 支持向量库侧用户过滤 |
| `embedding_model` | String | ✅ | 向量模型名称 |
| `embedding_dim` | Integer | ✅ | 向量维度 |
| `embedding_text` | Text | ✅ | 用于生成 embedding 的摘要文本 |
| `content_hash` | String | ✅ | 检测内容是否变化 |
| `vector_ref` | String nullable |  | 外部向量库向量 ID 或 collection key |
| `status` | Enum: `pending / ready / failed / disabled` | ✅ | 向量状态 |
| `last_embedded_at` | Timestamp nullable |  | 最近向量化时间 |
| `error_message` | Text nullable |  | 向量化失败摘要 |
| `created_at` | Timestamp | ✅ | 创建时间 |
| `updated_at` | Timestamp | ✅ | 更新时间 |
| `metadata` | JSON | ✅ | 扩展字段 |

约束：

1. `embedding_text` 不等于完整原文，只能是 LTM content + 少量 tags；
2. `memory_id` 被遗忘后，对应 embedding 必须置为 `disabled`；
3. 向量化失败不应导致主流程失败，但该记忆不能参与语义检索；
4. 内容更新后 `content_hash` 改变，必须重新生成 embedding。

---

### 6.3 `ltm_memory_events`

长期记忆变更审计表。

| 字段名 | 类型 | 必填 | 设计理由 |
|---|---|---|---|
| `event_id` | String / UUID | ✅ | 事件唯一 ID |
| `memory_id` | String / UUID | ✅ | 关联长期记忆 |
| `user_id` | String | ✅ | 用户隔离 |
| `event_type` | Enum | ✅ | created / updated / merged / forgotten 等 |
| `request_id` | String nullable |  | 来源请求 ID |
| `session_id` | String nullable |  | 来源 session ID |
| `trace_id` | String nullable |  | 来源 Trace ID |
| `actor` | String | ✅ | `user / assistant / system / admin` |
| `reason` | Text nullable |  | 变更原因摘要 |
| `old_snapshot` | JSON nullable |  | 变更前关键字段快照 |
| `new_snapshot` | JSON nullable |  | 变更后关键字段快照 |
| `created_at` | Timestamp | ✅ | 事件创建时间 |
| `metadata` | JSON | ✅ | 扩展字段 |

约束：

1. 这是 LTM 内部审计事件，不是 AgentTraceEvent；
2. 不用于前端执行 Timeline；
3. 不保存完整对话、完整 prompt、完整模型输出；
4. 忘记或删除敏感内容时，`old_snapshot` 只能保留脱敏摘要或置空。

---

### 6.4 `ltm_forget_requests`

长期记忆遗忘请求表。

| 字段名 | 类型 | 必填 | 设计理由 |
|---|---|---|---|
| `forget_request_id` | String / UUID | ✅ | 遗忘请求唯一 ID |
| `user_id` | String | ✅ | 用户隔离 |
| `request_id` | String nullable |  | 触发遗忘的请求 ID |
| `session_id` | String nullable |  | 触发遗忘的 session ID |
| `trace_id` | String nullable |  | 触发遗忘的 trace ID |
| `query` | Text nullable |  | 用户表达的遗忘范围 |
| `target_memory_ids` | JSON Array | ✅ | 命中的记忆 ID |
| `status` | Enum: `pending / completed / partial / failed` | ✅ | 处理状态 |
| `strategy` | String | ✅ | `soft_forget / hard_delete / disable_embedding` |
| `reason` | Text nullable |  | 遗忘原因摘要 |
| `created_at` | Timestamp | ✅ | 创建时间 |
| `completed_at` | Timestamp nullable |  | 完成时间 |
| `metadata` | JSON | ✅ | 扩展字段 |

约束：

1. 默认策略为 `soft_forget`；
2. 实时请求不得直接大批量物理删除；
3. `target_memory_ids` 为空时不得执行模糊删除；
4. hard delete 必须由后台任务或管理员流程执行。

---

## 7. Schema 定义

### 7.1 长期记忆主对象

```python
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class LTMMemoryItem(BaseModel):
    memory_id: str
    user_id: str

    scope: LTMMemoryScope = LTMMemoryScope.USER
    project_id: str | None = None

    memory_type: LTMMemoryType
    status: LTMMemoryStatus = LTMMemoryStatus.ACTIVE

    title: str | None = None
    content: str

    normalized_key: str | None = None

    subject: str | None = None
    predicate: str | None = None
    object_value: str | None = None

    tags: list[str] = Field(default_factory=list)

    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    sensitivity: LTMSensitivity = LTMSensitivity.LOW

    source_type: LTMSourceType = LTMSourceType.ASSISTANT_SUMMARY
    source_session_id: str | None = None
    source_request_id: str | None = None
    source_trace_id: str | None = None
    source_turn_id: str | None = None
    source_entry_ids: list[str] = Field(default_factory=list)
    source_summary: str | None = None

    version: int = 1
    supersedes_memory_id: str | None = None
    superseded_by_memory_id: str | None = None

    valid_from: datetime | None = None
    valid_until: datetime | None = None
    expires_at: datetime | None = None

    last_used_at: datetime | None = None
    use_count: int = 0

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    metadata: dict[str, Any] = Field(default_factory=dict)
```

---

### 7.2 长期记忆向量对象

```python
class LTMEmbeddingRecord(BaseModel):
    embedding_id: str
    memory_id: str
    user_id: str

    embedding_model: str
    embedding_dim: int
    embedding_text: str
    content_hash: str

    vector_ref: str | None = None
    status: LTMEmbeddingStatus = LTMEmbeddingStatus.PENDING

    last_embedded_at: datetime | None = None
    error_message: str | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    metadata: dict[str, Any] = Field(default_factory=dict)
```

---

### 7.3 长期记忆事件对象

```python
class LTMMemoryEvent(BaseModel):
    event_id: str
    memory_id: str
    user_id: str

    event_type: LTMEventType

    request_id: str | None = None
    session_id: str | None = None
    trace_id: str | None = None

    actor: str = "assistant"
    reason: str | None = None

    old_snapshot: dict[str, Any] | None = None
    new_snapshot: dict[str, Any] | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

---

### 7.4 遗忘请求对象

```python
class LTMForgetRequest(BaseModel):
    forget_request_id: str
    user_id: str

    request_id: str | None = None
    session_id: str | None = None
    trace_id: str | None = None

    query: str | None = None
    target_memory_ids: list[str] = Field(default_factory=list)

    status: LTMForgetStatus = LTMForgetStatus.PENDING
    strategy: str = "soft_forget"
    reason: str | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
```

---

### 7.5 写入候选 DTO

`MemoryWriteCandidate` 是 Reflection / Memory Gate 输出给 LTM Service 的写入候选，不建议直接落库。

```python
class MemoryWriteCandidate(BaseModel):
    user_id: str

    memory_type: LTMMemoryType
    scope: LTMMemoryScope = LTMMemoryScope.USER
    project_id: str | None = None

    content: str
    normalized_key: str | None = None

    subject: str | None = None
    predicate: str | None = None
    object_value: str | None = None

    tags: list[str] = Field(default_factory=list)

    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    sensitivity: LTMSensitivity = LTMSensitivity.LOW

    source_type: LTMSourceType = LTMSourceType.REFLECTION
    source_session_id: str | None = None
    source_request_id: str | None = None
    source_trace_id: str | None = None
    source_turn_id: str | None = None
    source_entry_ids: list[str] = Field(default_factory=list)
    source_summary: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
```

---

### 7.6 检索结果 DTO

`LTMRetrievalResult` 是 LTM 检索结果。注入 `AgentState` 前需要转换成 `EvidenceItem`。

```python
class LTMRetrievalResult(BaseModel):
    memory_id: str
    memory_type: LTMMemoryType
    scope: LTMMemoryScope

    content: str
    title: str | None = None
    tags: list[str] = Field(default_factory=list)

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_score: float | None = Field(default=None, ge=0.0, le=1.0)
    keyword_score: float | None = Field(default=None, ge=0.0, le=1.0)
    recency_score: float | None = Field(default=None, ge=0.0, le=1.0)
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    source_summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

---

### 7.7 注入 AgentState 的 EvidenceItem 转换

长期记忆不得直接把完整 `LTMMemoryItem` 注入 `AgentState`。

推荐转换格式：

```python
def to_evidence_item(result: LTMRetrievalResult) -> dict:
    return {
        "source_type": "memory",
        "source_id": result.memory_id,
        "title": result.title,
        "content": result.content,
        "score": result.score,
        "metadata": {
            "memory_type": result.memory_type.value,
            "scope": result.scope.value,
            "tags": result.tags,
            "importance": result.importance,
            "confidence": result.confidence,
        },
    }
```

约束：

1. 只写入 `memory_candidates` 或 `evidence`；
2. 不新增 `AgentState.long_term_memory` 字段；
3. `content` 应限制长度，默认不超过 500 字；
4. 不包含 `old_snapshot`、`source_entry_ids` 等内部字段。

---

## 8. API 接口设计

### 8.1 生成写入候选

```python
def build_memory_candidates_from_state(
    state: dict,
    trace_id: str | None = None,
) -> list[MemoryWriteCandidate]:
    """
    从 AgentState 中提取可进入长期记忆门控的候选信息。

    只允许读取摘要化字段：
    - user_input 摘要
    - route_result 摘要
    - final_answer 摘要
    - observations 摘要
    - tool_results 摘要
    - short_term_context 摘要

    禁止读取或写入：
    - debug_info
    - 完整 prompt
    - 完整 raw model output
    - 完整 Chain-of-Thought
    - 内部异常堆栈
    """
```

---

### 8.2 写入或合并长期记忆

```python
def upsert_memory(
    candidate: MemoryWriteCandidate,
    merge_strategy: str = "normalized_key_first",
) -> LTMMemoryItem:
    """
    写入长期记忆。

    默认策略：
    1. 按 user_id + memory_type + normalized_key 查找 active 记忆；
    2. 如果不存在，创建新记忆；
    3. 如果存在且内容兼容，merge / update；
    4. 如果存在冲突，创建新版本并 supersede 旧记忆；
    5. 每次 create / update / merge / supersede 都写入 ltm_memory_events。
    """
```

---

### 8.3 检索长期记忆

```python
def retrieve_memories(
    user_id: str,
    query: str,
    memory_types: list[LTMMemoryType] | None = None,
    scope: list[LTMMemoryScope] | None = None,
    project_id: str | None = None,
    tags: list[str] | None = None,
    limit: int = 8,
    min_score: float = 0.25,
    include_sensitive: bool = False,
    include_archived: bool = False,
) -> list[LTMRetrievalResult]:
    """
    检索长期记忆。

    默认只返回：
    - status=active
    - user_id 匹配
    - 未过期
    - 非 restricted 或策略允许

    排序建议：
    final_score = semantic_score * 0.55
                + keyword_score * 0.20
                + importance * 0.15
                + recency_score * 0.10
    """
```

---

### 8.4 构建 AgentState Patch

```python
def long_term_memory_patch(
    user_id: str,
    query: str,
    request_id: str,
    session_id: str,
    trace_id: str | None = None,
    project_id: str | None = None,
    limit: int = 8,
    token_budget: int = 1200,
) -> dict:
    """
    检索长期记忆并返回可 merge 到 AgentState 的 patch。

    返回格式：
    {
        "memory_candidates": [EvidenceItem, ...]
    }

    注意：
    不直接写入 final_answer。
    不直接写入 short_term_context。
    不新增 long_term_memory 字段。
    """
```

---

### 8.5 更新长期记忆

```python
def update_memory_item(
    user_id: str,
    memory_id: str,
    patch: dict,
    reason: str,
    request_id: str | None = None,
    session_id: str | None = None,
    trace_id: str | None = None,
    actor: str = "assistant",
) -> LTMMemoryItem:
    """
    更新指定长期记忆。

    必须：
    - 校验 user_id 权限；
    - 增加 version；
    - 写入 ltm_memory_events；
    - 内容变化后重新生成 embedding。
    """
```

---

### 8.6 标记记忆被使用

```python
def record_memory_use(
    user_id: str,
    memory_ids: list[str],
    request_id: str | None = None,
    session_id: str | None = None,
    trace_id: str | None = None,
) -> None:
    """
    记录长期记忆被实际用于回答。

    只更新：
    - last_used_at
    - use_count
    - ltm_memory_events(event_type=used)

    不改变 content。
    """
```

---

### 8.7 遗忘指定记忆

```python
def forget_memory(
    user_id: str,
    memory_id: str,
    reason: str | None = None,
    request_id: str | None = None,
    session_id: str | None = None,
    trace_id: str | None = None,
    strategy: str = "soft_forget",
) -> LTMForgetRequest:
    """
    遗忘指定长期记忆。

    默认 soft_forget：
    - ltm_memory_items.status = forgotten
    - ltm_memory_embeddings.status = disabled
    - 写入 ltm_forget_requests
    - 写入 ltm_memory_events(event_type=forgotten)
    """
```

---

### 8.8 按查询批量遗忘

```python
def forget_memories_by_query(
    user_id: str,
    query: str,
    memory_types: list[LTMMemoryType] | None = None,
    project_id: str | None = None,
    dry_run: bool = True,
    limit: int = 20,
) -> list[LTMRetrievalResult] | LTMForgetRequest:
    """
    根据自然语言查询定位待遗忘记忆。

    dry_run=True 时只返回候选，不执行遗忘。
    dry_run=False 时必须有明确 target_memory_ids 后才能执行。
    """
```

---

### 8.9 用户查看长期记忆

```python
def list_user_memories(
    user_id: str,
    memory_type: LTMMemoryType | None = None,
    project_id: str | None = None,
    status: LTMMemoryStatus = LTMMemoryStatus.ACTIVE,
    limit: int = 100,
    offset: int = 0,
) -> list[LTMMemoryItem]:
    """
    面向记忆管理页的查询接口。
    默认只展示 active 记忆。
    """
```

---

## 9. 与 LangGraph 节点的集成约定

### 9.1 读取长期记忆

V0 阶段建议由现有 `retrieve` 节点统一读取长期记忆。

```text
route_task
  ↓
retrieve
  ├── document retriever
  └── long_term_memory.retrieve_memories()
  ↓
AgentState.retrieved_context / memory_candidates
```

如果后续复杂度上升，可拆分为独立节点：

```text
retrieve_documents
retrieve_long_term_memory
merge_context
```

但 V0 不强制拆分，避免过早增加节点数量。

---

### 9.2 写入长期记忆

V0 阶段仍由 `update_memory` 节点统一处理记忆写入。

```text
respond
  ↓
update_memory
  ├── write_short_term_memory()   # 每轮始终写入
  └── write_long_term_memory()    # 门控后写入
  ↓
END
```

约束：

1. STM 写入和 LTM 写入相互独立；
2. STM 写入失败不应影响 LTM 写入；
3. LTM 写入失败不应影响最终回答；
4. LTM 写入失败必须追加 `errors` 并 emit `memory_write_failed`；
5. `update_memory` 节点只返回 AgentState patch，不返回下一个节点名。

---

### 9.3 推荐文件位置

```text
app/
  memory/
    long_term.py              # LTM Store / Service
    memory_gate.py            # 记忆门控，可后续独立为 Reflection

  agent/
    nodes/
      retrieve.py             # 读取 LTM + 文档上下文
      memory.py               # 写入 STM + LTM

  api/
    routes/
      memory.py               # 用户查看 / 修改 / 删除长期记忆
```

文件职责：

| 文件 | 职责 |
|---|---|
| `memory/long_term.py` | LTM Schema、CRUD、检索、更新、遗忘 |
| `memory/memory_gate.py` | 判断是否生成 MemoryWriteCandidate |
| `agent/nodes/retrieve.py` | 读取长期记忆并写入 `memory_candidates` |
| `agent/nodes/memory.py` | 调用 STM 与 LTM 写入接口 |
| `api/routes/memory.py` | 用户管理长期记忆 |

---

## 10. 检索策略

### 10.1 默认混合检索

长期记忆检索默认使用：

```text
metadata filter
  + semantic vector search
  + keyword / normalized_key match
  + importance / confidence / recency rerank
```

推荐排序公式：

```text
final_score = semantic_score * 0.55
            + keyword_score  * 0.20
            + importance     * 0.15
            + recency_score  * 0.10
```

其中：

| 分数 | 说明 |
|---|---|
| `semantic_score` | 当前 query 与 memory embedding 的相似度 |
| `keyword_score` | normalized_key、tags、content 的关键词命中 |
| `importance` | 记忆重要度 |
| `recency_score` | 最近使用或最近更新时间 |

---

### 10.2 检索过滤条件

默认必须过滤：

```text
user_id == current_user_id
status == active
expires_at is null or expires_at > now
valid_until is null or valid_until > now
sensitivity != restricted unless allowed
```

可选过滤：

```text
project_id
memory_type
scope
tags
confidence >= threshold
importance >= threshold
```

---

### 10.3 注入上下文约束

默认建议：

```text
limit = 8
token_budget = 1200
single_memory_max_chars = 500
include_source_summary = false
include_sensitive = false
```

注入原则：

1. 优先注入与当前问题直接相关的记忆；
2. 偏好类记忆优先影响回答风格和约束；
3. 事实类记忆优先补充背景；
4. 经验类记忆优先辅助项目决策；
5. 不要把无关长期记忆塞入上下文；
6. 长期记忆不能覆盖用户当前显式指令。

---

## 11. 更新策略

### 11.1 Preference 更新

偏好类记忆按 `normalized_key` 更新。

示例：

```text
normalized_key = "job.city.preference"
旧值：用户优先考虑北京 / 上海。
新值：用户明确表示北京优先，暂不考虑广州。
```

处理方式：

1. 如果新旧偏好兼容：更新同一条 memory，version + 1；
2. 如果新旧偏好冲突：新建 memory，旧 memory 标记为 `superseded`；
3. 如果用户明确要求忘记：旧 memory 标记为 `forgotten`。

---

### 11.2 Fact 更新

事实类记忆必须区分“补充”和“冲突”。

处理方式：

1. 新事实补充旧事实：merge；
2. 新事实纠正旧事实：supersede；
3. 临时事实过期：archive 或设置 `valid_until`；
4. 低置信事实：进入 `pending_review`，默认不注入上下文。

---

### 11.3 Experience 更新

经验类记忆以“总结粒度”为核心。

处理方式：

1. 相同主题的新经验可 merge；
2. 更高层总结可 supersede 多条低层经验；
3. 不要把每次执行细节都追加为 experience；
4. experience 应保留为可指导后续决策的一句话或短段落。

---

## 12. 遗忘策略

### 12.1 软遗忘

默认策略。

```text
ltm_memory_items.status = forgotten
ltm_memory_embeddings.status = disabled
```

特点：

1. 实时安全；
2. 不影响审计；
3. 后续检索完全排除；
4. 可保留最小 tombstone 防止重复恢复或误用。

---

### 12.2 物理删除

仅适用于明确合规要求或用户强制删除。

约束：

1. 必须异步执行；
2. 必须先定位 `target_memory_ids`；
3. 必须删除 embedding；
4. 必须清理可恢复内容；
5. `ltm_forget_requests` 可保留脱敏记录。

---

### 12.3 用户可见管理

用户应能通过 API / UI：

1. 查看当前 active 长期记忆；
2. 按类型筛选偏好 / 事实 / 经验；
3. 修改错误记忆；
4. 删除或遗忘指定记忆；
5. 查看某条记忆的简要来源说明。

---

## 13. 安全与隐私约束

长期记忆禁止保存：

```text
完整 Chain-of-Thought
完整隐藏推理草稿
完整 prompt
完整 raw model output
内部异常堆栈
工具密钥
数据库连接信息
未脱敏 token
用户敏感凭证
图片 / 文件二进制本体
完整对话日志
未经确认的高敏感个人信息
```

长期记忆允许保存：

```text
用户显式要求保存的偏好
稳定事实摘要
项目背景摘要
可复用经验总结
用户确认的长期约束
可审计的来源摘要
图片 / 文件引用与摘要
```

高敏感信息处理原则：

1. 默认不主动写入；
2. 用户显式要求保存时才可写入；
3. 写入时 `sensitivity` 至少为 `high`；
4. 检索时默认不注入；
5. 用户要求遗忘时优先执行。

---

## 14. 边界与约束

| 边界情况 | 处理方式 |
|---|---|
| `user_id` 为空 | 不允许写入用户级长期记忆 |
| 用户显式要求记住 | 进入安全校验后可生成 MemoryWriteCandidate |
| 用户显式要求忘记 | 优先执行 forget flow，不依赖普通写入门控 |
| 与旧偏好冲突 | supersede 旧记忆，不静默覆盖 |
| 与旧事实冲突 | 低置信进入 pending_review，高置信 supersede |
| 记忆重复 | 按 `normalized_key` + 语义相似度 merge |
| embedding 失败 | item 可保存，但 embedding 状态为 failed，不参与语义检索 |
| 向量库不可用 | 降级关键词 + metadata 检索 |
| 检索结果过多 | 按 score、importance、token_budget 裁剪 |
| 敏感记忆命中 | 默认不注入，除非策略允许 |
| 记忆过期 | 默认排除，必要时 archive |
| 软遗忘后再次检索 | 必须排除 forgotten 记忆 |
| LTM 写入失败 | 不阻断主流程，写入 AgentState.errors 并 emit Trace 失败事件 |
| Trace 写入失败 | 不影响 LTM 读写 |
| 用户当前指令与长期偏好冲突 | 当前指令优先 |
| 代码临时新增字段 | 禁止，必须先更新 Schema |

---

## 15. V0 最小实现范围

V0 必须实现：

```text
LTMMemoryItem
LTMEmbeddingRecord
LTMMemoryEvent
LTMForgetRequest
MemoryWriteCandidate
LTMRetrievalResult

upsert_memory()
retrieve_memories()
long_term_memory_patch()
update_memory_item()
record_memory_use()
forget_memory()
forget_memories_by_query()
list_user_memories()
```

V0 可暂不实现：

```text
复杂 Reflection Log
多用户共享记忆
组织级记忆
复杂知识图谱关系
跨用户协同过滤
完整记忆推荐系统
自动物理删除任务
复杂敏感信息分类器
```

---

## 16. 验收标准

1. 所有 LTM 表都使用 `ltm_` 前缀；
2. 不新增 `conversation_turns` / `session_meta`；
3. 不复用 `stm_*` 表保存长期记忆；
4. 不复用 `agent_trace_*` 表保存长期记忆；
5. `user_id` 为空时不能写入用户级 LTM；
6. 能写入 preference / fact / experience 三类记忆；
7. 能按 user_id 隔离检索；
8. 能按 memory_type / project_id / tags 过滤；
9. 检索结果能转换为 `EvidenceItem`；
10. 不向 `AgentState` 新增 `long_term_memory` 字段；
11. 不保存完整 Chain-of-Thought；
12. 不保存完整 prompt；
13. 不保存完整 raw model output；
14. 不保存图片 / 文件二进制；
15. 更新记忆时 version 递增并写入 `ltm_memory_events`；
16. 遗忘记忆后默认检索不到；
17. embedding 失败不阻断主流程；
18. LTM 写入失败不阻断最终回答；
19. 用户当前指令优先于长期偏好；
20. 新增字段必须先更新本 Schema。

---

## 17. Vibe Coding 注意事项

```text
⚠️ 特别注意：

1. 不要新增 long_term_memory 字段到 AgentState，长期记忆检索结果统一转换为 EvidenceItem 后写入 memory_candidates / evidence。
2. 不要复用 stm_sessions / stm_turns / stm_entries；LTM 必须使用 ltm_memory_items 等 ltm_ 前缀表。
3. 不要复用 agent_trace_runs / agent_trace_events；Trace 只记录 LTM 读写事件，不保存 LTM 本体。
4. 不要在 LTM 中保存完整 prompt、完整 raw model output、完整 Chain-of-Thought、完整对话日志。
5. Preference / Fact / Experience 必须通过 memory_type 区分，不要用 metadata 临时塞类型。
6. 用户显式“记住”可以触发写入候选，但仍要做安全校验；用户显式“忘记”必须优先执行遗忘流程。
7. 更新已有记忆时不要静默覆盖，必须 version + 1，并写入 ltm_memory_events。
8. 软遗忘后必须禁用 embedding，并确保 retrieve_memories 默认排除 forgotten。
9. 当前请求中的用户显式指令优先级高于长期记忆。
10. 新增字段或枚举前必须先更新本 Schema，不允许在代码中临时乱加字段。
```

---

## 18. 最终原则

```text
Short-Term Memory 负责当前 session 上下文
Long-Term Memory 负责跨 session 稳定沉淀
Reflection / Memory Gate 负责记忆晋升判断
AgentState 负责单次请求状态传递
AgentTrace 负责执行事件流和前端 Timeline
```

`Long-Term Memory` 不是聊天记录库，不是 Trace 系统，也不是新的状态机。

它是 Nexa Agent 跨会话个性化与项目连续性的持久化记忆层。
