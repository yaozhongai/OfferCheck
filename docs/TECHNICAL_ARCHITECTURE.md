# Nexa Agent V0 — 技术架构文档

> 版本：V4.0 | 日期：2026-06-11 | 架构：LangGraph 原生 + TOOL_ACT (ReAct) + FALLBACK + STM/LTM

---

## 1. 项目概览

Nexa Agent V0 是一个 **LangGraph 原生、两层路由驱动、带完整 Trace 的多路径 Agent 系统**。

**Direct First, Agent When Needed。** VLM 直答不走 LLM，简单问答不反思，临时图片不写记忆。

---

## 2. 目录结构

```
app/
├── agent/                      # LangGraph 原生架构
│   ├── state.py                # AgentState TypedDict + Reducer + Sub-Schema
│   ├── graph.py                # build_agent_graph() → compiled StateGraph
│   ├── routers.py              # conditional edge
│   └── nodes/
│       ├── normalize.py        # 归一化
│       ├── load_context.py     # 加载上下文 (STM + LTM)
│       ├── route.py            # L1 规则 + L2 LLM 路由
│       ├── react_decide.py     # ReAct 推理决策
│       ├── react_execute.py    # 工具执行 + finish
│       ├── react_routers.py    # ReAct 条件路由
│       ├── respond.py          # 最终响应
│       ├── memory.py           # 记忆持久化
│       └── fallback.py         # 兜底
├── trace/                      # Trace 事件系统
│   ├── schema.py               # Trace 枚举 + 模型 + Payload
│   ├── store.py                # SQLAlchemy 存储 (agent_trace_runs / agent_trace_events)
│   ├── service.py              # create / emit / complete / fail / get / timeline
│   └── sse.py                  # SSE 流式推送 + after_seq 重连
├── storage/                    # 持久化层 (SQLAlchemy)
│   ├── database.py             # Engine + Session
│   └── models.py               # ORM 模型 (7表)
├── llm/                        # LLM 客户端
│   └── client.py               # DeepSeek / Kimi / GLM
├── api/
│   ├── schemas.py              # Pydantic API 契约
│   ├── deps.py                 # 依赖注入
│   └── routes/
│       ├── chat.py             # POST /api/v0/chat (Trace 集成)
│       ├── upload.py           # POST /api/v0/upload
│       ├── memory.py           # GET /api/v0/memory/*
│       └── trace.py            # GET /api/v0/trace/* (SSE + Timeline)
├── memory/                     # STM Schema + LTM Schema
│   ├── stm_schema.py           # STM 枚举 + Pydantic
│   ├── short_term.py           # STM Store (Turn/Entry/Session)
│   ├── ltm_schema.py           # LTM 枚举 + Pydantic (Preference/Fact/Experience)
│   └── long_term.py            # LTM Store (MemoryItem/Event/Forget + Gate)
├── pipeline/                   # llama.cpp VLM 引擎 + 提取管线
├── utils/                      # 日志 + 路由规则 + 校验
├── main.py / cli.py / streamlit_app.py
└── .env / Makefile / .env.example
```

---

## 3. 两层路由 + 5 条路径

```
normalize_input → load_short_term_context → route_task
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
      TOOL_ACT                 FALLBACK
          │                       │
    react_decide              respond
      ⇄ 工具执行 ⇄
    react_finish
          │
       respond
          │
    update_memory
              │
            END
```

| 路径 | LLM | VLM | 工具 | STM | LTM |
|------|-----|-----|------|-----|-----|
| TOOL_ACT (简单) | 1 | 0 | 0 | ✅ | ❌ |
| TOOL_ACT (+搜索) | 2-3 | 0 | 1-2 | ✅ | ❌ |
| TOOL_ACT (+图片) | 1-2 | 0-1 | 1 | ✅ | ❌ |
| FALLBACK | 0 | 0 | 0 | ✅ | ❌ |
> STM 每轮始终写入，LTM 由记忆门控控制。

---

## 4. AgentState (对齐 Schema V2)

- **TypedDict** + Reducer (追加型字段: `action_trace` / `observations` / `errors` / `model_calls` / `validation_results`)
- 节点返回 `dict` partial update，不返回下一节点名
- 路由由 `routers.py` conditional edge 完成
- `ModelCallRecord` 含 `prompt_tokens` / `completion_tokens` / `total_tokens`

---

## 5. Trace 系统 (对齐 AgentTrace_Schema)

| 表 | 用途 |
|----|------|
| `agent_trace_runs` | 一次请求的 Trace 总览 (trace_id / status / 耗时 / 调用计数) |
| `agent_trace_events` | 事件明细 (node_started / model_call_completed / route_decided ...) |

| API | 说明 |
|-----|------|
| `GET /api/v0/trace/{id}/events?after_seq=` | 事件列表 (支持断线重连) |
| `GET /api/v0/trace/{id}/timeline` | 前端时间线 |
| `GET /api/v0/trace/{id}/stream` | SSE 实时推送 |

每请求自动 `create_trace_run` → `complete_trace_run`，CLI / Streamlit 同步展示。

---

## 6. LLM / VLM

| Provider | 模型 | 用途 |
|----------|------|------|
| DeepSeek V4 | v4-flash | L2 路由 + 推理 |
| DeepSeek V4 | v4-pro | 高精度推理 |
| Kimi K2.6 | kimi-k2.6 | 推理 (temp=0.6, thinking=disabled) |
| GLM-5.1 | glm-5.1 | 推理 |
| llama.cpp | MiniCPM-V | VLM 图像理解 (127.0.0.1:8080/v1, ctx=4096) |

---

## 7. API 总览

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v0/chat` | 多轮对话 |
| `POST` | `/api/v0/upload` | 图片上传 |
| `GET/DELETE` | `/api/v0/memory/session/{id}` | 会话记忆 |
| `GET` | `/api/v0/memory/invoices` | 历史票据 |
| `GET/POST` | `/api/v0/memory/preferences` | 偏好 |
| `GET` | `/api/v0/health` | 健康检查 |
| `GET` | `/api/v0/trace/{id}/events` | Trace 事件 |
| `GET` | `/api/v0/trace/{id}/timeline` | Trace 时间线 |
| `GET` | `/api/v0/trace/{id}/stream` | Trace SSE |
| `POST` | `/api/v0/reset` | 重置 |

---

## 8. 启动

```bash
make          # 一键前后端
make backend  # :8000
make frontend # :8501
```
