# Nexa Agent

> LangGraph 原生多路径 Agent — V0: 发票 / 图片 / 文档识别与问答

---

## 简介

Nexa Agent V0 是一个 **LangGraph 原生 ReAct Agent 系统**，所有请求统一走 TOOL_ACT (ReAct) 路径，LLM 自行判断是否需要调工具。专注于多模态票据/图片/文档识别与问答。

```
所有请求 → route_task → TOOL_ACT (ReAct)
  react_decide (LLM 规划) ⇄ execute_tool (工具执行) → finish
  工具: web_search / wikipedia / calculator / time / analyze_image (VLM)
```

### 核心能力

- 多模态输入：文本 + 图片（JPG / PNG / PDF）
- 上传后图片预识别缓存：`POST /api/v0/files/analyze` + `image_analysis_cache`
- Cached Image QA：同会话追问复用 `active_file.vlm_text`，避免重复调用 VLM
- 票据字段结构化提取：JSON 输出 + 基础校验
- ReAct Agent 工具集：web_search / wikipedia / calculator / analyze_image / tavily_extract / save_content
- STM 会话上下文 + LTM 长期记忆 + KB 知识库分层检索
- Trace 可视化：Trace Events / Timeline / SSE
- Streamlit Chat UI

---

## 快速开始

要求：Python `3.10.18`

```bash
git clone <repo-url> && cd Nexa_Agent
cp .env.example .env   # 编辑填入 API Key
pip install -r requirements.txt
make                   # 一键启动前后端
```

- 后端 → `http://localhost:8000/docs`
- 前端 → `http://localhost:8501`

### CLI

```bash
python -m app.cli -m "你好"
python -m app.cli -i invoice.jpg -m "金额多少"
python -m app.cli -b kimi -m "是否可以报销" -i invoice.jpg
```

---

## 架构

```
app/
├── agent/          LangGraph ReAct Agent (TOOL_ACT + FALLBACK)
├── tools/          ReAct 工具 (web_search/wikipedia/calculator/time/analyze_image/tavily_extract/save_content)
├── trace/          Trace 事件系统 (SSE + Timeline + SQLAlchemy)
├── storage/        持久化层 (SQLAlchemy: trace + LTM + KB)
├── llm/            DeepSeek V4 / Kimi K2.6 / GLM-5.1
├── pipeline/       llama.cpp VLM (MiniCPM-V) + 提取管线
├── api/            FastAPI (chat / upload / files/analyze / memory / trace)
├── memory/         STM Schema + LTM Schema
```

### 执行路径

| 路径 | 场景 | LLM | VLM | 工具 |
|------|------|-----|-----|------|
| TOOL_ACT | 全部请求 (ReAct Agent) | 1-6 | 0-1 | 0-5 |
| FALLBACK | 异常兜底 | 0 | 0 | 0 |
> ReAct 循环最多 6 步，LLM 自行判断是否需要调工具

### API

| 路由 | 说明 |
|------|------|
| `POST /api/v0/chat` | 多轮对话，创建 AgentState 并执行 LangGraph |
| `POST /api/v0/upload` | 上传图片/PDF，返回 `file_id`、`file_sha256`、服务端路径 |
| `POST /api/v0/files/analyze` | 上传后图片预识别，写入/命中 `image_analysis_cache` |
| `GET /api/v0/trace/{id}/events` | Trace 事件明细 |
| `GET /api/v0/trace/{id}/timeline` | 前端时间线 |
| `GET /api/v0/trace/{id}/stream` | Trace SSE |
| `GET/DELETE/PATCH /api/v0/memory/ltm` | LTM 记忆管理（查看/遗忘/修改） |
| `GET /api/v0/health` | 后端、LLM、VLM 健康状态 |

### LLM / VLM 支持

| Provider | 模型 |
|----------|------|
| DeepSeek V4 | deepseek-v4-flash / deepseek-v4-pro |
| Kimi K2.6 | kimi-k2.6 |
| GLM-5.1 | glm-5.1 |
| VLM | llama.cpp / MiniCPM-V（OpenAI 兼容 API） |

### Trace

每请求自动记录 `agent_trace_runs` / `agent_trace_events`，CLI / Streamlit 展示节点路径、状态、耗时和模型调用摘要。

图片预识别与缓存复用不新增协议外 TraceEventType：

- 上传后预识别：`model_call_completed`，`payload.purpose="image_analysis_precompute"`
- Agent 节点复用缓存：`ActionTraceItem.action="use_cached_image_analysis"`，由 `chat.py` 派生为 `node_completed`

### Streamlit UI

- 主 Chat Panel 支持真实点击上传和拖拽上传
- 上传成功后自动触发图片预识别
- active file 在当前会话内持续作为图片上下文
- 用户消息中的图片缩略图会保留在历史消息里
- 助手回答正文使用 Streamlit 原生 `st.markdown()` 渲染，支持加粗、列表、代码和段落

---

## 文档

| 文档 | 说明 |
|------|------|
| [DEVELOPMENT_PLAN.md](docs/DEVELOPMENT_PLAN.md) | 三阶段开发规划 |
| [TECHNICAL_ARCHITECTURE.md](docs/TECHNICAL_ARCHITECTURE.md) | 技术架构 |
| [AgentState_SchemaV2.md](docs/AgentState_SchemaV2.md) | 状态协议 |
| [AgentTrace_Schema.md](docs/AgentTrace_Schema.md) | Trace 协议 |
| [Short-Term_Memory_Schema.md](docs/Short-Term_Memory_Schema.md) | 短期记忆协议 |
| [Long-Term_Memory_Schema.md](docs/Long-Term_Memory_Schema.md) | 长期记忆协议 |
| [SPEC.md](SPEC.md) | 项目规范与当前实现约束 |
