# Nexa Agent

> 场景无关的自主调查 Agent 引擎（ReAct + Reflexion），OfferCheck 是它的第一个应用。

---

## 简介

Nexa Agent 分三层：**核心引擎（headless）→ 场景层 → 服务与前端**。

- **`nexa_agent/`** — 可复用核心引擎（无 FastAPI/DB/UI，可直接 import 或 CLI 运行）：ReAct 原生 tool calling + Reflexion 外循环 + Verifier 事实门控 + Evaluator + Eval Harness + 12 个工具 + 模型分层路由。
- **`offercheck/`** — 求职机会防坑场景层（建在核心之上，通过注入 stage 任务定义区分阶段）。
- **`server/`** — FastAPI 后端（瘦转发核心 + SSE 流式 trace + STM/LTM + Trace 持久化）。
- **`web/`** — Next.js 前端（实时调查轨迹时间线 + 裁定卡）。

```
调查一次的核心循环（nexa_agent）：
  ReflexionReActAgent.execute(task, stage, on_event)
    ├─ Trial 1~N（Reflexion 外循环）
    │   ├─ Scratchpad 已确认事实注入 + 历史教训注入
    │   ├─ react_loop() — 原生 tool calling
    │   │   ├─ 12 个工具：web_search / read_pdf / calculator / domain_whois_lookup / ...
    │   │   ├─ 中途纠偏：URL 去重 + loop 检测 + 策略切换
    │   │   ├─ 分档 Observation 截断（≤15K 零截断）
    │   │   └─ 动态升级：连续 2 步未发 tool_calls → 切备援模型
    │   ├─ Evaluator：结果优先两阶段评估
    │   ├─ Verifier：事实核查网关（触发条件满足时）
    │   └─ 失败 → 反思 + 教训提取 + Scratchpad 事实提取
    └─ 返回最佳答案（on_event 全程发射结构化事件供 SSE 流式推送）
```

---

## 核心能力

**引擎（`nexa_agent/`）**
- ReAct + Reflexion 双循环：局内 tool calling + 局外自我反思重试
- 原生 Function Calling（OpenAI 兼容），消灭正则解析失败
- 12 个工具：`web_search` / `wikipedia_search` / `web_fetch` / `tavily_extract` / `read_pdf` / `read_xlsx` / `calculator` / `analyze_image`（端侧 MiniCPM-V）/ `analyze_image_cloud`（云端 Gemini 3.1）/ `save_content` / `get_current_time` / `domain_whois_lookup`
- 模型分层路由：strong（首步规划）+ fast（后续执行）+ upgrade（tool-call 备援，动态升级）
- 中途纠偏：URL 级跨工具去重 + loop 早期检测 + 策略切换
- Scratchpad 跨 Trial 事实传递 + Reflexion 教训提取
- Verifier 事实网关：结论绑定 `[Fact]/[Source]/[Confidence]` 证据
- 结构化 trace 发射钩子（`on_event`）：工具/信源/步骤/裁定事件流，供服务层转 SSE
- Eval Harness：回归评测流水线 + failure mode 归因 + 运行对比
- 可插拔搜索层：Tavily → 自建 SearXNG → Exa → DuckDuckGo 有序降级 + 熔断

**场景层（`offercheck/`）**
- 求职机会防坑：输入 offer/JD/公司名 → 自主联网调查 + 主动证伪 → 裁定「靠谱/存疑/大概率有坑」+ 红旗证据链 + 待确认项
- 已就绪：stage1（选岗调研）/ stage4（offer 证伪）prompt + `domain_whois_lookup` + 裁定标签 `[Verdict]/[RedFlag]/[NeedUserConfirm]`
- 规划中：stage2（简历定向）/ stage3（沟通证伪）/ `company_registry_search` / eval_suite

**服务与前端（`server/` + `web/`）**
- FastAPI 瘦后端：`/run_stage`（阻塞）+ `/run_stage/stream`（SSE 实时 trace）
- STM 会话上下文 + LTM 长期记忆（增删改查 API）
- Trace 持久化 + 查询 + SSE
- Next.js 前端：阶段选择 + 实时调查轨迹时间线 + 裁定卡

---

## 快速开始

要求：Python `3.10+`（conda 环境推荐），Node `18+`（前端）

```bash
git clone <repo-url> && cd Nexa_Agent
cp .env.example .env   # 填入 GMI_API_KEY（或 DEEPSEEK_API_KEY）、TAVILY_API_KEY
pip install -r requirements.txt
```

### CLI（核心引擎，无需起服务）

```bash
# 通用问答
python -m nexa_agent.reflexion_agent "北京到上海的直线距离是多少？"

# OfferCheck 阶段（stage1=选岗调研 | stage4=offer 证伪）
python -m nexa_agent.reflexion_agent "帮我核实这个远程 offer 是否靠谱：..." --stage stage4

# 从文件读取
python -m nexa_agent.reflexion_agent --file question.txt --stage stage4

# 或经 Makefile
make agent Q="帮我证伪这个 offer..." STAGE=stage4
```

### 全栈（后端 + 前端）

```bash
# 后端 (:8000) — 开发期务必带 --reload，改引擎代码后自动重启（否则跑的是旧代码）
python -m uvicorn server.main:app --port 8000 --reload
# 或 make backend（含 llama.cpp VLM，已内置 reload）

# 前端 (:3000) — 另开终端
cd web && npm install && npm run dev
```

打开 http://localhost:3000 ，粘贴 offer / JD / 公司名 → 选阶段 → 实时看调查轨迹 + 裁定。

---

## 架构

```
nexa_agent/                    可复用核心引擎（headless）
├── react_agent.py            ReAct 主循环（原生 tool calling + on_event 发射钩子）
├── reflexion_agent.py        Reflexion 外循环（Trial → Evaluate → Verify → Reflect）
├── evaluator.py              结果优先混合评估器（启发式 + LLM Judge）
├── verifier.py               事实核查网关 + OfferCheck 裁定解析
├── tools.py                  12 个工具
├── memory.py                 Reflexion 情景记忆（教训提取 + Jaccard 去重）
├── eval_harness.py           系统化评测流水线（多维评分 + 回归对比）
├── config.py                 模型分层路由 + 超参数（含 GMI/DeepSeek provider）
├── logger.py                 per-run 独立日志
├── llm/                      LLM 客户端层（多 provider，OpenAI 兼容）
├── trace/                    trace 事件 schema
├── search/                   可插拔搜索 provider 层（router/providers/enrich）
├── prompts/                  System Prompt + Reflection + OfferCheck stage1/stage4
└── eval_suites/              评测套件（GAIA 回归子集 ID）

offercheck/                    场景层（建在核心上，当前为骨架）
├── stages/  tools/  eval_suite/

server/                        FastAPI 后端
├── api/                       路由（run_stage / memory / trace；chat/upload 待接入）
├── trace_store/               Trace 持久化 + SSE 推送
├── persistence/               SQLAlchemy
├── memory/                    STM + LTM
├── config.py  main.py

web/                           Next.js 前端
└── app/                       page.tsx（SSE 流式 UI）+ layout + globals.css
```

### API

| 路由 | 说明 | 状态 |
|------|------|------|
| `POST /api/v0/run_stage` | 执行 OfferCheck 阶段（瘦调用核心，阻塞返回） | ✅ |
| `POST /api/v0/run_stage/stream` | 同上，SSE 实时推送调查轨迹事件 | ✅ |
| `GET /api/v0/health` | 后端 + 引擎模型链路健康状态 | ✅ |
| `GET /api/v0/trace/{id}/events` | Trace 事件明细 | ✅ |
| `GET /api/v0/trace/{id}/timeline` | 前端时间线 | ✅ |
| `GET /api/v0/trace/{id}/stream` | Trace SSE | ✅ |
| `GET/DELETE/PATCH /api/v0/memory/ltm` | LTM 记忆管理 | ✅ |
| `POST /api/v0/chat` | 旧对话端点 | 501（待接入核心） |
| `POST /api/v0/upload`、`/files/analyze` | 旧上传/预识别端点 | 501（待接入核心） |

### LLM / VLM 支持

比赛期主推理经 **GMI Cloud Inference Engine**（OpenAI 兼容）承载；无 GMI key 时回落 DeepSeek 官方 API。

| Provider / 模型 | 层级 |
|------|------|
| GMI · `deepseek-ai/DeepSeek-V4-Pro` | strong（首步规划） |
| GMI · `deepseek-ai/DeepSeek-V4-Flash` | fast（后续执行、反思、评估） |
| GMI · `Qwen/Qwen3.6-35B-A3B` | upgrade（tool-call 备援，动态升级） |
| GMI · `google/gemini-3.1-flash-lite-preview` | vision（云端普通图片） |
| GMI · `google/gemini-3.1-pro-preview` | vision（复杂/模糊图，空响应自动升级） |
| DeepSeek 官方 API | 备用 provider（无 GMI key 时） |
| Kimi K2.6 | 云端视觉回落（无 GMI key 时） |
| llama.cpp / MiniCPM-V | 端侧图片分析（`analyze_image`） |

> **动态升级**：ReAct 循环中连续 2 步未发 `tool_calls` 时，自动从 fast 切至 upgrade 层（复杂 function schema 下的 tool-call 可靠性兜底），恢复正常后回落。
>
> 模型选择的唯一真源是 `nexa_agent/config.py`（`MODEL_TIER` + `MODEL_ROUTING`），可用环境变量覆盖。

### Eval Harness

```bash
# 自定义 JSONL 套件评测
python -m nexa_agent.eval_harness run --suite path/to/cases.jsonl

# 分析结果 / 对比两次运行
python -m nexa_agent.eval_harness analyze --input results/eval_xxx.jsonl
python -m nexa_agent.eval_harness compare --baseline run_A.jsonl --current run_B.jsonl
```

> GAIA 套件（`--suite gaia_l1`）需本地放置 GAIA 数据集于 `GAIA/2023/validation/`（默认不含）。

---

## 文档

| 文档 | 说明 |
|------|------|
| [DEVELOPMENT_PLAN.md](docs/DEVELOPMENT_PLAN.md) | 开发规划 |
| [AgentTrace_Schema.md](docs/AgentTrace_Schema.md) | Trace 协议 |
| [Short-Term_Memory_Schema.md](docs/Short-Term_Memory_Schema.md) | 短期记忆协议 |
| [Long-Term_Memory_Schema.md](docs/Long-Term_Memory_Schema.md) | 长期记忆协议 |
