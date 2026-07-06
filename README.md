# OfferCheck — Job Offer Due-Diligence Agent

> A skeptical research copilot for job seekers. Paste an offer, JD, company name, recruiter message, or screenshot — it independently investigates that **specific opportunity** on the live web and returns a verdict of **Looks Legit / Suspicious / Likely a Scam**, with a source-verified evidence chain and items for you to confirm yourself.

**Live demo**: https://offercheck.up.railway.app/ — try: *“I received an offer from apple-hiring-team.com asking for a $200 gift-card background-check fee — is this legit?”*

- **Actively falsifies, not statically scores** — targets the highest-frequency scam pattern: impersonation of real companies (forged offers, lookalike domains, fake HR). Competitors return a static company credit score; OfferCheck investigates *this* opportunity, multi-source, and links every conclusion to sources it actually retrieved.
- **No evidence, no verdict** — a four-layer grounding stack (grounding rules → mandatory-evidence gate → structured `submit_verdict` → source-attribution audit) blocks hallucinated verdicts by design.
- **One engine, full journey** — role research → resume fit → recruiter-message check → offer verification run on a single autonomous investigation engine (per-stage prompts), with cross-stage memory carry-over and in-conversation capability routing.
- **Powered by GMI Cloud** — DeepSeek-V4 Pro/Flash with layered thinking control, Kimi-K2 tool-call fallback, Gemini 3.1 vision OCR — routed per role by a model router.

<p align="center">
  <img src="assets/architecture.svg" alt="OfferCheck architecture: multimodal input → four stages (one engine, different stage prompts) → grounded ReAct + Reflexion investigation loop → three-state verdict with evidence" width="760">
</p>

*以下为中文详细文档 · Detailed docs in Chinese below*

---

## 四阶段（同一引擎，不同 stage prompt）

| 阶段 | 输入 | Agent 自主做什么 | 输出 |
|---|---|---|---|
| ① 选岗调研 | 简历 + 意向/候选公司 | 查业务真实性、融资健康度、团队背景、僵尸岗 | 优先级建议 + 依据 |
| ② 简历定向 | JD + 简历 | 定位差距，给「这几点突出/这几点补」 | 定向修改清单（附依据） |
| ③ 沟通证伪 | 招聘方消息/截图 | 核实身份、检测异常沟通（预付款/无面录用/紧迫感） | 红旗清单 + 建议 |
| ④ offer 证伪（核心） | offer/合同 + 全程上下文 | 最深度交叉验证、主动找反证 | 裁定 + 证据链 + 待确认项 |

**接地优先，宁可存疑不可编造**：结论必须绑定**真实检索到**的证据。四层纵深防御堵死幻觉——接地铁律 prompt → 强制取证 gate（no evidence, no verdict）→ `submit_verdict` 结构化终止 → AIS 来源对账（臆造来源标 `⚠️[未验证]`）。

---

## 底层引擎 Nexa Agent

场景无关、可复用的调查内核（headless，无 FastAPI/DB/UI，可直接 import 或 CLI 运行）：

- **ReAct + Reflexion 双循环**：局内原生 tool calling（OpenAI 兼容，消灭正则解析失败）+ 局外自我反思重试（Trial → Evaluate → Verify → Reflect）
- **12 个工具**：`web_search` / `wikipedia_search` / `web_fetch` / `tavily_extract` / `read_pdf` / `read_xlsx` / `calculator` / `analyze_image`（端侧 MiniCPM-V）/ `analyze_image_cloud`（云端 Gemini 3.1）/ `save_content` / `get_current_time` / `domain_whois_lookup`
- **模型分层路由**：strong（首步规划/裁定）+ fast（后续执行/反思/评估）+ upgrade（tool-call 备援，动态升级）
- **中途纠偏**：URL 级跨工具去重 + loop 早期检测 + 策略切换 + 分档 Observation 截断
- **Verifier 事实网关**：结论绑定 `[Fact]/[Source]/[Confidence]`，stage-aware 校准 + CoVe 逐条核查
- **Eval Harness**：回归评测流水线 + failure-mode 归因 + 运行对比（`compare` 2pp 门禁即回归 gate）
- **可观测性**：结构化 trace 发射钩子（`on_event`）→ 工具/信源/步骤/裁定事件流，服务层转 SSE
- **可插拔搜索层**：Tavily → 自建 SearXNG → Exa → DuckDuckGo 有序降级 + 健康熔断

```
调查一次的核心循环（nexa_agent）：
  ReflexionReActAgent.execute(task, stage, on_event)
    ├─ Trial 1~N（Reflexion 外循环）
    │   ├─ Scratchpad 已确认事实注入 + 历史教训注入
    │   ├─ react_loop() — 原生 tool calling + 接地取证 gate
    │   ├─ Evaluator：结果优先两阶段评估（stage-aware）
    │   ├─ Verifier：事实核查网关（触发条件满足时）
    │   └─ 失败 → 反思 + 教训提取 + Scratchpad 事实提取
    └─ 返回最佳答案（on_event 全程发射结构化事件供 SSE 流式推送）
```

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
# OfferCheck 阶段（stage1=选岗 | stage2=简历 | stage3=沟通 | stage4=offer 证伪）
python -m nexa_agent.reflexion_agent "帮我核实这个远程 offer 是否靠谱：..." --stage stage4

# 从文件读取 / 或经 Makefile
python -m nexa_agent.reflexion_agent --file question.txt --stage stage4
make agent Q="帮我证伪这个 offer..." STAGE=stage4

# 通用问答（不带 --stage 即纯引擎）
python -m nexa_agent.reflexion_agent "北京到上海的直线距离是多少？"
```

### 全栈（后端 + 前端）

```bash
# 后端 (:8000) — 开发期务必带 --reload，改引擎代码后自动重启（否则跑的是旧代码）
python -m uvicorn server.main:app --port 8000 --reload

# 前端 (:3000) — 另开终端
cd web && npm install && npm run dev
```

打开 http://localhost:3000 ，粘贴 offer / JD / 公司名 / 截图 → 选阶段 → 实时看调查轨迹 + 裁定。

---

## 架构

```
nexa_agent/                    可复用核心引擎（headless）
├── react_agent.py            ReAct 主循环（原生 tool calling + on_event 发射钩子）
├── reflexion_agent.py        Reflexion 外循环（Trial → Evaluate → Verify → Reflect）
├── evaluator.py              结果优先混合评估器（启发式 + LLM Judge，stage-aware）
├── verifier.py               事实核查网关 + OfferCheck 裁定解析
├── tools.py                  12 个工具
├── memory.py                 Reflexion 情景记忆（教训提取 + Jaccard 去重）
├── eval_harness.py           系统化评测流水线（裁定级 + 关键词召回 + 回归对比）
├── config.py                 模型分层路由 + 超参数（含 GMI/DeepSeek provider）
├── llm/  trace/  search/     LLM 客户端 · trace schema · 可插拔搜索层
├── prompts/                  System Prompt + Reflection + OfferCheck stage1~stage4
└── eval_suites/              评测套件（GAIA 回归子集 ID）

offercheck/                    场景层（建在核心上）
├── stages/  tools/           场景占位（prompt/工具现集中在 nexa_agent/）
└── eval_suite/               cases.jsonl（32 例四阶段评测集，含 6 条全英文）

server/                        FastAPI 后端（瘦转发核心）
├── api/                       路由（run_stage / upload / memory / trace）
├── trace_store/               Trace 持久化 + SSE 推送
├── persistence/  memory/      SQLAlchemy · STM + LTM

web/                           Next.js 前端
└── app/                       page.tsx（三栏 SSE 流式 UI）+ ui.tsx + layout + icon.svg
```

### API

| 路由 | 说明 | 状态 |
|------|------|------|
| `POST /api/v0/run_stage` | 执行 OfferCheck 阶段（瘦调用核心，阻塞返回） | ✅ |
| `POST /api/v0/run_stage/stream` | 同上，SSE 实时推送调查轨迹事件 | ✅ |
| `POST /api/v0/upload` | 截图/PDF 上传（引擎运行时内联 OCR） | ✅ |
| `GET /api/v0/health` | 后端 + 引擎模型链路健康状态 | ✅ |
| `GET /api/v0/trace/{id}/{events,timeline,stream}` | Trace 明细 / 时间线 / SSE | ✅ |
| `GET/DELETE/PATCH /api/v0/memory/ltm` | LTM 记忆管理 | ✅ |
| `POST /api/v0/files/analyze` | 图片预识别缓存（引擎已内联 OCR，暂未接） | 501 |

### 模型链路（LLM / VLM）

比赛期主推理经 **GMI Cloud Inference Engine**（OpenAI 兼容）承载；无 GMI key 时回落 DeepSeek 官方 API。经 `extra_body={"enable_thinking": bool}` **分层控制思考**——fast 层（Flash）关思考保证快/省与稳定 tool-calling，strong 层（Pro）保留思考用于首步规划与裁定；多轮 tool-calling 的 `reasoning_content` 回传由引擎处理，实测零 400、首步产出 tool_calls 而非空响应。

| Provider / 模型 | 层级 |
|------|------|
| GMI · `deepseek-ai/DeepSeek-V4-Pro` | strong（首步规划、裁定；保留思考） |
| GMI · `deepseek-ai/DeepSeek-V4-Flash` | fast（后续执行、反思、评估、教训提取；关思考） |
| GMI · `moonshotai/Kimi-K2-Instruct-0905` | upgrade（tool-call 备援，动态升级） |
| GMI · `google/gemini-3.1-flash-lite-preview` / `pro-preview` | vision（云端图片，空响应自动升级） |
| DeepSeek 官方 API · Kimi 多模态 · llama.cpp/MiniCPM-V | 无 GMI key 时的回落 / 端侧 |

> **动态升级**：ReAct 循环中连续 2 步未发 `tool_calls` 时，自动从 fast 切至 upgrade 层（复杂 function schema 下的 tool-call 可靠性兜底），恢复后回落。模型选择唯一真源是 `nexa_agent/config.py`（`MODEL_TIER` + `MODEL_ROUTING`），可用环境变量覆盖。

### Eval Harness

```bash
# OfferCheck 四阶段评测集（32 例，含 6 条全英文；裁定级准确率/误报率/漏报率 + stage2 关键词召回）
python -m nexa_agent.eval_harness run --suite offercheck

# 分析结果 / 对比两次运行（compare 的 2pp 门禁即回归 gate）
python -m nexa_agent.eval_harness analyze --input results/eval_xxx.jsonl
python -m nexa_agent.eval_harness compare --baseline run_A.jsonl --current run_B.jsonl
```

> GAIA 套件（`--suite gaia_l1`）需本地放置 GAIA 数据集于 `GAIA/2023/validation/`（默认不含）。

---

项目「宪法」（核心目标 / 明确不做 / 关键决策与理由）见 [SPEC.md](SPEC.md)。
