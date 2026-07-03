# Nexa Agent / OfferCheck — 项目规格（SPEC）

> 版本：v3（2026-07-03） | 维护：本文件是项目的「宪法」，记录**核心目标 / 明确不做 / 关键决策与理由**。
> 每次重大方向或架构决策后必须回来更新；带 `TODO(补充)` 的小节留待推进时填。
> 配套文档：飞书《OfferCheck 参赛方案与开发计划》（wiki AUwFwSk3oiTf0ik12kgca9HZnVh）、
> [docs/agent_paradigms_and_hallucination.md](docs/agent_paradigms_and_hallucination.md)、
> [docs/run_20260702_234927_failure_analysis.md](docs/run_20260702_234927_failure_analysis.md)。

---

## 0. 一句话定义

**Nexa Agent** 是一个场景无关的**自主调查 Agent 引擎**（ReAct + Reflexion + 事实门控 + Eval Harness）；
**OfferCheck** 是它的第一个应用——**贯穿求职全程的「怀疑型研究管家」**，对一份工作机会主动联网调查、证伪，输出「靠谱 / 存疑 / 大概率有坑」+ 证据链 + 待用户自行确认事项。

---

## 1. 核心目标

### 1.1 统御原则（最高优先级，任何取舍先过这一关）

**一台引擎，两个出口：主线是 DeepSeek Harness 工程岗面试项目，比赛是顺手挤出的副产品。**
- 工程价值（给 DeepSeek 看）= 底层引擎：多工具编排 + 接地/验证 harness + Reflexion 自纠错 + Eval Harness + Trace 可观测。**这是主货币。**
- 比赛产物 = 同一台引擎套一层 OfferCheck 场景壳 + demo 视频，限时完成，绝不反客为主。
- 判据：任何功能若只增加「产品广度」而不增强「引擎工程深度或 demo 可展示性」，一律降级或不做。

### 1.2 产品目标（OfferCheck）

覆盖求职全链路的四阶段，**同一个自主调查引擎被投喂不同输入**（不是四个独立按钮）：

| 阶段 | 输入 | Agent 自主做什么 | 输出 |
|---|---|---|---|
| ① 选岗调研 | 简历 + 意向/候选公司 | 查业务真实性、融资健康度、团队背景、僵尸岗 | 优先级建议 + 依据 |
| ② 简历定向（小功能） | JD + 简历 | 定位差距，给「这几点突出/这几点补」 | 定向修改清单（附依据） |
| ③ 沟通证伪 | 招聘方消息/截图 | 核实身份、检测异常沟通（预付款/无面录用/紧迫感） | 红旗清单 + 建议 |
| ④ offer 证伪（核心） | offer/合同 + 全程上下文 | 最深度交叉验证、主动找反证 | 裁定 + 证据链 + 待确认项 |

**差异化内核**：竞品最多做「公司实体静态信用打分」（如求职方舟接企查查）；OfferCheck 做**这次具体机会本身**的主动、多源、持续证伪——覆盖「冒名顶替真公司」这一最高频诈骗模式（伪造 offer / 仿冒域名邮箱 / 冒充 HR）。

### 1.3 工程目标（引擎）

- 真 Agent：开放式、发现驱动的调查循环，路径随发现变，会撞死胡同并 Reflexion 绕路。
- 接地优先：结论必须绑定**真实检索到**的证据；宁可「存疑」，不可自信编造。
- 可观测 + 可量化：全程 Trace（查了什么/为什么/耗时/置信度）+ Eval Harness（准确率/误报率/拒答率/失败归因）。

---

## 2. 明确不做（Non-Goals）

> 这些是多轮迭代沉淀的硬边界。触碰前必须回本文件评估。

**产品边界**
1. **不做真实自动投递 / 海量海投**——数据证明群投失效（100+ 份面试率仅 2.58%），且违反 LinkedIn ToS、有封号风险。行业共识是「AI 辅助研究 + 定向，人来拍板」。
2. **不做多平台消息实时监听 / 自动化收发**——沟通阶段以用户主动粘贴消息/截图为输入，不接微信/WhatsApp 自动化（明令禁止 + 封号 + 监管雷区）。
3. **简历定向不做「一键生成新简历」全文重写**——只做定向建议清单（哪几点突出/补）。
4. **不做用户账号体系、支付、多租户**等产品化基建（比赛期用户数为 0）。
5. **C 端个人产品，不服务出海市场 / 不做跨境电商**——「出海」仅作英文化外壳（终端用户在海外），不是「帮别人出海」这件事。

**Agent 设计边界**
6. **不做固定 workflow**（如「拍照→查表→输出」三步定式 = 假 Agent，否决）。
7. **不做「简单调研报告生成器」**——输出必须是**裁定 / 决策 / 行动 / catch**（主动证伪，找坑），不是被动汇总。
8. **不依赖陌生领域知识库**——凡价值来自「查一个本人不懂的领域 KB」（3D 打印/医药/法律/电商辨假）一律否决；价值须来自通用调查推理 + 用户自身正在求职的真实场景。
9. **不做 coding agent**（市场饱和，且本人明确排除）——比赛产物是调查/证伪类 Agent。

**当前阶段技术不做（有明确触发条件后再上）**
10. **不引入 embedding / 向量检索**——现阶段记忆池 ≤10 条、LTM 数据为零、eval 15~20 条，Jaccard + 结构化实体键控足够；bge-m3 等自部署是过早优化。触发条件：LTM 积累 100+ 条真实反馈、需跨任务语义召回时再上。
11. **不用思考模型**（见 §3.2）——GMI 上思考协议坑无法规避，改用非思考 instruct 模型。
12. **不换 ReAct 范式**（见 §3.4）——幻觉与循环范式正交，换范式治不了幻觉。

---

## 3. 关键决策（含为什么）

### 3.1 架构：core + app 分层，nexa_agent 为可复用核心引擎
- **决策**：单仓库拆三层——`nexa_agent/`（headless 核心引擎，可 import/CLI，无 FastAPI/DB/UI）+ `offercheck/`（场景层）+ `server/`（唯一 FastAPI 后端）+ `web/`（Next.js 前端）。app/ 整体拆散：agent/tools/pipeline/streamlit 删除，llm→nexa_agent/llm，api/trace/storage/memory→server/。
- **为什么**：① nexa_agent 场景无关可复用，是「Model + Harness = Agent，harness 跨任务复用」最硬的 DeepSeek 叙事；② 旧 app/ 的 LangGraph ReAct 较弱，被 nexa_agent 的 ReAct+Reflexion 取代；③ 引擎干净暴露 trace/接口本身就是 harness 工程能力。

### 3.2 模型：GMI 全非思考模型
- **决策**：`strong=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`、`fast=deepseek-ai/DeepSeek-V3.2`、`upgrade=moonshotai/Kimi-K2-Instruct-0905`；视觉 `google/gemini-3.1-flash-lite/pro-preview`。全部非思考（instruct）。
- **为什么**：GMI 上 DeepSeek-V4 思考模型带来**无法规避**的协议坑——多轮 tool-calling 要求回传 `reasoning_content` 否则 400；reasoning-only 空响应（反思 2873 token 换 0 字）；且 GMI 不支持 `thinking` 参数（422），关不掉。实测非思考模型：零 400、零空响应、tool-calling 稳定。**幻觉问题不是靠思考模型解决的**（用思考模型那次同样失败），而是靠接地层（§3.3）。
- **可逆性**：纯 `config.py` 的 `MODEL_TIER` 三行 / 环境变量，无逻辑锁定。

### 3.3 防幻觉：接地层「纵深防御」四层（借鉴 Claude，非只加 prompt）
- **背景**：回归中模型零工具调用、编造整份调查报告（假 GitHub 仓库/招聘页），三层校验全放行还自判「成功」。诊断：光靠 prompt 会被模型无视（stage prompt 本就要求调研，仍被跳过）。
- **决策**（四层，见 [failure_analysis](docs/run_20260702_234927_failure_analysis.md)）：
  1. **接地铁律 prompt（软）**：react_system 顶部 4 条最高优先级 + 「怀疑型调查员」人设。
  2. **强制取证 gate（硬）**：裁定/事实型输出在零成功检索时被拦截（最多 2 次防死锁），逼先取证（no evidence, no answer）。
  3. **submit_verdict 结构化终止工具（硬）**：gate + 来源对账的落点。
  4. **AIS 来源对账（硬）**：`seen_urls` registry 收集真实见过的 URL；逐条核对 `[Source]` 引用的 URL/工具是否真出现过，臆造标 `⚠️[未验证]` 并降信。
- **为什么**：研究一致——幻觉根因是「弱接地 → 自信 best guess」，解法是「no evidence no answer + 来源归属(AIS) + 独立验证」；Claude 产品稳是「prompt + 工具设计 + 脚手架 + 训练」四层叠加，不是一句咒语。**这条路一箭三雕：治幻觉 = 强化产品核心「证伪」= 纯 harness 工程叙事。**

### 3.4 ReAct 范式与终止契约
- **决策 A**：保留 **ReAct（内循环）+ Reflexion（外循环）**，不换 Plan-Execute/ReWOO/LATS。
  - **为什么**：范式解决控制流/规划效率，与接地性正交；OfferCheck 本质是「求职诈骗垂直的 Deep Research Agent」，DRA 标配就是 ReAct 内核 + 引用溯源 + 独立验证——缺的是接地层不是范式。
- **决策 B**：**原生终止**——无 `tool_calls` 的实质文本即最终答案，弃用强制 `Final Answer:` 字符串哨兵（文本 ReAct 时代遗留反模式，制造无效 nag 浪费 ~40% token）；`submit_verdict` 工具作显式结构化终止。
  - **为什么**：原生 function calling 下「无 tool_calls = 完成」是行业约定（Claude Agent SDK / Strands / ai-sdk）；结构化终止工具解析 100% 可靠且与产品裁定 schema 同构。

### 3.5 搜索/检索
- **决策**：可插拔 provider 层（Tavily → 自建 SearXNG → Exa → DuckDuckGo 有序降级）+ 健康熔断 + per-provider 指标 + enrich 并行正文增强；web_fetch 走 Jina Reader → trafilatura 兜底。
- **为什么**：Tavily 免费额度有限，自建 SearXNG 兜底保证零成本可持续；provider 抽象 + 熔断 + 指标本身是 harness 工程展示点。
- **已知短板（待优化，见 §5）**：跨步/跨 Trial 无硬缓存去重、无检索溯源 registry 复用、trafilatura 环境漂移、登录墙域名(x.com 451)无兜底。

### 3.6 记忆
- **决策**：Reflexion 情景记忆（教训提取 + Jaccard 去重，池 ≤10）；跨阶段/跨 Trial 用 Scratchpad 结构化事实传参；服务态 STM/LTM 在 server/。**不用向量检索**（见 §2.10）。

### 3.7 前端与部署
- **决策**：全新 Next.js（web/），不复用旧 Streamlit；开发期 `/api/v0/*` 代理到 server 避 CORS；SSE 实时 trace 时间线 + 裁定卡。
- **为什么**：Streamlit 是内部工具观感、Trace 耦合旧 LangGraph；重做得消费级 + 可 Vercel 部署（比赛要求 ≥6 截图页面 + 体验链接），且倒逼引擎干净暴露 trace 接口。

### 3.8 Evaluator/Verifier 的 stage-aware 校准
- **决策**：`Evaluator.evaluate()` 和 `VerifierAgent.verify()` 均加 `stage` 参数；`stage4` 时使用 OfferCheck 专用评估/核查标准，由 `reflexion_agent.execute()` 透传。
- **为什么**：通用 LLM eval prompt 会把正确的 stage4 裁定（「靠谱/存疑/大概率有坑」+ 媒体报道来源）判为 wrong_reasoning 或 unreliable_source，导致 Trial 1 必然失败。根因是「通用事实精确性」标准与「调查类证伪产品」的证据链标准不同：offer 证伪允许媒体报道作为佐证，裁定本身的三态都是合法结论。stage-aware 分离了两套标准，不影响通用任务的严格评估。

---

## 4. 当前状态（2026-07-03）

**已建成 ✅**
- nexa_agent 核心引擎：ReAct（原生 tool calling）+ Reflexion + Verifier + Evaluator + Eval Harness + 12 工具 + 搜索 provider 层 + trace `on_event` 钩子
- GMI 全非思考模型接入 + 端到端验证；reasoning_content 400 / extra_body 422 修复
- 接地层四件套（接地铁律 / 强制取证 gate / submit_verdict / AIS 对账）——回归验证幻觉路径被堵死
- server：`/run_stage` + `/run_stage/stream`(SSE) 瘦调用引擎 + SSE keepalive（20s 心跳，防 proxy 断流）；web：Next.js Walking Skeleton
- offercheck/ 骨架：stage1/stage4 prompt + domain_whois_lookup + 裁定标签
- **Evaluator 校准（stage-aware）**：`evaluate()` / `_build_llm_eval_prompt()` 加 `stage` 参数；stage4 使用 OfferCheck 专用评估标准（三态裁定 + 证据链 + red_flags），LLM 评估结果记 INFO 日志；stage4 端测 trial 1 成功（置信度 90%）
- **Verifier 校准（全量完成）**：stage-aware 来源标准（stage4 宽松媒体来源 + 放行规则）；**CoVe factored**：`_llm_verify` 改为逐条事实独立评判（单次 LLM，per-fact JSON 输出）+ `_parse_cove_response` 解析；**三态 VerdictResult**：`status: "verified"|"unverified"|"failed"` 替代 `passed: bool`（`.passed` 属性保持向后兼容）；端测 status=verified，置信度 90%
- **submit_verdict 步数预警**：在剩余 ≤1 步时向对话注入一次 user 提示，催 agent 调用 `submit_verdict` 而非触发兜底汇总；端测从"12步触发兜底"改善为"10步主动 submit_verdict"

**未建成 / 规划中 ⬜**（详见 §5）
- offercheck stage2/stage3 prompt、company_registry_search 工具、eval_suite 标注案例集
- Verifier CoVe 化 + 评估器校准
- 检索缓存/去重/溯源 registry 优化
- 前端四阶段完整页面 + 部署（Vercel）+ GMI 调用证明 + demo 视频

---

## 5. 待后续推进时补充（TODO）

> 推进到对应项时，把设计细节 / 决策理由回填到这里，并更新 §3 / §4。

- **Verifier 校准（P6）**：✅ 已完成。stage-aware 标准 + CoVe factored 逐条评判 + 三态 VerdictResult。详见 §4。
- `TODO(补充)` **检索工具优化（P0/P1）**：跨步硬缓存去重 + 检索溯源 registry（同时服务 AIS）；trafilatura 装齐 + 登录墙域名策略；结果去重/轻量重排；Corrective 再检索。
- `TODO(补充)` **offercheck eval_suite**：15~20 个真实+模拟诈骗/正常案例（带 expected_verdict）+ 裁定级评分（准确率/误报率/拒答率）；接入 Eval Harness 回归门禁。
- `TODO(补充)` **stage2/stage3 场景层**：简历定向 prompt；沟通证伪 prompt + company_registry_search。
- `TODO(补充)` **前端四阶段 + 部署**：选岗/简历/沟通/offer 页面；Vercel + 后端部署；≥6 截图；3 分钟 demo 视频；GMI 调用证明素材。
- `TODO(补充)` **few-shot 正例**：纵深防御中间层（原生 tool-calling 下的正确「先查证再裁定」示范）。
- `TODO(补充)` **DeepSeek 面试补口径**：7/6 提交后另花 1–2 天加 read_file/edit_file/run_shell + 「读 repo→改 bug→跑 pytest→Reflexion 重试」coding demo（只为补「代码味道」，不进比赛）。

---

## 6. 运行速查

```bash
# 环境：conda agent（Python 3.10；base 的 3.9 会因 pydantic union 报错）
conda activate agent

# CLI（核心引擎，无需起服务）
python -m nexa_agent.reflexion_agent "…" --stage stage4   # 或 make agent Q=… STAGE=stage4

# 全栈
python -m uvicorn server.main:app --port 8000 --reload      # 后端
cd web && npm run dev                                        # 前端 :3000

# 评测 / 测试
python -m nexa_agent.eval_harness run --suite <cases.jsonl>
python -m pytest tests/ -q
```

模型路由唯一真源：`nexa_agent/config.py` 的 `MODEL_TIER` + `MODEL_ROUTING`。
