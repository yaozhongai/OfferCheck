# Nexa Agent / OfferCheck — 项目规格（SPEC）

> 版本：v4（2026-07-03） | 维护：本文件是项目的「宪法」，记录**核心目标 / 明确不做 / 关键决策与理由**。
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

### 3.7·补 前端信息架构矫正（2026-07-04）：三栏各归其位
- **背景**：初版把「过程 trace」和「产物（裁定/证据/来源）」全堆右侧画布，对话区退化成「详情见右侧 →」的指针；左栏是四阶段功能菜单，视觉上把产品做成「四个独立按钮」，与 §1.2「同一引擎不同输入、非四个按钮」的叙事自相矛盾；界面中文主导，与「英文优先/出海」冲突。同类产品（Manus/MiniMax/Kimi/Perplexity/Claude/Scamio）已收敛出明确分工。调研与方案见飞书《OfferCheck 产品形态调研与界面矫正方案》(docx `IOCgdc1teolYw6xEEkWc7888nPc`) 与 [docs/产品界面改造执行计划.md](docs/产品界面改造执行计划.md)。
- **决策**（三栏）：① **中栏对话流**承载全过程——trace 以 `InlineTrace` 折叠聚合条内联（running 实时一行 / done 汇总条可展开完整步骤），裁定摘要气泡随后；② **右栏 Evidence Board** 只放产物（裁定卡/红旗/事实/来源/待确认），空态折叠为窄条、有结论才展开；③ **功能入口下沉**为 composer 顶部 stage chips（Claude 式）+ 数据驱动精简引导字段 + 「Skip, just type」自由文本模式；左栏改为 Case/旅程导向（P1 落 Case 模型）。全界面英文优先。
- **红线**：矫正只动 `web/` 渲染层——`reduceRun`/`runSSE`/SSE 事件 schema、`parseStructuredAnswer`/`detectVerdict` 的中文裁定关键词匹配（引擎契约）一律冻结；英文化只改 chrome，`VERDICT_STYLES` 的中文 key 保留。（注：冻结指不改既有事件语义；2026-07-05 的 `stage_routed` 是 additive 新事件、`answer_delta`/`retry` 同理，不违反本红线。）
- **为什么**：过程归对话、产物归画布是 agentic 产品已验证的主导范式；Evidence Board 是相对 Scamio 类「只给 verdict 不给证据」竞品的差异化落点；Case 化同时承载「选岗→沟通→offer 全链路」完整度叙事与「单次证伪→求职季订阅」商业闭环。

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
- **Evaluator 去 confidence**：`EvalResult` 删除 `confidence: float` 字段，LLM 只输出 `success/reason/feedback/failure_mode`；消除 LLM 自评置信度带来的虚假精度
- **Evaluator 轨迹截断修复**：`react_loop` 所有返回点附加 `action_history`/`seen_urls`/`successful_retrievals`；Evaluator 新增 `_build_action_summary()` 将行动日志转为紧凑结构文本（工具调用逐步列出 + 已访问域名 + 成功检索率），彻底替代截断后的 trajectory 字符串；LLM Judge 现在能看到调查全程，不再受中间步骤丢失影响
- **login-wall 黑名单校正**：移除 x.com/twitter.com（GFW 屏蔽而非登录墙，海外用户可正常访问），保留真正需要登录的 linkedin/facebook/instagram
- **前端信息架构矫正 P0（2026-07-04，见 §3.7·补）**：`web/app/page.tsx` 三栏重排完成——① trace 内联对话流（`InlineTrace`，删除右栏 `TracePanel`）；② 右栏 Evidence Board（只放产物，空态折叠窄条、有结论展开）；③ composer stage chips + 数据驱动精简引导字段 + Skip-just-type 自由文本（`buildInput` raw 模式）；④ 全界面英文化（`UI` 文案常量，裁定英文主中文辅注）。引擎契约冻结（`reduceRun`/SSE schema/中文裁定关键词未动）；`tsc --noEmit` 通过 + 浏览器验证三栏/切 stage/raw 切换/错误路径下 board 展开均正常
- **前端信息架构矫正 P1（2026-07-04）**：左栏 Case 化——`cases[]`+`activeCaseId` 替换 `stage/forms/stageStates`，`patchActiveCase` 稳定更新器；每 Case = 一次求职机会（含 forms + 四阶段 stageStates + 跨阶段记忆容器），侧栏显示 Case 列表 + 每 Case 四阶段进度点 + New/重命名/删除；Case 名从 stage1 company 带出（raw/空为 Untitled）；`localStorage` 持久化（水合时 running→error 清洗、配额兜底剥离 trace 明细、确定性默认 case 避免 SSR 水合不一致）；Hero 空态（fresh case 显示）。浏览器验证：多 case 创建/切换/删除、per-case 运行态还原、刷新持久化、名称派生均正常
- **前端信息架构矫正 P2（2026-07-04）**：**T7 内联引用↔来源联动**——`ChatSummary` 加 `Cited: [n]` 引用行、Evidence Board 事实行加 `[n]` 引用标记、来源列表加编号锚点（`${anchorId}-src-${i}`）；点击引用复用 `jumpToAnchor` 滚动右栏并高亮对应来源（Perplexity 式）。**T8 组件拆分**——`page.tsx`（1781 行）按类型抽出 `app/ui.tsx`（875 行：全部类型/常量/纯helper/展示组件），`page.tsx` 瘦身至 915 行仅留 Home 容器 + Case 管理；纯移动无逻辑改动。二者均 `tsc --noEmit` 通过 + 浏览器验证（注入模拟裁定案例）引用跳转/高亮生效、拆分后零回归
- **端到端联调 5 问修正（2026-07-05，见 [docs/端到端问题修正计划.md](docs/端到端问题修正计划.md)）**：真实后端多轮调研暴露的 5 问全部修复并端到端验证——**①** `TraceView` 自动滚动加 `!isDone` 门禁（首轮展开后折叠条不再被滚出视口）；**②** step 摘要改用 args 派生的查询词（`summarizeArgs`）而非不稳定的 thought，展开卡去掉 Thought 块（多轮一致，`anyThoughtInLabel:false`）；**③** `call_llm_with_tools` 对瞬时错误（连接/超时/5xx/429）指数退避重试 3 次、4xx 不重试 + `OpenAI(max_retries=3)` + `retry` 事件前端可见；**④** `submit_verdict` schema 加 `summary_for_user`（面向用户摘要）+ `suggested_followups`（可点击建议 chips → 发起取证式追问），研究背书（Perplexity/Google DR/ChatGPT DR 标准收尾）；**⑤** 引擎在 `final_answer` 直传结构化 `sources`（`seen_urls`+AIS 对账，`verified` 标注），前端优先消费、正则降级兜底——引用全程稳定（实测 12 条全 verified）。**附带修复**：`done{success:false}` 但有 answer 时不再误判为 "Investigation failed"——Verifier 质量 caveat ≠ 硬失败，有答案即渲染裁定（仅无 answer 才判 error）

- **截图/图片上传（2026-07-05）**：前端 composer 加「📎 Attach screenshot / PDF」按钮——选图 POST `/api/v0/upload`（已有端点）拿 server path，显示缩略图 chip；`image_path` 透传进 `/run_stage/stream` 请求（`runSSE`→start/followup），引擎经 `build_user_message` 提示 + `analyze_image_cloud` 工具 OCR 图片内容再展开调查。支持纯图片提交（无文字也可发起）。端到端验证：上传伪造招聘诈骗聊天截图 → 引擎 OCR 提取「bytedance-recruit.com / 加密货币押金 / 无需面试」→ 正确裁定「大概率有坑」。补齐了 SPEC「图片/截图输入 ✅ 引擎支持 → 待实现上传界面」的前端缺口。
- **追问回答模式 + 真流式（2026-07-05，见 §5 对话摘要第二期）**：followup 走 `answer_mode`——非裁定型追问基于上文已取证结论对话式作答（0 步、gate 放行、画布不弹卡），裁定型追问仍自动重新取证出裁定；`stream_llm_with_tools` 逐 token 流式（`_stream_answer_portion` 剔除 Thought 脚手架防泄漏、异常回退非流式）；前端打字机渲染 + 完整对话文本。端到端验证通过。
- **stage2/stage3 场景层上线（2026-07-05）**：`offercheck_stage2.txt`（简历定向——JD×简历差距→优先级修改清单，纯文本非裁定；`_answer_requires_evidence` 豁免 stage2，不强制取证）+ `offercheck_stage3.txt`（沟通证伪——核实招聘方身份/异常沟通，`submit_verdict` 复用三态 靠谱/存疑/大概率有坑；Evaluator/Verifier 的 stage4 宽松分支扩展到 stage3）。前端 `STAGE_META` stage2/stage3 `engine: soon→live`。四阶段全部可跑。端测：stage2 零检索出结构化清单、gate 不误拦；stage3 联网核查招聘方身份出裁定。**附带**：`_build_structured_sources` 过滤 localhost/127.0.0.1 噪声来源

- **offercheck eval_suite 全四阶段评测（2026-07-05）**：`offercheck/eval_suite/cases.jsonl` 标注 **26 例**，按阶段分两套评分——**裁定级（stage1×5 / stage3×4 / stage4×12，共 21 例）**：`expected_verdict` ∈ reliable/suspicious/likely_scam；stage4/3 用 靠谱/存疑/大概率有坑，stage1 选岗用 推荐/谨慎/不推荐（分类器映射到同一枚举，「不推荐」靠有序匹配先于「推荐」命中）；覆盖冒名真公司/仿冒域名/预付款/无面录用/超额支票/礼品卡/僵尸岗/不可核实公司等模式。**关键词召回（stage2×5）**：简历定向是非裁定自由文本清单，用 `expected_keywords` 预置「JD 要求而简历缺失、正确定向分析必须指出的差距关键词」，按命中召回率评分。**接入 Eval Harness**：`EvalCase` 加 `expected_verdict`/`stage`/`expected_keywords`（透传 stage 加载阶段 prompt，每条恰带其一，harness 据此选评分模式）；裁定级 `classify_prediction_verdict`+`compute_verdict_metrics`（**准确率 / 误报率(把靠谱判成有坑) / 漏报率(把诈骗判成靠谱) / 拒答率 + 混淆矩阵**）；关键词 `score_keyword_recall`+`compute_keyword_metrics`（**平均召回率 + 达标率**，recall≥0.6 判 correct）；`run --suite offercheck` 一键跑，`analyze`/`compare` 均已扩展显示两套指标（compare 2pp 门禁即回归 gate）。**附带发现并规避**：`_classify_verdict_level` 对整行 [Verdict] 做子串匹配，会被 reason 里否定语境的 scam 关键词（「未发现…诈骗案例」）误伤把「靠谱」判成 likely_scam——评分器改为只信任裁定 label 本身（分隔符前那截）规避（同名产品 bug 待单独修，见 §5）。端到端 smoke：stage4（scam+ok）2/2、stage2 关键词召回 100%、stage1 裁定路径打通（借此暴露引擎对「不可核实公司」偏严判成 likely_scam 而非 suspicious，与 §3.3「无法核实→存疑」doctrine 不符，属真实校准信号）。

- **跨阶段记忆真接线 + followup 多能力路由（2026-07-05）**：回应主办方「智能/自动化 + 全链路持续辅助」维度。**(P0 跨阶段携带)** `buildCrossStageContext`（web/app/ui.tsx）把同 Case 内已完成早前阶段的最新结论（裁定/关键事实/红旗/来源；stage2 为清单摘录）打包成紧凑 JSON 块，`startInitial` 注入 `[本案早前阶段的已取证结论…]` + `[本阶段任务]`——stage4 证伪自动带着 stage1 调研上下文跑。**修正一处 UI 假宣传**：原 stage4 banner 宣称 "findings are carried automatically" 但代码从未注入；banner 已改为仅在真有携带时展示并列出携带阶段。接地不弱化：携带块明示「新裁定仍须独立取证核实」，evidence gate / Verifier 照常。**(P1 followup stage 路由)** 单一对话内多能力组合——`nexa_agent/stage_router.py` 两级路由：关键词门（无跨阶段信号词→零成本 keep，覆盖绝大多数普通追问）+ fast 层 LLM 单次确认（防「offer」这类高频词误伤；失败/超时安全回落 keep）；`RunStageRequest.auto_route`（前端 followup 默认开）；路由发生时发 `stage_routed` 事件（additive，不破坏既有 SSE 契约）并以路由后 stage prompt 执行，`done` 事件回传 effective stage；前端 routed badge + trace 注记。端测：stage4 追问「简历和 JD 差距」→ 路由 stage2 并以 stage2 清单格式作答（19.5s）；「offer 竞业条款正常吗」同阶段词不触发（零成本 keep）；浏览器验证 banner 展示 + 请求体真实携带 stage1 裁定/事实/来源。

- **英文 eval 用例 + PDF 附件修复 + Profile 决策（2026-07-05）**：**(1) 英文用例**：eval_suite 26→32——6 条全英文（tag `english`：scam×2 礼品卡/WhatsApp-USDT、suspicious×1 查无此司、reliable×2 GitLab/Cloudflare、stage2 关键词×1 Rust/Kafka/observability）；en_scam_01 live 验证裁定 likely_scam 正确——英文输入端到端 + 裁定 label 分类器双语兼容，出海维度硬证据。**(2) PDF 附件死路修复**：`build_user_message` 此前对一切附件都说「上传了一张图片…用 analyze_image」，而 `analyze_image_cloud` 拒绝非图片扩展名——前端「Attach screenshot / PDF」的 PDF 路径实际是坏的；改为按扩展名指路（.pdf → `read_pdf`，图片 → VLM OCR），并补装 `pymupdf4llm`（requirements 有但 agent env 未装）。live 验证：PDF 简历 → read_pdf 提取 → stage2 输出基于 PDF 真实内容的差距清单（正确识别 RabbitMQ≠Kafka、缺 Rust 等）；**顺带验证接地层**：修复前 read_pdf 失败时 agent 诚实报告「PDF 解析失败」并拒绝编造简历内容。**(3) 简历解析决策**：不做结构化简历解析器（ATS 式实体抽取）——LLM 直接对 raw text 做 JD 对比已足够，结构化解析是过早工程化且市场实测幻觉重灾区；PDF 上传→read_pdf→分析 这条已通的路就是「简历解析」的正确形态。**(4) Profile 机制决策**：比赛期不做——Case 级轻 Profile 已事实存在（`forms` 为 Case 级，resume 在 stage1/2 间自动复用），单用户 demo 下跨 Case 复用无可见收益；三件真缺口（全局 Profile / LTM 挂接 / 画像驱动红旗校准）列为赛后代办（飞书文档专节 + §5）。

- **followup 记忆管理修复：对话窗口 + 路由粘滞 + 材料注入（2026-07-05）**：真实使用暴露多轮连续性断裂——stage1 里追问「按简历和岗位比对」被路由 stage2 并要求提供 JD，用户回「如图」+JD 截图后引擎却**再次输出岗位靠谱裁定**，未做比对。**三重根因**：① `buildFollowupInput` 一步马尔可夫 + 有损解析——只带上一条 answer 的 `parseStructuredAnswer` 结构化字段，对话式回复（无 [Verdict]/[Fact]）解析后≈空，「请提供 JD 和简历」的关键上下文整体丢失；② 路由不粘滞——`runFollowup` 恒发面板 stage，「如图」无关键词过不了路由门→回落 stage1 prompt→重做岗位验证；③ 用户材料（forms.resume）不进 followup 上下文。**修复**：① 滚动对话窗口——最近 3 轮 user/assistant 转写（裁定型回复→紧凑结构化，**对话式回复→原文摘录**以保住「要材料」类指令），引用连续性取最近裁定轮 sources；context 标记改 `[对话上下文 - 供参考]`（`[追问/补充信息]` 标记不动，stage_router 契约不破）；② 路由粘滞——下一轮默认继承上一轮 `routedStage`（请求带 requestStage、UI 状态仍挂面板 stage；新 followup 预置 routedStage 保证链式继承 + badge 显示；引擎路由器仍可显式切走）；③ `forms.resume/jd` 作为 `user_materials` 注入（各截 1500 字）。**验证**：引擎侧模拟三轮对话 payload（sticky stage2）→ 输出【匹配概览】简历比对清单、引用简历中的 LangChain/RAG 原型、不再重复裁定；UI 侧预览种入 bug 现场 + fetch 拦截 → 实发 stage=stage2、历史含双轮（裁定轮 + 要材料轮）、user_materials 含简历、包体 854 字符紧凑。

- **检索工具优化（P0/P1 核心项，2026-07-03）**：react_loop 跨步硬缓存（同 tool+args 直接命中缓存）；web_fetch 登录墙拦截（linkedin/facebook/instagram blocklist）；web_search URL 去重；trafilatura 安装。未做残留见 §5。
- **对话摘要第二期全量（2026-07-05）**：`submit_verdict` 加 `summary_for_user`/`suggested_followups`；追问回答模式 `answer_mode`（非裁定追问 0 步对话式作答、gate 放行）；`stream_llm_with_tools` 逐 token 真流式（`_stream_answer_portion` 剔除 Thought 防泄漏，异常回退非流式）；前端打字机渲染。端测全通过。

**未建成 / 规划中 ⬜**（详见 §5）
- offercheck company_registry_search 工具（需外部工商 API）
- 前端部署（Vercel）+ GMI 调用证明 + demo 视频（需本人操作）
- eval_suite **全量跑分**（32 例 live，需 GMI 额度 + 时长，本人触发）
- 记忆管理 L1–L4 + 调查 playbook 层（赛后，方向已定，见 §5）

---

## 5. 待后续推进时补充（TODO）

> 推进到对应项时，把设计细节 / 决策理由回填到这里，并更新 §3 / §4。**已完成项统一移至 §4**（Verifier 校准 / 检索优化核心项 / eval_suite / stage2/3 / 对话摘要第二期 / 四阶段前端页面均已迁移）。

- `TODO(修复)` **`_classify_verdict_level` 否定语境误伤**：verifier.py 的裁定分类器对整行 [Verdict] 做子串匹配，reason 里出现「未发现…诈骗案例」会把「靠谱」误判为 likely_scam（导致 server 裁定卡给合法 offer 渲染成诈骗）。eval 评分器已用「只取 label 片段」规避，产品侧（server/前端裁定卡）仍受影响，应把分类改为优先取裁定 label（分隔符前那截）。
- `TODO(补充)` **部署与提交素材（7/6，本人操作）**：Vercel + 后端部署（⚠️ SSE 经代理的 buffering 断流风险，部署后第一件事跑一条完整调查流）；≥6 截图；3 分钟 demo 视频（persona 叙事）；GMI 调用证明 ≥2 张；32 例 eval live 跑分数字进提交文档。
- `TODO(补充)` **eval_suite 残留**：真实脱敏案例替换部分模拟案例；stage2 关键词召回可加 LLM-judge「建议采纳率」维度（当前用关键词召回作可判定代理）。
- `TODO(补充)` **检索优化残留**：Corrective 再检索、轻量重排（优先级不高）。
- `TODO(补充)` **company_registry_search**：需外部工商 API；stage3/4 当前用 web_search + whois 兜底（符合「不依赖陌生领域 KB」边界，不阻塞）。
- **用户画像 Profile 层（产品内）**：⏸ 决策不做（2026-07-05，见 §4）——比赛期性价比低（Case 级轻 Profile 已存在：`forms.resume` 在 stage1/2 间复用；单用户 demo 看不到跨 Case 收益）。赛后三件套代办已列飞书文档专节：全局 Profile（约半天）/ LTM 挂接（约半天）/ 画像驱动红旗校准（约 1 天，含薪资-市场基准信号）。
- **记忆管理优化方向（2026-07-05 评审，全部赛后）**：现状四层——①Trial 内 messages + Observation 分档截断（健康）②Trial 间 Scratchpad + Reflexion 教训池（健康）③轮次间前端 3 轮滚动对话窗口 + 路由粘滞 + user_materials（刚修复，天花板低）④跨阶段结构化携带 + LTM 零使用（最薄）。**结构性上限**：轮次/跨阶段记忆全靠前端打包塞单条 user prompt（server 每请求新建 agent 无状态）——好处是部署简单，代价是第 4 轮遗忘、事实不累积。**优先级**：**L1 Case 事实账本**（每 run 的事实/红旗/来源按实体键控合并进 Case 级账本，去重+来源累积+冲突标记，后续轮次注入累积账本而非最近 N 轮——Scratchpad 思想上移产品层，≈1–2 天，第一个做）→ **playbook 层**（见下条）→ **L2 对话压缩**（窗口外旧轮 fast 摘要 running summary，有 L1 后优先级降，≈半天）→ **L4 LTM 激活**（诈骗域名黑名单/已核实公司缓存/Profile，绑商业化推进）→ **L3 server 端真会话**（case_id + messages 数组续写，engine `execute` 接受 history，≈2–3 天，L1 落地后再评估必要性）。**L5 向量检索继续不做**（§2.10 触发条件未到）。
- **Skills / 调查 playbook 决策（2026-07-05）**：**完整 skills 基建不做**（SKILL.md 规范/目录发现/渐进式加载是开放任务域 agent 的上下文摊薄手段；四阶段封闭域用不上，违反 §1.1 统御原则）。**轻量调查 playbook 层值得做（赛后 1–2 天）**：stage prompt 本就是原始 skill 机制（stage router 即 dispatcher）——把高频诈骗模式的调查手册写成独立 prompt 片段（`playbooks/impersonation`：whois 注册时间+官方域名比对+MX 记录；`advance-fee`：礼品卡/押金取证要点；`ghost-job`：僵尸岗信号），调查中检测到模式信号时动态注入（复用 stage router 两级门：关键词→fast 确认）。与 L1 配套（账本存事实、playbook 存方法）；**Reflexion 教训池反复出现的教训可蒸馏固化为 playbook**（episodic→procedural memory distillation）——dynamic context loading 是 DeepSeek Harness 叙事强点。
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
