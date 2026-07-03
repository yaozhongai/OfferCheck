# 运行失败分析与修复方案 — run_20260702_234927_reflexion.log

> 分析日期:2026-07-03 | 场景:OfferCheck stage1(选岗调研),输入 = DeepSeek Harness 工程师 JD
> 结论:2 轮 Trial 全部判失败,总耗时 ~4.5 分钟、累计 ~25 万输入 token,但**实质性答案其实已经产出**(1870 字符调查报告)——失败是系统自己造成的,不是调查失败。

---

## 一、场景与运行时间线

用户拿自己求职目标(DeepSeek Harness 岗位 JD)做 dogfooding,跑 stage1 选岗调研。

| 阶段 | 发生了什么 | 结果 |
|---|---|---|
| Trial 1 step 1–3 | Pro 首步规划 + 8 次搜索/抓取,信息采集正常 | 正常 |
| **Trial 1 step 4** | 请求 400:`reasoning_content must be passed back` | **崩溃,Trial 1 判 llm_error** |
| Trial 1 反思 | 反思/教训/Scratchpad 提取正常(4 条事实) | 正常 |
| Trial 2 step 1–8 | 再次调查,**重复抓取 Trial 1 已抓过的 SCMP/verdent 页面**;中途纠偏触发 2 次 | 低效但推进 |
| **Trial 2 step 9–12** | 连续 4 步输出 1500–2000 token 的完整报告文本,但无 `Final Answer:` 哨兵 → 被判"无效",nag 注入 → 动态升级 Qwen(2 次)也无效 → max_steps 兜底 | **浪费 ~85s + ~80K token** |
| Trial 2 评估 | LLM Judge 400(模型名不被官方 API 认)→ 回退启发式,置信度 0.51 | **Judge 永久失效** |
| Trial 2 Verifier | passed=False(unreliable_source)→ Trial 2 判失败 | 实质性答案被丢弃 |
| Trial 2 反思 | **反思 len=0(消耗 2873 token)** → 记忆跳过,教训 0 条 | **Reflexion 学习信号丢失** |

---

## 二、问题清单(按根因聚类,附证据)

### P0 部署卫生:stale 进程跑旧代码(Trial 1 崩溃的直接原因)

**证据**:日志打 `react_agent.py:479 - Step N: 调用 LLM`,但当前文件该语句在 **511 行**(479 是加入 `_assistant_msg_to_dict` 修复*之前*的布局);traceback 的行号与源码文本错位(bytecode 行号来自导入时,显示文本读自当前磁盘文件);`ps` 抓到 19:59 启动的 uvicorn(PID 47886),早于 20:55 的修复落盘。

**结论**:`reasoning_content` 400 的修复**已在磁盘上**,但 23:49 的请求经由一个修复前启动、从未重启的进程执行。这不是代码 bug,是流程 bug——但它揭示了需要开发期热重载。

### P1 DeepSeek 思考模型输出协议覆盖不完整(最深的一类)

**注意执法点:错误来自 GMI 的 serving 栈,不是 DeepSeek 官方 API。** 日志中 400 是 GMI 网关包装的(`'type': 'backend_error'` 外壳 + DeepSeek 风格 details)——GMI 自己部署的 DeepSeek-V4 后端按 DeepSeek 思考模式协议强制校验。协议本身见 DeepSeek 官方文档([Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)):

> "若模型执行了工具调用,中间 assistant 的 `reasoning_content` **必须**参与上下文拼接并在后续请求中回传……If your code does not correctly pass back `reasoning_content`, the API will return a 400 error."
> "若无工具调用,中间 assistant 的 `reasoning_content` 不需要参与拼接。"

**GMI 与官方 API 的致命差异**:官方 API 可用 `extra_body={"thinking": ...}` 开关思考;**GMI 不认该参数(实测 422 `Unsupported parameter(s): thinking`)**。即在 GMI 上,V4 何时思考、思考多长完全不受调用方控制——`reasoning_content` 随时可能出现、也随时可能耗尽输出预算导致 `content` 为空。**防御性处理(回传保留 + 读取回退)不是可选加固,是唯一手段。**

这是行业级坑,opencode/anything-llm/n8n/open-webui/nanobot 都踩过(见来源清单)。当前代码状态:

| 路径 | 状态 |
|---|---|
| tool_calls assistant 回传(react_agent 549)| ✅ 已修(`_assistant_msg_to_dict`) |
| 纯文本 assistant 回传(react_agent 634)| ⚠️ 丢 reasoning_content(按官方协议可不回传,但 GMI 网关实现可能更严;统一保留无害) |
| **反思生成只读 `.content`(reflexion_agent 551)** | ❌ **本次 len=0 的直接原因**:GMI 上的 V4 是混合思考模型,输出可能全部在 `reasoning_content` 里、`content` 为空(2873 token 全烧在思考上)。picoclaw #966、LM Studio #1602、NemoClaw #246 报告了同类"reasoning-only 空响应"问题 |
| 教训提取 / Scratchpad 提取 / 兜底汇总 / 策展 / Evaluator / Verifier 的读取 | ❌ 同样只读 `.content`,同样暴露 |

### P2 Evaluator LLM Judge 逃出了 GMI(base_url 与系统其余部分不一致)

**证据**:日志 135 行 `The supported API model names are deepseek-v4-pro or deepseek-v4-flash, but you passed deepseek-ai/DeepSeek-V4-Flash` —— 这个措辞是 **DeepSeek 官方 API** 的报错(只有官方端点才会列官方模型名),证明 evaluator 的请求根本没进 GMI。

**根因**:整个系统的模型都跑在 GMI 上,唯独 `evaluator.py:257-262` 的 `base_url` 默认值写死 `DEEPSEEK_BASE_URL`(官方端点),而 `model_name` 用 `get_model_for_role()` 返回 GMI 风格名——名字和端点错配 → 400 → **每次都静默回退启发式**,hybrid 评估模式名存实亡。verifier.py:200 用 `MODEL_CONFIG`(GMI)是对的,evaluator 漏改。

### P3 终止契约错配:`Final Answer:` 哨兵 vs 原生 tool-calling 约定(最大浪费源)

**证据**:step 9–12 连续 4 步,模型每步输出 1532–2033 token 的**完整调查报告**(stage1 prompt 要求的结构化 schema),但因文本里没有 `Final Answer:` 字样被 `parse_llm_response` 判为无效 → 注入 nag("请调用工具或给出 Final Answer")→ 模型重写一遍 → 再被拒……四轮共烧 ~80K token、85 秒,最终 max_steps 兜底。

**根因**:行业约定(见 [Claude Agent SDK agent loop](https://platform.claude.com/docs/en/agent-sdk/agent-loop)、[Strands](https://strandsagents.com/docs/user-guide/concepts/agents/agent-loop/)、[ai-sdk loop control](https://ai-sdk.dev/docs/agents/loop-control)):**模型返回不含 tool_calls 的纯文本 = 最终答案,循环终止**。`Final Answer:` 哨兵是 text-ReAct 时代(正则解析)的遗产;迁移到原生 function calling 后,两套终止语义并存且冲突。stage prompt 要求输出"调查报告 schema",模型照做了,报告里自然不含哨兵。**这是格式契约自相矛盾,不是模型能力问题。**

### P4 动态升级误诊 + 计数器不封顶

**证据**:`consecutive_no_toolcall=3/2, 4/2`;Qwen 升级两次同样"无 tool_calls"——因为模型根本不是不会调工具,而是**在写最终答案**。升级白烧 42 秒 + 45K token。

**根因**:升级的触发信号(连续无 tool_calls)与它假设的故障(tool-call 能力不足)不匹配;P3 修复后该场景大部分消失。另:计数器超阈值后无终止策略,只会永远升级。

### P5 跨 Trial URL 去重失效

**证据**:Trial 2 step 2 重新抓取 SCMP、verdent 页面(Trial 1 已抓)。代码 `reflexion_agent.py:263`:`", ".join(sorted(visited_urls)[:5])` —— **字母序截前 5 条**(任意子集),以软提示注入,模型无视。

**根因**:软约束(提示)对抗模型的信息饥渴天然弱;去重应在**工具层硬拦截**,不该指望 prompt。

### P6 Verifier 过度驳回 + 全败语义粗暴

**证据**:Trial 2 答案 1870 字符、启发式判"实质性",Verifier 却 `passed=False (unreliable_source)` → 整个 run 报"所有 Trial 均失败",尽管 `ReflexionResult.answer` 里就躺着可用报告。

**文献支撑**:LLM Judge 存在系统性校准问题——[Overconfidence in LLM-as-a-Judge](https://arxiv.org/html/2508.06225v2)(高置信区间过度自信)、[Catching One in Five](https://arxiv.org/pdf/2606.10315)(生产多轮 agent 场景 judge 盲区、召回极低)。为 GAIA factoid 精确匹配校准的事实门控,对**研究型长报告**天然过严。二值 pass/fail 把"有据但未全部核实"和"彻底失败"混为一谈——这与 OfferCheck 自己的产品口径("证据不足显式标存疑",而非拒绝输出)直接矛盾。

### P7 反思质量与空反思

**证据**:Trial 2 反思 len=0 → `反思过短跳过` → 记忆不更新、教训 0 条。直接原因是 P1(reasoning-only 输出),但另有结构性弱点:[Reflexion 论文](https://arxiv.org/pdf/2303.11366)及后续分析([Honest Lying](https://arxiv.org/pdf/2605.29463))指出,trajectory 级单一失败信号的 credit assignment 很差,"换个搜索词"式泛化教训不收敛。当前 `_generate_reflection` 只有 exception 路径的 fallback,**空字符串不触发**。

### P8 工具层碎项

1. **trafilatura 未安装**(agent env;requirements.txt 里有)→ web_fetch 的备用提取引擎缺失,Jina 451(x.com)后无兜底(日志 103–104)。
2. **x.com/twitter 登录墙**:Jina 返回 451,该类域名应硬黑名单 + 提示换源。
3. thinking 日志误导:请求 flag `thinking=on`,完成日志打 `thinking=off`(判断条件是 `"pro" in model.lower()`)。纯 cosmetic。

---

## 三、修复方案(按比赛 ROI 排序)

### 方案 1(P2,一行级,立即):Evaluator 客户端对齐引擎配置
`evaluator.py` 的 `base_url`/`api_key` 默认值改为 `MODEL_CONFIG["base_url"]`/`MODEL_CONFIG["api_key"]`(与 verifier 一致)。LLM Judge 立即在 GMI 下复活,hybrid 评估恢复真实语义。

### 方案 2(P3+P4,核心,~1 小时):终止契约统一到原生 tool-calling 约定
1. **主判定**:无 `tool_calls` 且 `content` 实质非空(长度 > 阈值,或含 `[Verdict]` 等裁定标签)→ **即为最终答案**,终止循环。与 Claude Agent SDK/Strands/ai-sdk 的行业约定一致。
2. **哨兵向后兼容**:文本含 `Final Answer:` 时仍按哨兵截取。
3. **nag 降级**:仅当文本过短(疑似"思考自语")才 nag,且同一 nag 只注入一次。
4. **动态升级触发条件收紧**:改为"tool call 格式错误/参数解析失败"触发,纯文本输出不再计数;计数器达阈值后策略改为"强制 finalize"(接受当前文本或触发兜底),不再无限升级。

预期收益:本次 run 中 step 9 就会正常结束(而非 12 步 + 兜底),省 ~45% token 与 ~40% 时延;答案以模型第一次完整输出为准(质量高于兜底汇总)。

### 方案 3(P1+P7,~40 分钟):思考模型输出统一收口
前提认知:**GMI 上无法用参数关闭 V4 的思考**(`thinking` 参数 422),所以防御性处理是唯一手段,不是可选加固。
1. 新增 `extract_text(message) -> str`:`content` 为空时回退 `reasoning_content`(可加"思考内容仅兜底使用"截断策略)。
2. 替换所有单读 `.content` 的位置:反思生成、教训提取、Scratchpad 提取、兜底汇总、策展、Evaluator judge、Verifier。
3. 纯文本 assistant 回传(634 行)也走 `_assistant_msg_to_dict` 统一保留 `reasoning_content`(对官方协议冗余但无害,对 GMI 后端更稳)。
4. `_generate_reflection` 空串时走既有 fallback(`任务执行失败(mode)...`),保证记忆池永远有可注入教训。
5. **可选(GMI 特有)**:反思/教训提取/Scratchpad 这类简单结构化输出任务,不需要深度思考却在 GMI 上被迫烧 reasoning token(本次反思 2873 token 换来 0 字输出)。可把这几个 role 路由到 GMI 非思考模型,从根上避开 reasoning-only 问题并省 token。

### 方案 3B(P1+P4 根治,推荐):整体切换到 GMI 非思考模型

**核心认知**:GMI 上无法用参数关闭 V4 思考,那就换掉模型本身。这从**根上**消灭 P1(reasoning_content 400 / reasoning-only 空 content)整类问题,比方案 3 的防御性收口更彻底(收口仍应保留作廉价保险)。

**GMI 命名判据(2026-07-03 实测 68 模型目录)**:`-Instruct`=非思考(安全);`-Thinking`=思考;Qwen 3.5/3.6 不带后缀=默认思考;DeepSeek-V4(Pro/Flash)=思考,DeepSeek-V3.2/V3=非思考。

**实测(工具调用 + reasoning_content + 多轮 content + 400)**:

| 模型 | tool_calls | reasoning_content | 多轮 content | 400 | 延迟 |
|---|---|---|---|---|---|
| `deepseek-ai/DeepSeek-V3.2` | ✅ | 无 | ✅ | 无 | 5.2s |
| `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` | ✅ | 无 | ✅ | 无 | 4.8s |
| `moonshotai/Kimi-K2-Instruct-0905` | ✅ | 无 | ✅ | 无 | 5.5s |
| `Qwen/Qwen3.6-35B-A3B`(**旧 upgrade 层**)| ✅ | **有** | ✅ | 无 | 6.1s |
| `Qwen/Qwen3.5-35B-A3B` | ✅ | **有** | ✅ | 无 | 6.1s |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 429 容量不稳,暂不选 | | | | |

**推荐路由(全非思考)**:
- strong(react_first / verifier / evolver)= `deepseek-ai/DeepSeek-V3.2`(非思考、留 DeepSeek 家族)
- fast(react_main / reflection / lesson_extract / evaluator_llm / scratchpad)= `deepseek-ai/DeepSeek-V3.2`(或 `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`)
- upgrade(tool-call 备援)= `moonshotai/Kimi-K2-Instruct-0905` —— **必须换掉旧的 `Qwen/Qwen3.6-35B-A3B`:它本身是思考模型,作为"保证发 tool_calls 的备援"选择方向正好反了**(日志 step 11-12 升级到它后照样只输出思考、不发 tool_calls)。

**权衡**:丢掉深度 CoT 对 OfferCheck 影响有限——ReAct+Reflexion 把推理外化为可见的多步工具调用(对 demo 与"证伪"叙事更有利),强 instruct 模型规划工具调用足够;而 GMI 思考不可控的代价(token 烧穿、400、reasoning-only 空响应)是实打实的。改动量=`config.py` 的 `MODEL_TIER` 三行 + 保留方案 3 收口作保险。

### 方案 4(P0,运维,5 分钟):开发期热重载
`make backend`/开发启动加 `uvicorn --reload`;或至少在 README/Makefile 注明"改引擎代码后必须重启 server"。本次 Trial 1 崩溃即为此坑。

### 方案 5(P6,~1 小时):裁定分级替代二值成败
1. `ReflexionResult` 增加 `verification: verified | unverified | failed` 三态;Verifier 驳回时**不再判整轮失败**,而是答案降级为 `unverified` + 把 Verifier 的具体质疑附进 `[NeedUserConfirm]`。
2. 这正是 OfferCheck 的产品口径(诚实标注存疑,而非拒答)——判官召回低([2606.10315](https://arxiv.org/pdf/2606.10315))决定了驳回不可作为丢弃答案的充分条件。
3. stage 输出的校验改为 schema 完整性(有无 Verdict/RedFlag/证据链)+ 抽样事实核查,而非全文 factoid 门控。
4. 赛后:按 [DeepEval 最佳实践](https://deepeval.com/blog/llm-as-a-judge) 用 eval suite(15–20 例起步)做 judge 校准。

### 方案 6(P5,~30 分钟):URL 去重下沉到工具层
`execute_tool` 入口检查 `visited_urls`(跨 Trial 传入 react_loop):命中则不打网络,直接返回缓存摘要 + "此 URL 本会话已抓取"。软提示注入可保留但不再是唯一防线。配合已有 extracts 缓存实现零成本。

### 方案 7(P8,~10 分钟):工具层碎项
`pip install trafilatura`(并核对 agent env 与 requirements 漂移);web_fetch 加登录墙域名黑名单(x.com/twitter.com/linkedin.com/facebook.com)返回明确提示;修 thinking 完成日志。

---

## 四、实施顺序建议

| 顺序 | 方案 | 工时 | 为什么先做 |
|---|---|---|---|
| 1 | 方案 1(evaluator 一行) | 5 min | 修完 judge 立即恢复,后续所有 run 的评估才可信 |
| 2 | 方案 4(--reload) | 5 min | 防止后续修复再次"改了但没生效" |
| 3 | 方案 2(终止契约) | ~1 h | 最大 token/时延/体验收益,demo 核心路径 |
| 4 | 方案 3(思考模型收口) | ~40 min | 恢复 Reflexion 学习信号,防伪失败 |
| 5 | 方案 7(trafilatura 等) | ~10 min | 顺手 |
| 6 | 方案 5(裁定分级) | ~1 h | 产品口径正确性,demo 输出观感 |
| 7 | 方案 6(URL 硬去重) | ~30 min | 省 token,非阻塞 |

全部落地后,建议用同一 JD 输入回归一次,对比:步数、token、是否兜底、反思是否非空、judge 是否走 LLM——这五个指标就是本次日志暴露的五道伤。

---

## 五、检索来源

**DeepSeek 官方协议**
- [Thinking Mode | DeepSeek API Docs](https://api-docs.deepseek.com/guides/thinking_mode) — tool-call 轮 reasoning_content 必须回传、否则 400;无 tool call 轮可不回传
- [Reasoning Model | DeepSeek API Docs](https://api-docs.deepseek.com/guides/reasoning_model)

**同类 bug 的行业案例(reasoning_content 剥离 / reasoning-only 空响应)**
- [opencode #24130 — V4 Flash reasoning_content must be passed back](https://github.com/anomalyco/opencode/issues/24130)
- [anything-llm #5683 — agent loop 400 with DeepSeek v4 thinking](https://github.com/Mintplex-Labs/anything-llm/issues/5683)
- [n8n #29119 — Missing reasoning_content when using tools](https://github.com/n8n-io/n8n/issues/29119)
- [open-webui #23175 — reasoning_content stripped from assistant tool call messages](https://github.com/open-webui/open-webui/issues/23175)
- [nanobot #390 — missing reasoning_content in assistant tool call messages](https://github.com/HKUDS/nanobot/issues/390)
- [picoclaw #966 — thinking model returns empty content when reasoning consumes all tokens](https://github.com/sipeed/picoclaw/issues/966)
- [lmstudio #1602 — reasoning_content populated but content empty](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1602)
- [NemoClaw #246 — reasoning models return empty content](https://github.com/NVIDIA/NemoClaw/issues/246)

**Agent 循环终止约定**
- [How the agent loop works — Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/agent-loop) — "continues calling tools until it produces a response with no tool calls"
- [Agent Loop — Strands Agents](https://strandsagents.com/docs/user-guide/concepts/agents/agent-loop/)
- [Loop Control — AI SDK](https://ai-sdk.dev/docs/agents/loop-control)

**Reflexion 与反思质量**
- [Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023)](https://arxiv.org/pdf/2303.11366)
- [Honest Lying: Understanding Memory Confabulation in Reflexive Agents](https://arxiv.org/pdf/2605.29463)

**LLM-as-Judge 校准与过度驳回**
- [Overconfidence in LLM-as-a-Judge: Diagnosis and Confidence-Driven Solution](https://arxiv.org/html/2508.06225v2)
- [Catching One in Five: LLM-as-Judge Blind Spots in Production Multi-Turn Transaction Agents](https://arxiv.org/pdf/2606.10315)
- [LLM-as-a-Judge in 2026: best practices — DeepEval](https://deepeval.com/blog/llm-as-a-judge)
