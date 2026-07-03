# Agent 范式调研 × 幻觉根治 —— 针对 OfferCheck 的选型

> 日期:2026-07-03 | 缘起:run_20260703_002637 中 Agent 零工具调用、编造完整调查报告并被判"成功"
> 结论先行:**幻觉与循环范式正交,换模型/换范式都治不好;根治在于补"接地层"(强制取证 + 来源归属 + 独立验证)。OfferCheck 应保留 ReAct+Reflexion,定位为「求职诈骗专用 Deep Research Agent」。**

---

## 一、问题重述:这是"接地"问题,不是"范式"问题

现象:strong 模型(Qwen235B-Instruct)在 step 1、`tool_calls=False`、宣称"我已有足够信息",直接产出带 `[Source] github.com/DeepSeek-Project/Harness`、`domain_whois_lookup(...)返回` 的裁定报告——**全部编造,工具零调用**;而三层校验(LLM Judge 422 挂、启发式只量长度、Verifier 跳过)全部放行,判"成功 0.60"。

研究佐证幻觉根因与范式无关:
- "agentic 设置下,幻觉风险被迭代放大……根因是**知识缺口 + 弱接地→自信 best guess**,解法是检索接地与 'no evidence, no answer'"([getmaxim](https://www.getmaxim.ai/articles/llm-hallucination-detection-and-mitigation-best-techniques/))。
- 连 RAG 也不免疫:法律检索工具幻觉率仍达 33%,"检索消除幻觉"是伪命题([Agentic RAG SoK](https://arxiv.org/pdf/2603.07379))。
- 忠实度取决于**紧归属**:每条事实都要能溯源到具体检索片段(AIS, Attributable to Identified Sources);有引用时忠实度显著更高([citation-enforced RAG](https://www.mdpi.com/2076-3417/16/6/3013))。

→ 换模型(变量)、换范式(控制流)都不触及根因。

---

## 二、主流 Agent 范式全景(2026)

来源:[The 4 single-agent patterns](https://theaiengineer.substack.com/p/the-4-single-agent-patterns)、[Navigating Modern LLM Agent Architectures](https://www.wollenlabs.com/blog-posts/navigating-modern-llm-agent-architectures-multi-agents-plan-and-execute-rewoo-tree-of-thoughts-and-react)、[ReCode arXiv:2510.23564](https://arxiv.org/pdf/2510.23564)、[PreAct arXiv:2402.11534](https://arxiv.org/pdf/2402.11534)

| 范式 | 机制 | 主要解决 | 主要代价 | 对幻觉 | OfferCheck |
|---|---|---|---|---|---|
| **ReAct** | 推理↔行动↔观察 交替,按观察决定下一步 | 工具接地、自适应、可见轨迹 | 步数/token 浪费 | 中性(不强制取证) | ✅ 内循环保留 |
| **Plan-and-Execute** | 先全局规划再逐步执行,可重规划 | 长程连贯、减少 LLM 调用 | 重规划复杂度 | 中性 | ○ 可作 stage 内子结构 |
| **ReWOO** | 一次性规划全部工具调用,不看中间观察 | token 效率、少调用 | 盲规划,难应对需观察的任务 | **更差**(不能随发现调整) | ✗ 反诈需边查边变 |
| **Reflexion** | 失败后生成语言自省,注入下轮 | 从失败学习、无需训练 | 延迟;反思质量不稳 | 弱正向 | ✅ 外循环保留 |
| **LATS** | 对动作序列做 MCTS/A* 树搜索 | 困难探索、回溯 | 算力/调用爆炸 | 中性 | ✗ 5 天太重 |
| **CodeAct** | 动作表达为可执行代码 | 动作可组合、复用代码能力 | 需沙箱、非检索型 | 中性 | ✗ 场景不符 |
| **CoVe(验证链)** | 出稿→规划核查问题→独立作答→修订 | **直接降幻觉**,F1 +23% | 多轮调用 | ✅✅ 强正向 | ✅ 用于强化 Verifier |
| **Deep Research Agent** | 多角色(规划/检索/综合/验证)+ 强制引用溯源 | 带引用的可信长报告 | 编排复杂 | ✅✅ 参考架构 | ✅✅ **即 OfferCheck 本体** |

关键判断:**除 CoVe 与 Deep Research 的"验证/引用"环节外,其余范式都是控制流优化,与接地性正交。** 选范式治幻觉是走错科室。

---

## 三、CoVe(Chain-of-Verification)—— 唯一直接对症的范式

来源:[CoVe 原论文 arXiv:2309.11495](https://arxiv.org/pdf/2309.11495)、[ACL 2024 Findings](https://aclanthology.org/2024.findings-acl.212.pdf)、[learnprompting](https://learnprompting.org/docs/advanced/self_criticism/chain_of_verification)

四步:①出基线答案 → ②规划核查问题 → ③**独立**回答核查问题(Factored 变体:不看原答案,避免自我印证)→ ④据核查结果修订。F1 从 0.39→0.48(+23%)。

落地到 OfferCheck:把现有 Verifier 从"一次性 pass/fail"升级为 CoVe factored——针对裁定里每条 `[Fact]`,生成"这条来源真的存在吗/工具真的返回了吗"的核查问题,用独立检索回答,再修订裁定。这正是"主动证伪"的产品内核。

---

## 四、Deep Research Agent —— OfferCheck 的参考架构

来源:[LangChain Deep Agents](https://www.langchain.com/blog/deep-agents)、[VeriTrace arXiv:2605.26081](https://arxiv.org/pdf/2605.26081)、[DR³-Eval arXiv:2604.14683](https://arxiv.org/pdf/2604.14683)、[Microsoft: Enterprise Deep Research Agents](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/building-enterprise-grade-deep-research-agents-in-house-architecture-and-impleme/4435256)

DRA 的共性:自主规划长程流程 + 遍历异构网页源 + **综合成结构化、带引用溯源的报告**;实现上常是 **ReAct 风格内核**(每轮可并行多查询)+ **多角色分解**(检索/推理/综合/**评估**各司其职)。

→ OfferCheck = 「求职诈骗垂直的 Deep Research Agent」。你已有 ReAct+Reflexion+Verifier 骨架,**不需要换范式,只需把 DRA 标配的"引用溯源 + 独立验证"补实。**

---

## 五、根治方案(接地层三件套,均与现有 ReAct+Reflexion 兼容)

1. **强制取证 gate(no evidence, no answer)**
   含 `[Verdict]` 的输出,finalize 前必须已有 ≥N 次成功检索类工具调用(web_search/web_fetch/whois…);否则拒绝该 Final Answer,注入"你尚未取证,请先调查"。落点:react_loop 的 final_answer 分支 + 裁定型 stage 标记。

2. **来源对账(AIS 归属)**
   解析裁定里的 `[Source]`,逐条比对本次 `action_history`/`extracts`:命中真实工具调用→保留;找不到→标 `[未验证来源]` 并下调该条置信度。落点:verifier 或一个后处理器。这直接兑现 OfferCheck "证据不足显式标存疑" 的口径。

3. **Verifier 默认触发 + CoVe 化**
   裁定型任务(stage1/stage4)默认触发 Verifier(不再依赖启发式触发条件);Verifier 内部改 CoVe factored:对每条事实独立复查再修订。

**保留**:ReAct(内)、Reflexion(外)、动态升级(改为 tool-call 格式失败触发)、之前的 reasoning_content 收口作保险。
**顺带已修**:evaluator 的 base_url + extra_body(P2 全修完,LLM Judge 现能真正跑,能发现"引用的工具从未被调用")。

---

## 六、为什么这条路对"用比赛打造 DeepSeek 面试项目"最优

- **叙事**:"我诊断出幻觉与循环范式正交,构建了强制取证 + 来源归属(AIS)+ 对抗验证(CoVe)的 grounding/verification **harness**"——纯 harness 工程故事,正中 DeepSeek Harness 岗;远强于"我把 ReAct 换成 X"。
- **产品=工程同构**:OfferCheck 的核心卖点"证伪"本身就是 verification,把接地层做强 = 产品力与工程叙事同时变强。
- **可量化**:接地层每一项都能进 Eval Harness(取证率、来源命中率、幻觉拦截率、误报率),回应 JD"以真实任务反馈量化迭代"。

---

## 七、检索来源汇总

**范式对比**
- [The 4 single-agent patterns: ReAct vs Plan-and-Execute vs ReWOO vs Reflexion](https://theaiengineer.substack.com/p/the-4-single-agent-patterns)
- [Navigating Modern LLM Agent Architectures (Plan-Execute / ReWOO / ToT / ReAct)](https://www.wollenlabs.com/blog-posts/navigating-modern-llm-agent-architectures-multi-agents-plan-and-execute-rewoo-tree-of-thoughts-and-react)
- [ReCode: Unify Plan and Action (arXiv:2510.23564)](https://arxiv.org/pdf/2510.23564)
- [PreAct: Prediction Enhances Planning (arXiv:2402.11534)](https://arxiv.org/pdf/2402.11534)

**幻觉根因与接地**
- [LLM Hallucination Detection and Mitigation (getmaxim)](https://www.getmaxim.ai/articles/llm-hallucination-detection-and-mitigation-best-techniques/)
- [Prevent AI Agent Hallucinations in Production (StackAI)](https://www.stackai.com/insights/prevent-ai-agent-hallucinations-in-production-environments)
- [SoK: Agentic RAG (arXiv:2603.07379)](https://arxiv.org/pdf/2603.07379)
- [Reducing Hallucinations via Citation-Enforced Prompting in RAG (MDPI)](https://www.mdpi.com/2076-3417/16/6/3013)

**CoVe**
- [Chain-of-Verification Reduces Hallucination (arXiv:2309.11495)](https://arxiv.org/pdf/2309.11495)
- [CoVe — ACL 2024 Findings](https://aclanthology.org/2024.findings-acl.212.pdf)
- [CoVe 教程 (learnprompting)](https://learnprompting.org/docs/advanced/self_criticism/chain_of_verification)

**Deep Research Agent**
- [LangChain Deep Agents](https://www.langchain.com/blog/deep-agents)
- [VeriTrace: Evolving Mental Models for Deep Research Agents (arXiv:2605.26081)](https://arxiv.org/pdf/2605.26081)
- [DR³-Eval (arXiv:2604.14683)](https://arxiv.org/pdf/2604.14683)
- [Building Enterprise-Grade Deep Research Agents (Microsoft)](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/building-enterprise-grade-deep-research-agents-in-house-architecture-and-impleme/4435256)
