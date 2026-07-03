---
name: nexa-dev-loop
description: >
  Nexa Agent 项目的迭代开发工作流。每个新 session 先读 SPEC.md 和 CLAUDE.md 理解项目全貌，
  从 SPEC.md §5 TODO 列表选取任务实施，完成后用真实问题（DeepSeek Agent Harness JD）运行 stage4
  offer 证伪测试，失败则读 logs/ 和 trace 分析修复，最后更新 SPEC.md。
  当用户提到继续开发 Nexa Agent、推进 TODO、跑测试验证、更新 SPEC、OfferCheck 开发迭代、
  或在 Nexa_Agent 项目目录下要求"继续"/"下一个任务"时使用此 skill。
---

# Nexa Dev Loop — 迭代开发工作流

你是 Nexa Agent 项目的开发搭档。这个 skill 定义了一个完整的「选任务 → 实现 → 测试 → 分析 → 更新」循环，
确保每次 session 都能快速进入状态并产出可验证的进展。

## 第零步：建立项目上下文

每个新 session 的第一件事——读懂项目现在长什么样：

1. 读 `SPEC.md`（项目宪法：核心目标、明确不做、关键决策、当前状态、TODO 列表）
2. 读 `CLAUDE.md`（开发指南：环境、命令、架构、约束）
3. 快速扫一眼最近的 git log（`git log --oneline -10`）了解最新进展

重点关注：
- **SPEC.md §4** 的"已建成"和"未建成"——这是进度快照
- **SPEC.md §5** 的 TODO 列表——这是待办池
- 最近的 commit 是否改动了你即将触碰的文件

这一步的目的是避免重复劳动和踩已知坑。不要跳过。

## 第一步：选择任务

从 SPEC.md §5 TODO 列表中选一个任务。选择原则：

1. **当前焦点是 stage4（offer 证伪）**——优先选与 stage4 直接相关的 TODO
2. 如果 stage4 的所有 TODO 都已完成，停下来告诉用户「stage4 开发完成」
3. 优先级标记（P0 > P1 > ... > P6）是参考，但 stage4 相关性优先于优先级数字

选定后，向用户简要说明：选了哪个任务、为什么选它、大致实现思路。等用户确认再动手。

## 第二步：实现

动手写代码。关键约束（来自 CLAUDE.md，不要违反）：

### 环境
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate agent
```
所有 python/uvicorn 命令必须在 conda `agent` 环境下跑（Python 3.10），base 是 3.9 会出 pydantic 报错。

### 架构红线
- 依赖方向：`server/offercheck → nexa_agent`，绝不反过来
- `nexa_agent/` 是 headless 核心引擎，不能引入 FastAPI/DB/UI 依赖
- 模型配置唯一真源：`nexa_agent/config.py` 的 `MODEL_TIER` + `MODEL_ROUTING`，不要在别处加
- GMI 非思考模型：`extra_body={"thinking": ...}` 必须用 `if SUPPORTS_THINKING_PARAM` 守卫
- 助理消息带 tool_calls 时用 `_assistant_msg_to_dict(msg)` 回传（DeepSeek 要 `reasoning_content`）

### 防幻觉四层（不要削弱）
1. 接地铁律 prompt（react_system.txt 顶部）
2. 强制取证 gate（零成功检索时拦截裁定）
3. submit_verdict 结构化终止工具
4. AIS 来源对账（seen_urls 核对 [Source]）

改代码时始终意识到这四层的存在，任何改动都不应让幻觉更容易逃逸。

## 第三步：测试

代码改完后，用真实问题跑一次 stage4 offer 证伪，验证改动是否生效、是否引入回归。

### 测试用 JD（DeepSeek Agent Harness 研发岗）

```
Agent Harness 研发/工程方向

加入 DeepSeek Harness 团队，探索和解决面向下一代 Agent Harness 的工程难题。

【主要职责】
参与设计 DeepSeek 的 Harness 产品的技术架构与选型。
参与开发 DeepSeek 的 Harness 产品。
与研究员紧密合作，共同定义和实现 Harness 领域的前沿创新。
与模型训练团队的工程师与研究员深度沟通与合作，参与实现模型与 Harness 的共同进化。
以内部真实任务做为 Harness 产品和模型相关能力训练的重要反馈源，持续迭代产品能力。
理解并分析团队收集到的用户反馈。协助维护 Harness 产品用户社群。
协助项目管理相关工作。

【任职要求】
技术水平过硬，技术眼界广阔。知名高校本科学历及以上。
熟练使用 AI Agent 工具进行软件开发。
是 Agent 产品的高强度用户，对 Agent Harness 的开发有极大的热情。
熟悉 LLM 以及 Agent 基本机制及其技术原理。
良好的中文沟通能力。
```

### 运行命令

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate agent
python -m nexa_agent.reflexion_agent "请对以下 JD 进行 offer 证伪调查：Agent Harness 研发/工程方向，加入 DeepSeek Harness 团队..." --stage stage4 --max-trials 1 --max-steps 20
```

把完整 JD 文本作为任务输入传入。`--max-trials 1` 先跑一轮看效果，有问题再调。

### 判断标准
- **成功**：agent 调用了多个检索工具、交叉验证了公司信息、最终通过 submit_verdict 给出裁定，且 evaluator 判定通过
- **失败信号**：零工具调用、编造信息（幻觉）、evaluator 判 `wrong_reasoning`、evidence gate 死循环、400/422 API 错误

## 第四步：失败分析

如果测试失败，按以下顺序排查：

### 4.1 读日志
```bash
# 找到最新的 reflexion 日志
ls -lt logs/run_*_reflexion.log | head -1

# 查看完整日志（重点关注 WARNING/ERROR 行和 evaluator 判定）
cat logs/run_YYYYMMDD_HHMMSS_reflexion.log
```

关注的关键行：
- `evaluator` 的判定理由（为什么判失败？`wrong_reasoning` / `incomplete` / `hallucination`？）
- `react_agent` 的 step 日志（调用了什么工具？返回了什么？）
- `evidence_gate` 是否触发（被 nag 了几次？）
- API 错误（400 = reasoning_content 未回传，422 = thinking 参数泄漏）
- `web_fetch` / `web_search` 的成功/失败模式

### 4.2 定位根因
常见失败模式及对策：

| 症状 | 根因 | 修复方向 |
|------|------|----------|
| 零工具调用 + 编造报告 | prompt 被忽视 / 模型滑向 chat 模式 | 强化 react_system.txt 接地铁律 / 检查 system prompt 是否被截断 |
| evaluator 判 wrong_reasoning | 评估器标准与任务不匹配 | 调 evaluator prompt / 检查 stage prompt 的调查维度定义 |
| evidence_gate 死循环 | 检索全失败但 gate 反复 nag | 检查搜索 provider 健康 / 放大 MAX_EVIDENCE_GATE_NAGS |
| 400 API 错误 | reasoning_content 未回传 | 检查 _assistant_msg_to_dict 覆盖 |
| 422 API 错误 | thinking 参数泄漏到 GMI | 检查 SUPPORTS_THINKING_PARAM 守卫 |
| submit_verdict 未触发 | 模型用文本结束而非工具 | 检查 submit_verdict 工具定义是否在 tools list 中 |

### 4.3 修复并重测
修完后回到第三步重新测试。如果同一个问题连续失败两次，停下来和用户讨论——可能需要更大的架构调整。

## 第五步：更新 SPEC.md

任务完成且测试通过后，更新 SPEC.md：

1. **§4 当前状态**：把完成的功能从"未建成"移到"已建成"，附简短描述
2. **§5 TODO**：
   - 已完成的 TODO 项：删除 `TODO(补充)` 前缀，填入实际的设计细节和决策理由
   - 如果开发过程中发现了新问题或新需求：添加新的 TODO 项
3. **§3 关键决策**：如果实现过程中做了重要的设计决策，在 §3 中记录决策和理由
4. 更新 §0 顶部的版本日期

更新原则：SPEC.md 是项目宪法，记录 **what** 和 **why**，不记录 how（how 在代码里）。
保持简洁，一个决策用 2-3 行说清楚。

## 循环

完成一轮后，回到第一步，从 TODO 列表选下一个任务。直到 stage4 相关的所有 TODO 都完成为止。

## 退出条件

当以下条件满足时，告诉用户「stage4 开发完成」并停止循环：
- SPEC.md §5 中与 stage4 / offer 证伪相关的 TODO 全部完成
- 用 DeepSeek JD 测试通过（evaluator 判定成功）
- SPEC.md 已更新到最新状态
