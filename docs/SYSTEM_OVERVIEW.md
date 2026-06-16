# ReAct + Reflexion 自进化智能体系统 — 功能全景文档

> 最后更新: 2026-06-16  
> 状态: Phase 1-2 全部完成, P0 全部实现, 方案 A+B (来源内生意识 + Verifier 外部审计) 已集成

---

## 一、系统架构总览

```
用户问题
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  ReflexionReActAgent (局外大循环)                                  │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌───────────────┐              │
│  │ 记忆检索  │───▶│ ReAct执行 │───▶│ 行为评估器    │              │
│  │ (情景记忆) │    │ (局内循环) │    │ (Heuristic)   │              │
│  └──────────┘    └──────────┘    └───────────────┘              │
│       ▲                                │                         │
│       │                    ┌───────────┴───────────┐             │
│       │                    │   ✅ 通过              │  ❌ 失败    │
│       │                    ▼                        ▼             │
│       │          ┌──────────────────┐    ┌──────────────────┐   │
│       │          │ Verifier Agent   │    │  反思生成器       │   │
│       │          │ (方案B-事实网关)  │    │  (独立LLM)        │   │
│       │          └──────────────────┘    └──────────────────┘   │
│       │               │         │                │              │
│       │          ✅ 可靠   ❌ 驳回            │              │
│       │               │         │                │              │
│       │               ▼         ▼                ▼              │
│       │          ┌────────┐ ┌──────────────────────────┐       │
│       │          │ 输出给  │ │ 更新记忆 + 教训提取       │       │
│       │          │ 用户    │ │ (ReflexionMemory)        │       │
│       └──────────│        │ └──────────────────────────┘       │
│                  └────────┘                                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 二、已实现功能

### 2.1 ReAct 推理引擎 (`react_agent.py`)

**原理**: ReAct (Reasoning + Acting) 将 LLM 推理与外部工具调用交织在一起。每步由 Thought → Action → Observation 组成。

**关键实现细节**:

| 机制 | 实现方式 | 作用 |
|------|---------|------|
| Stop Token | `stop=["Observation:"]` | 强制 LLM 输出到 Action 后停住 |
| Thinking 开关 | 首步开启, 后续关闭 | 首步深度规划, 后续快速响应 |
| max_tokens 动态调整 | 首步 thinking=on 时 8192, 后续 4096, 兜底 6144 | 防止 thinking 消耗过多导致输出截断 |
| 解析容错 | 正则兼容 Markdown 加粗、中文冒号 | 提高 Action 解析成功率 |
| URL 引号剥离 | 解析时自动去掉 args 首尾引号 | 修复 LLM 输出 `web_fetch("url")` 导致的 URL 无效 |
| 达最大步数兜底 | 快模型汇总已有信息 | 不强制编造答案, 交由 Reflexion 重试 |
| 动态升级 | 连续 2 次解析失败 → 强模型 + thinking | 避免快模型能力不足导致循环卡死 |

**工具集** (9 个):
- `web_search` — Tavily 优先, DuckDuckGo 兜底
- `wikipedia_search` — Wikipedia REST API
- `web_fetch` — **Jina Reader + trafilatura 双引擎网页正文提取（免费，无需 API Key）**
- `tavily_extract` — Tavily Extract 网页提取（需 API Key）
- `save_content` — 缓存内容写入本地 Markdown
- `analyze_image` — 端侧 MiniCPM 视觉理解
- `analyze_image_cloud` — 云端 Kimi K2.6 视觉理解
- `calculator` — numexpr 安全数学计算
- `get_current_time` — 获取当前时间

**记忆注入接口**: `long_term_memory` 参数接受字符串列表, 作为 User Message 前缀注入。

---

### 2.2 Reflexion 局外大循环 (`reflexion_agent.py`)

**原理**: 在 ReAct 单次执行外包裹"试错→评估→反思→重试"循环。

**执行流程** (每个 Trial):

```
阶段1: 记忆检索 → 从 ReflexionMemory 检索相关教训
阶段2: ReAct 执行 → 将记忆作为前缀注入, 运行完整 ReAct 循环
阶段3: 行为评估 → 启发式规则检测 loop/overflow/tool_misuse/wrong_reasoning
阶段4: Verifier 节点 → (条件触发) 评估数据来源可信度
阶段5: 反思生成 → 独立 LLM 会话分析轨迹, 生成 2-3 句具体反思
阶段6: 教训提取 → LLM 归纳为 1-3 条标准化教训, Jaccard 去重
阶段7: 记忆更新 → 反思+教训存入 ReflexionMemory
```

**配置**: 最大 3 轮 Trial, 每轮最多 16 步。

**三明治截断**: 反思时保留轨迹首 1000 + 末 1940 字符, 确保看到"初始规划 + 失败现场"。

**教训提取** (`_extract_lessons`): LLM 归纳为 ≤50 字的标准化教训, 以"下次..."开头, Jaccard 去重, lesson_counter 追踪频次。

---

### 2.3 方案 A — Agent 内生来源意识 (Prompt 级改造)

**设计理念**: 在 Agent 的 System Prompt 中植入"必须关注数据来源可信度"的意识。

**实现**:
- Final Answer 后强制附加「数据溯源」段落, 格式为 `[Fact] / [Source] / [Confidence]`
- 来源可信度评级标准: High (官方/学术) > Medium (Wikipedia/媒体) > Low (论坛/UGC)
- 新增规则 #8: "来源可信度优先 — 对统计数据优先从官方来源获取, 不接受 Quora/论坛作为数据支撑"

**示例输出**:
```
Final Answer: 41

## 数据溯源
[Fact] Nature 2020年发表了1002篇Article
[Source] https://www.nature.com/nature/articles?year=2020 — Nature官方网站
[Confidence] High — 官方一手数据
```

---

### 2.4 方案 B — Verifier Agent 事实网关 (`verifier.py`)

**设计理念**: 在行为评估器通过后, 设立独立的"主编"节点对 Agent 输出的数据来源进行事实核查。

**两层核查**:

| 层级 | 机制 | 开销 |
|------|------|------|
| 快捷规则 | 正则匹配 Quora/知乎/Reddit/AI聚合站等 URL → 直接驳回 | 零 Token |
| LLM 深度评估 | 提取 `[Fact/Source/Confidence]` 键值对, 调用 fast 模型评估来源可信度 | ~100 tokens |

**动态触发机制** (满足任一即触发):
1. **意图路由**: 问题含"统计/数据/年份/财报/多少"等关键词
2. **行为触发**: 上轮 Trial 步数 ≥ 5 或触发过 loop/tool_misuse
3. **记忆触发**: 上轮被 Verifier 驳回 (unreliable_source)

**驳回反馈格式** (具象操作指引, 不是抽象错误):
```
你在回答中使用了来自 Quora（问答网站）的数据。这类来源属于用户生成内容,
数据未经验证。请重新搜索:
1. 直接前往 Nature 官网 (nature.com) 寻找官方期刊指标
2. 搜索关键词包含 "official"、"annual report"、"journal metrics"
3. 严禁使用 Quora/知乎/Reddit/论坛/第三方聚合站作为数据支撑
```

---

### 2.5 多策略评估器 (`evaluator.py`)

**四种启发式规则** (按优先级):

| 优先级 | 检测项 | failure_mode | 判定逻辑 |
|--------|--------|-------------|---------|
| 1 | 达到最大步数 | context_overflow | 轨迹含"达到最大步数" |
| 2 | 重复动作 | loop | 相同工具+参数 ≥3 次 (args 去引号归一化) |
| 3 | 连续工具错误 | tool_misuse | `[错误]` 出现 ≥3 次 |
| 4 | 不确定性标志 | wrong_reasoning | 答案含"我不确定"等短语 |

**Loop 检测改进**:
- 检测 1: 相同工具+参数（归一化后）≥3 次 → 死循环
- 检测 2: 同名工具 ≥15 次（不同参数）→ 搜索策略失效
- Args 归一化: 去掉首尾引号、归一化空白 → `web_fetch("url")` 和 `web_fetch(url)` 算相同

**Hybrid 模式**: 高严重度 (loop/tool_misuse/context_overflow) → 直接返回; 低严重度 (wrong_reasoning) → LLM 复审确认。

---

### 2.6 记忆系统 (`memory.py`)

**ReflexionMemory**: 有界滑动窗口, FIFO 淘汰 (max=3), JSON 持久化。新增 `lessons` 字段存储标准化教训。

**质量过滤**: 长度校验 (20-500 字符) + Jaccard 去重 (相似度 >0.8 跳过)。

---

### 2.7 模型路由 — 任务感知的动态模型选择

```python
MODEL_TIER = {
    "strong": {"model": "deepseek-v4-pro"},
    "fast":   {"model": "deepseek-v4-flash"},
}

MODEL_ROUTING = {
    "react_first":    "strong",  # 首步规划
    "react_main":     "fast",    # 后续步骤
    "reflection":     "fast",    # 反思生成
    "evaluator_llm":  "fast",    # LLM 评估
    "lesson_extract": "fast",    # 教训提取
    "evolver":        "strong",  # 规则演化
}
```

**max_tokens 动态调整**:

| 场景 | max_tokens |
|------|-----------|
| 首步 (thinking=on) | 8192 |
| 后续步 (thinking=off) | 4096 |
| 兜底汇总 | 6144 |

**动态升级**: 连续 2 次 Action 解析失败 → 自动切换强模型 + thinking。

**预期收益**: Token 成本降低约 60%。

---

### 2.8 信用分配 — 步骤效用计算 (`react_agent.py`)

**效用规则**:

| 事件 | 效用值 |
|------|--------|
| 搜索返回有效信息 | +0.5 |
| 搜索无结果 | 0.0 |
| 重复搜索同一 (tool, args) | -0.5 |
| fetch/extract 成功 (>500 chars) | +1.0 |
| fetch/extract 失败 | -0.3 |
| 工具报错 | -0.5 |
| 解析失败 | -0.5 |
| 计算成功 | +0.3 |

**critical_step**: 效用最低的负效用步骤 → 传递给反思生成器, 加入 prompt 中的 `【关键步骤定位】`。

---

### 2.9 配置管理 (`config.py`)

所有超参数集中管理:
- **MODEL_TIER / MODEL_ROUTING**: 模型分层与路由策略
- **REACT_CONFIG**: max_steps=16, observation_max_chars=2000
- **REFLEXION_CONFIG**: max_trials=3, evaluator_mode=hybrid
- **MEMORY_CONFIG**: 记忆容量与淘汰策略
- **DYNAMIC_UPGRADE_THRESHOLD**: 动态升级阈值 = 2

---

### 2.10 批量执行脚本 (`batch_runner.py`)

从 TXT 文件读取问题, 执行 ReflexionReActAgent:
- 一个 TXT 文件 = 一个问题
- `#` 开头行为注释, 自动过滤
- 支持 `--output results.json` 输出结构化结果
- 支持 `--quiet` 静默模式

---

## 三、模块依赖关系

```
config.py ◀── 所有模块 (MODEL_TIER, MODEL_ROUTING, get_model_for_role)
    │
    ├── react_agent.py ◀── tools.py (9 个工具)
    │       ▲
    │       │
    ├── reflexion_agent.py ◀── evaluator.py
    │       ▲                       │
    │       │                       │
    │       ├── verifier.py ◀───────┘  (方案B: 事实网关)
    │       │       ▲
    │       │       └── 动态触发: 意图路由 + 行为触发 + 记忆触发
    │       │
    │       └── memory.py ◀─────────┘ (ReflexionMemory + lessons)
    │
    └── logger_config.py ◀── 所有模块
```

---

## 四、文件清单

| 文件 | 状态 | 职责 |
|------|------|------|
| `react_exp/react_agent.py` | ✅ | ReAct 引擎: 模型路由 + 信用分配 + max_tokens 动态调整 |
| `react_exp/reflexion_agent.py` | ✅ | Reflexion 控制器: Trial 循环 + 反思 + 教训提取 |
| `react_exp/evaluator.py` | ✅ | 多策略评估器: 4 条启发式 + Hybrid 模式 |
| `react_exp/verifier.py` | ✅ | Verifier Agent: 快捷规则 + LLM 深度评估 + 动态触发 |
| `react_exp/memory.py` | ✅ | ReflexionMemory: FIFO + 持久化 + 质量过滤 |
| `react_exp/config.py` | ✅ | 统一配置: MODEL_TIER/ROUTING + 各模块超参数 |
| `react_exp/tools.py` | ✅ | 9 个工具: 含 web_fetch (Jina + trafilatura 双引擎) |
| `react_exp/batch_runner.py` | ✅ | 批量执行脚本: TXT 输入 → Reflexion 执行 |
| `react_exp/prompts/react_system.txt` | ✅ | System Prompt: 含方案A的「数据溯源」格式要求 |
| `react_exp/prompts/reflection_system.txt` | ✅ | 反思 Prompt: 含来源审查指令 |

---

## 五、优先级总览

| 优先级 | 功能 | 状态 | 核心价值 |
|--------|------|------|---------|
| ~~P0~~ | 模型路由 (任务→模型映射) | **已完成** | 成本降低 ~60% |
| ~~P0~~ | 有界教训提取 + 三明治截断 | **已完成** | 反思质量 + 晋升链路 |
| ~~P0~~ | 信用分配 (critical_step + step_utilities) | **已完成** | 反思精准度提升 |
| ~~P0~~ | 方案A: Agent 内生来源意识 | **已完成** | 数据溯源 + 来源可信度评级 |
| ~~P0~~ | 方案B: Verifier Agent 事实网关 | **已完成** | 来源可靠性外部审计 |
| ~~P0~~ | 动态触发机制 | **已完成** | 意图路由 + 行为触发, 按需核查 |
| ~~P1~~ | Loop 检测改进 (arg 归一化) | **已完成** | 消除误判 |
| ~~P1~~ | URL 引号剥离 | **已完成** | 修复 LLM 输出格式导致的 URL 无效 |
| ~~P1~~ | max_tokens 动态调整 | **已完成** | 防止 thinking 截断输出 |
| P1 | 规则试用期验证 | 待实现 | 证据驱动的进化 |
| P1 | HarnessEvolver (trace 消费) | 待实现 | 闭环自进化 |
| P2 | 上下文压缩 | 待实现 | 长任务支持 |
| P2 | 重组触发器 | 待实现 | 架构自动演进 |
