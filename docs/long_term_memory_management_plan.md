# Agent 长期记忆管理与自进化系统方案

> 融合：Reflexion（语言强化学习）+ Harness Engineering（AHE自动演化）+ Mem0（生产级记忆层）+ Hermes（文件持久化自进化）
> 目标：构建一个能自我反思、自我纠错、自我进化的完整 Agent 系统
> 适配：react_exp 模块，DeepSeek-V4-Pro 模型

---

## 一、设计哲学

### 1.1 核心命题

> **不改模型权重，只改"怎么用模型"** —— Hermes Agent 自进化哲学

记忆管理的终极目标不是简单存储信息，而是让 Agent **从经验中学习并持续进化**。三层递进关系：

```
记忆 → 反思 → 进化
(存储) → (分析) → (行为改变)
```

### 1.2 设计原则

| 原则 | 来源 | 含义 |
|------|------|------|
| 可观测性驱动 | AHE | 每次记忆写入/读取都有 trace，改动有 manifest |
| 分层存储 | Mem0/MemGPT | 工作记忆、情景记忆、语义记忆分离 |
| 证据化演化 | AHE | 记忆更新基于运行证据，不靠"自我反思"幻觉 |
| 文件持久化 | Hermes | 记忆以文件形式存在，可审计、可版本控制 |
| 有界窗口 | Reflexion | 避免记忆膨胀，FIFO + 重要性衰减 |
| 最小侵入 | 项目约束 | 不破坏现有 ReAct 架构 |

### 1.3 演进路径定位

```
Prompt Engineering (2022-2023)
  → Context Engineering (2024)
    → Harness Engineering (2025-)
      → Memory-Driven Self-Evolution (目标)
```

---

## 二、记忆分层架构

### 2.1 三层记忆模型（认知科学启发）

```
┌─────────────────────────────────────────────────┐
│            Layer 3: 语义记忆 (Semantic Memory)     │
│  长期知识、通用规则、跨任务模式                       │
│  存储：skills/, rules/, patterns/                  │
│  生命周期：永久（除非被显式废弃）                     │
├─────────────────────────────────────────────────┤
│            Layer 2: 情景记忆 (Episodic Memory)     │
│  具体任务经历、成功/失败案例、反思                    │
│  存储：episodes/, reflections/                    │
│  生命周期：有界窗口（max=10, FIFO+重要性）           │
├─────────────────────────────────────────────────┤
│            Layer 1: 工作记忆 (Working Memory)      │
│  当前任务上下文、短期状态、即时观察                    │
│  存储：LLM上下文窗口 + 临时文件                     │
│  生命周期：单次任务/会话                            │
└─────────────────────────────────────────────────┘
```

### 2.2 各层详细设计

#### Layer 1: 工作记忆（Working Memory）

- **载体**：LLM 上下文窗口（DeepSeek-V4-Pro 128K context）
- **内容**：当前任务描述、已执行步骤、工具返回结果、中间推理
- **管理策略**：
  - 自动截断超长 Observation（保留前500 + 后200 字符）
  - 步骤超过 5 步时压缩早期 Thought（仅保留 Action 和关键 Observation）
  - 类似 MemGPT 的"软分页"：重要内容固定，次要内容按需召回

#### Layer 2: 情景记忆（Episodic Memory）

- **载体**：JSON 文件，每条记忆一个 Entry
- **内容结构**：

```python
@dataclass
class EpisodeEntry:
    id: str                    # UUID
    timestamp: str             # ISO 8601
    task_description: str      # 任务描述
    trajectory_summary: str    # 执行轨迹摘要（压缩版）
    outcome: str               # "success" | "failure" | "partial"
    reflection: str            # Reflexion 生成的自然语言反思
    failure_mode: str | None   # 失败模式分类（如果失败）
   lessons: list[str]         # 提炼的经验教训
    confidence: float          # 记忆置信度 0-1
    access_count: int          # 被检索次数
    last_accessed: str         # 最后访问时间
    source_task_id: str        # 关联的任务ID
```

- **管理策略**：
  - 窗口大小：max_size = 10（论文推荐3条用于注入，但存储可多些）
  - 淘汰策略：复合策略（非纯FIFO）
    ```
    score = 0.4 * recency + 0.3 * importance + 0.2 * access_freq + 0.1 * confidence
    淘汰 score 最低的条目
    ```
  - 注入策略：每次任务开始时，检索 top-3 相关记忆注入 System Prompt

#### Layer 3: 语义记忆（Semantic Memory）

- **载体**：Markdown 文件（类似 Hermes 的 SKILL.md / MEMORY.md）
- **内容类型**：

| 类型 | 文件格式 | 示例 |
|------|----------|------|
| 规则 (Rules) | `rules/*.md` | "搜索时优先用 Wikipedia，web_search 作为补充" |
| 技能 (Skills) | `skills/*.md` | "处理数学计算题的标准流程" |
| 模式 (Patterns) | `patterns/*.md` | "用户问时事新闻 → 先 web_search 再 summarize" |
| 约束 (Constraints) | `constraints/*.md` | "不要在第一步就给出 Final Answer" |

- **管理策略**：
  - 从情景记忆中自动提炼（当同一 lesson 出现 ≥3 次时晋升）
  - 版本控制：每次修改记录 change_manifest
  - 失效机制：连续 5 次使用该规则后任务失败 → 标记为 deprecated

---

## 三、自进化闭环机制

### 3.1 核心闭环（融合 Reflexion + AHE）

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  执行任务    │───→│  收集 Trace   │───→│  评估结果    │
│  (ReAct)    │    │  (Observability)│   │  (Evaluator) │
└─────────────┘    └──────────────┘    └──────────────┘
       ↑                                       │
       │                                       ↓
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  注入记忆    │←───│  更新记忆    │←───│  生成反思    │
│  (Retrieval) │   │  (Memory Mgr) │   │  (Reflexion) │
└─────────────┘    └──────────────┘    └──────────────┘
       ↑                                       │
       │           ┌──────────────┐            │
       └───────────│  规则演化    │←───────────┘
                   │  (Evolve)    │
                   └──────────────┘
```

### 3.2 四阶段详细流程

#### Phase 1: 执行与观测（Execute & Observe）

```python
class TaskExecution:
    """单次任务执行，产出完整 trace"""
    
    def execute(self, task: str) -> ExecutionTrace:
        # 1. 检索相关记忆
        relevant_memories = self.memory.retrieve(task, top_k=3)
        
        # 2. 构建增强 prompt（记忆注入）
        enhanced_prompt = self.inject_memories(task, relevant_memories)
        
        # 3. ReAct 循环执行
        trace = self.react_loop(enhanced_prompt)
        
        # 4. 记录完整轨迹（AHE Observability）
        self.tracer.log(trace)
        
        return trace
```

#### Phase 2: 评估与诊断（Evaluate & Diagnose）

```python
class Evaluator:
    """多策略评估器"""
    
    def evaluate(self, trace: ExecutionTrace) -> EvalResult:
        # 启发式评估
        heuristic_score = self.heuristic_check(trace)
      
        # LLM 评估（让模型判断任务是否完成）
        llm_score = self.llm_evaluate(trace)
        
        # 失败模式分类（AHE 风格）
        if not self.is_success(heuristic_score, llm_score):
            failure_mode = self.classify_failure(trace)
            # 失败模式包括：
            # - tool_misuse: 工具使用不当
            # - premature_answer: 过早给出答案
            # - loop_detected: 陷入循环
            # - context_overflow: 上下文溢出
            # - wrong_reasoning: 推理错误
            return EvalResult(success=False, failure_mode=failure_mode)
        
        return EvalResult(success=True)
```

#### Phase 3: 反思与记忆更新（Reflect & Update）

```python
class ReflectionEngine:
    """Reflexion 风格的反思引擎"""
    
    def reflect(self, trace: ExecutionTrace, eval_result: EvalResult) -> ReflectionOutput:
        # Hindsight 全局反思 prompt
        reflection_prompt = f"""
        ## 任务回顾
        任务：{trace.task}
        执行步骤数：{trace.num_steps}
        结果：{eval_result.outcome}
        失败模式：{eval_result.failure_mode or "N/A"}
        
        ## 关键轨迹（压缩版）
        {trace.compressed_trajectory}
        
        ## 历史反思（避免重复错误）
        {self.memory.get_recent_reflections(n=3)}
        
        ## 请生成反思
        1. 这次执行的核心问题是什么？
        2. 如果重新执行，应该在哪一步做出不同决策？
        3. 提炼 1-2 条可复用的经验教训（简洁、具体、可操作）
        4. 这个经验是否具有通用性？（是→可能晋升为规则）
        """
        
        reflection = self.llm.generate(reflection_prompt)
        
        # 写入情景记忆
        episode = self.create_episode(trace, eval_result, reflection)
        self.memory.episodic.add(episode)
        
        # 检查是否需要晋升为语义记忆
        self.check_promotion(episode)
        
        return reflection
```

#### Phase 4: 规则演化（Evolve — AHE 风格）

```python
class HarnessEvolver:
    """AHE 风格的 Harness 演化器"""
    
    def evolve(self, episodes: list[EpisodeEntry], eval_history: list[EvalResult]):
        # 1. 失败模式聚合
        failure_patterns = self.aggregate_failures(eval_history)
        
        # 2. 生成演化提案（类似 AHE 的 Evolve Agent）
        evolution_prompt = f"""
        ## 近期运行统计
        总任务数：{len(eval_history)}
        成功率：{self.success_rate(eval_history)}
        
        ## 高频失败模式
        {failure_patterns}
        
        ## 当前规则集
        {self.memory.semantic.list_rules()}
        
        ## 请提出 Harness 改进建议
        可修改对象：system_prompt / rules / constraints / tools_description
        要求：
        - 每条建议必须指明修改哪个文件
        - 说明预计修复哪类失败
        - 说明可能引入的退化风险
        """
        
        proposals = self.llm.generate(evolution_prompt)
        
        # 3. 写入 change_manifest（AHE 核心机制）
        manifest = ChangeManifest(
            iteration=self.current_iteration,
            proposals=proposals,
            expected_fixes=[...],
            potential_regressions=[...],
            timestamp=datetime.now().isoformat()
        )
        self.save_manifest(manifest)
        
        # 4. 应用改动（需要验证后才固化）
        self.apply_proposals(proposals, tentative=True)
```

### 3.3 Change Manifest 机制（AHE 精华）

每次记忆/规则的修改都必须产出 manifest，使改动可追溯：

```json
{
  "iteration": 5,
  "timestamp": "2025-01-15T10:30:00Z",
  "changes": [
    {
      "file": "rules/search_strategy.md",
      "action": "update",
      "reason": "近5次任务中3次因'先web_search再wikipedia'导致信息不全",
      "old_content": "优先使用 web_search",
      "new_content": "复杂知识问题优先 wikipedia_search，时事问题用 web_search",
      "expected_fix": ["knowledge_tasks_accuracy"],
      "potential_regression": ["news_tasks_latency"]
    }
  ],
  "validation_criteria": "下轮知识类任务成功率 > 70%",
  "rollback_trigger": "整体成功率下降 > 10%"
}
```

---

## 四、记忆检索与注入策略

### 4.1 检索机制

```python
class MemoryRetriever:
    """多策略记忆检索"""
    
    def retrieve(self, query: str, top_k: int = 3) -> list[MemoryEntry]:
        # 策略1：语义相似度（向量检索）
        semantic_results = self.vector_search(query, top_k=5)
        
        # 策略2：失败模式匹配
        # 如果当前任务类似于历史失败任务，优先召回其反思
        failure_matches = self.failure_pattern_match(query)
        
        # 策略3：最近性偏好
        recent = self.get_recent(n=2)
        
        # 融合排序
        candidates = semantic_results + failure_matches + recent
        scored = self.rank(candidates, query)
        
        return scored[:top_k]
    
    def rank(self, candidates, query):
        """复合评分"""
        for c in candidates:
            c.score = (
                0.35 * c.semantic_similarity +
                0.25 * c.recency_score +
                0.20 * c.importance_score +
                0.10 * c.access_frequency +
                0.10 * c.success_relevance  # 成功经验加分
            )
        return sorted(candidates, key=lambda x: x.score, reverse=True)
```

### 4.2 注入格式（User Message 前缀）

```python
def build_memory_prefix(memories: list[MemoryEntry]) -> str:
    """构建记忆注入前缀"""
    if not memories:
        return ""
    
    prefix = "## 历史经验（请参考但不要盲目照搬）\n\n"
    
    for i, mem in enumerate(memories, 1):
        if isinstance(mem, EpisodeEntry):
            prefix += f"### 经验 {i}（{mem.outcome}）\n"
            prefix += f"- 任务类型：{mem.task_type}\n"
            prefix += f"- 教训：{'; '.join(mem.lessons)}\n\n"
        elif isinstance(mem, SemanticRule):
            prefix += f"### 规则 {i}\n"
            prefix += f"- {mem.content}\n\n"
    
    prefix += "---\n\n"
    return prefix
```

---

## 五、记忆持久化架构（Hermes 风格）

### 5.1 文件组织结构

```
react_exp/
├── memory/
│   ├── MEMORY_INDEX.md          # 记忆索引（类似 Hermes MEMORY.md）
│   ├── config.yaml              # 记忆系统配置
│   ├── working/                 # Layer 1: 工作记忆（临时）
│   │   └── current_session.json
│   ├── episodes/                # Layer 2: 情景记忆
│   │   ├── ep_001.json
│   │   ├── ep_002.json
│   │   └── ...
│   ├── semantic/                # Layer 3: 语义记忆
│   │   ├── rules/
│   │   │   ├── search_strategy.md
│   │   │   └── answer_format.md
│   │   ├── skills/
│   │   │   └── math_solving.md
│   │   ├── patterns/
│   │   │   └── news_query.md
│   │   └── constraints/
│   │       └── no_premature_answer.md
│   ├── traces/                  # 运行轨迹（可观测性）
│   │   └── trace_20250115.json
│   └── manifests/               # 变更清单（AHE）
│       ├── manifest_iter001.json
│       └── manifest_iter002.json
```

### 5.2 配置文件

```yaml
# memory/config.yaml
memory:
  episodic:
    max_size: 10
    eviction_policy: "composite"  # composite | fifo | importance
    promotion_threshold: 3        # 同一教训出现N次晋升为规则
    injection_top_k: 3
  
  semantic:
    auto_deprecate_after: 5       # 连续N次使用后失败则废弃
    version_control: true
    max_rules: 20
    max_skills: 10
  
  working:
    max_context_tokens: 60000     # 预留给工作记忆的token数
    compression_after_steps: 5    # 超过N步开始压缩
    observation_max_chars: 700    # 单次Observation最大字符数
  
  retrieval:
    strategy: "hybrid"            # hybrid | semantic | recency | random
    semantic_weight: 0.35
    recency_weight: 0.25
    importance_weight: 0.20
    frequency_weight: 0.10
    success_weight: 0.10
  
  evolution:
    enabled: true
    evolve_after_n_tasks: 5       # 每N个任务触发一次演化
    rollback_threshold: 0.10      # 成功率下降超过此值则回滚
    manifest_required: true       # 强制写 manifest
```

---

## 六、与 Reflexion 的深度融合

### 6.1 记忆在 Reflexion 试次循环中的角色

```
Trial 1: ReAct(task) → Evaluate → Fail
                                    ↓
                             Reflect → 写入 Episode
                                    ↓
Trial 2: ReAct(task + memory[trial1_reflection]) → Evaluate → Fail
                                                              ↓
                                                        Reflect → 更新 Episode
                                                              ↓
Trial 3: ReAct(task + memory[trial1+2_reflections]) → Evaluate → Success
                                                              ↓
                                                    记录成功经验 → 检查晋升
```

### 6.2 Reflexion 与长期记忆的关系

| 维度 | Reflexion 短期记忆 | 长期记忆系统 |
|------|-------------------|-------------|
| 作用域 | 单任务多试次 | 跨任务跨会话 |
| 内容 | 本次失败的具体反思 | 提炼的通用教训 |
| 注入方式 | 追加到 messages | 前缀到 System Prompt |
| 淘汰 | 任务结束即清除 | 复合评分淘汰 |
| 与 AHE 关系 | 是 AHE "Experience Observability" 的一部分 | 是 AHE "State/Memory" 模块 |

---

## 七、Harness Engineering 六模块映射

将 AHE 六大模块映射到本系统：

| AHE 模块 | 本系统实现 | 可演化对象 |
|----------|-----------|-----------|
| 上下文/知识 | Memory Retriever + Injection | 注入策略、top_k、权重 |
| 工具/权限 | tools.py + tool descriptions | 工具描述、参数说明 |
| 验证/约束 | Evaluator + constraints/ | 评估规则、失败阈值 |
| 状态/记忆 | 三层记忆架构 | 淘汰策略、窗口大小 |
| 可观测性/反馈 | traces/ + manifests/ | trace 粒度、诊断逻辑 |
| 人类接管 | max_trials 限制 + 人工标注接口 | 接管触发条件 |

### 7.1 可观测性设计（AHE 三层可观测）

#### Component Observability（组件可观测）
- 每个记忆文件有明确路径和格式
- 新增规则必须注册到 MEMORY_INDEX.md
- 记忆系统状态可通过 `memory status` 命令查看

#### Experience Observability（经验可观测）
- 每次任务执行产出完整 trace
- 失败任务自动生成诊断报告
- 统计面板：成功率趋势、失败模式分布、记忆命中率

#### Decision Observability（决策可观测）
- 每次记忆/规则修改写 change_manifest
- Manifest 包含预期效果和回归风险
- 下轮验证 manifest 预测是否兑现

---

## 八、实施计划

### Phase 1: MVP（第 1-2 周）

**目标**：最小可运行的记忆系统

- [ ] 实现 `memory.py`：EpisodeEntry 数据结构 + JSON 持久化
- [ ] 实现基础检索：语义相似度（基于 task description 关键词匹配）
- [ ] 实现记忆注入：User Message 前缀方式
- [ ] 集成到 Reflexion 循环：反思结果写入情景记忆
- [ ] FIFO 淘汰策略

### Phase 2: 增强记忆（第 3-4 周）

**目标**：完整的分层记忆 + 自动晋升

- [ ] 实现语义记忆层：rules/ + skills/ 文件管理
- [ ] 实现晋升机制：情景→语义自动提炼
- [ ] 实现复合评分淘汰策略
- [ ] 实现 working memory 压缩
- [ ] 添加 traces/ 记录（可观测性基础）

### Phase 3: 自进化闭环（第 5-7 周）

**目标**：AHE 风格的 Harness 自动演化

- [ ] 实现 HarnessEvolver：失败模式聚合 + 演化提案
- [ ] 实现 change_manifest 机制
- [ ] 实现回滚机制：效果下降时自动恢复
- [ ] 实现 MEMORY_INDEX.md 自动维护
- [ ] 集成统计面板：成功率、记忆命中率、演化历史

### Phase 4: 高级优化（第 8+ 周）

**目标**：向量检索 + 多 Agent 记忆共享

- [ ] 接入 Embedding 模型实现真正的语义检索
- [ ] 实现记忆衰减（时间衰减 + 置信度衰减）
- [ ] 实现跨任务类型的记忆迁移
- [ ] 实现 Hermes 风格的 Skills 自动生成
- [ ] 探索记忆的图结构存储（Mem0g 风格）

---

## 九、关键设计决策与权衡

### 9.1 为什么不用向量数据库（Phase 1-3）？

- **原因**：项目定位是教学/实验性质，优先可解释性和可调试性
- **权衡**：关键词匹配足够 Phase 1-3 的场景，Phase 4 再引入 Embedding
- **Hermes 启示**：Hermes 使用 SQLite FTS5 全文检索，已证明对小规模记忆高效

### 9.2 为什么选择文件持久化而非数据库？

- **原因**：
  1. 可直接 git 版本控制
  2. 可人工审计和编辑
  3. 与 AHE manifest 机制天然兼容
  4. 降低依赖复杂度
- **Hermes 验证**：Hermes 的 MEMORY.md + SKILL.md 文件方案经过 6.4万 Star 验证

### 9.3 记忆注入位置选择

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| System Prompt 后缀 | 权重高，模型更遵循 | 与系统指令混淆 | ❌ |
| User Message 前缀 | 清晰分离，不影响指令 | 可能被忽略 | ✅ |
| 独立 Assistant 消息 | 模拟"回忆"过程 | 增加对话轮次 | ❌ |

### 9.4 涌现能力门槛

> **重要**：Reflexion + 长期记忆的有效性强依赖模型能力。
> - DeepSeek-V4-Pro 级别模型已验证可产出高质量反思
> - 弱模型（如 7B 级）的反思可能是幻觉，需要额外验证层

---

## 十、与现有方案的对比

| 特性 | 本方案 | Mem0 | Hermes | AHE | 纯 Reflexion |
|------|--------|------|--------|-----|-------------|
| 分层记忆 | ✅ 三层 | ✅ 多层 | ✅ 二层 | ✅ LTM/STM | ❌ 单层 |
| 自动晋升 | ✅ | ✅ 自适应 | ❌ 手动 | ❌ | ❌ |
| 可观测性 | ✅ 三层 | ❌ | ❌ | ✅ 核心 | ❌ |
| Change Manifest | ✅ | ❌ | ❌ | ✅ 核心 | ❌ |
| 回滚机制 | ✅ | ❌ | ❌ | ✅ | ❌ |
| 文件持久化 | ✅ | ❌ 向量DB | ✅ | ✅ | ❌ |
| 规则演化 | ✅ | ❌ | ❌ | ✅ | ❌ |
| 轻量级 | ✅ | ❌ 需DB | ✅ | ❌ 需Sandbox | ✅ |
| 适配 ReAct | ✅ | 需改造 | 需改造 | 需改造 | ✅ |

---

## 十一、预期效果与成功指标

### 量化指标

| 指标 | 基线（无记忆） | Phase 1 目标 | Phase 3 目标 |
|------|--------------|-------------|-------------|
| 任务成功率 | ~60% | ~70% | ~80% |
| 平均试次数 | 2.5 | 2.0 | 1.5 |
| 重复失败率 | ~40% | ~20% | ~10% |
| Token 消耗/任务 | 8K | 9K（记忆开销） | 7K（压缩+精准） |

### 质性指标

- Agent 不再重复犯相同错误（"每次错误转化为框架规则"）
- 对类似任务展现出"经验积累"效应
- 规则库随时间自动丰富和精炼
- 可通过 manifest 追溯每次行为改变的原因

---

## 十二、风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| 记忆污染（错误经验被记录） | 置信度衰减 + 连续失败自动废弃 |
| 记忆幻觉（反思不准确） | 基于结果验证 + 双重确认 |
| 记忆膨胀 | 有界窗口 + 复合评分淘汰 |
| 规则冲突 | 优先级机制 + manifest 冲突检测 |
| 过度依赖记忆 | 注入时标注"参考但不盲从" |
| 演化发散 | 回滚机制 + 成功率阈值 |

---

## 参考资源

1. **Reflexion** - Shinn et al., 2023 - 语言强化学习框架
2. **AHE** - 复旦/北大/奇绩智峰, 2025 - arxiv.org/abs/2604.25850
3. **Mem0** - mem0ai, 2025 - 生产级 AI 记忆层 (arxiv.org/abs/2504.19413)
4. **MemGPT/Letta** - 虚拟内存分层架构
5. **Hermes Agent** - Nous Research - 文件持久化自进化 (6.4万 Star)
6. **MemEvolve** - Meta-Evolution of Agent Memory Systems (arxiv.org/abs/2512.18746)
7. **O-Mem** - Omni Memory System for Self-Evolving Agents (arxiv.org/abs/2511.13593)
8. **Memory OS of AI Agent** - 分层存储架构 (arxiv.org/abs/2506.06326)
9. **Memory in the Age of AI Agents: A Survey** - NUS/人大/复旦/北大 (arxiv.org/abs/2512.13564)
10. **Long Term Memory: The Foundation of AI Self-Evolution** (arxiv.org/abs/2410.15665)