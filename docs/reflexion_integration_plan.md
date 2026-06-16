# Reflexion 接入 ReAct 模块改造方案

> 基于论文 [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/pdf/2303.11366)
> 目标：为现有 `react_exp/` 模块增加 Reflexion 自我反思能力

---

## 一、现有架构分析

### 当前 ReAct 模块结构

```
react_exp/
├── react_agent.py          # 核心：ReAct 主循环 (run_react_agent)
├── tools.py                # 工具注册表 (TOOLS dict)
├── logger_config.py        # 日志配置
├── wiki_search.py          # Wikipedia 搜索示例脚本
├── prompts/
│   └── react_system.txt    # ReAct System Prompt
└── logs/                   # 运行日志
```

### 当前 ReAct 流程（单次试验）

```
用户问题 → System Prompt + User Message → [Thought → Action → Observation]×N → Final Answer
```

**核心特征：**
- `run_react_agent()` 为单次执行，无重试机制
- 消息列表为 append-only 的短期记忆
- 超过 `max_steps` 后强制汇总兜底
- 无评估组件判断答案正确性
- 无跨轮次记忆传递机制

---

## 二、Reflexion 改造目标

在 ReAct 的**局内小循环**（Thought→Action→Observation）之外，包裹一层**局外大循环**（Trial→Evaluator→Self-Reflection），实现：

1. **失败检测**：判断 ReAct 输出是否满足任务需求
2. **轨迹捕获**：保存完整的失败推理链
3. **语义反思**：用独立的 LLM 会话分析失败原因，生成自然语言反思
4. **经验积累**：将反思存入长期记忆，注入下一轮 ReAct 的 Prompt

---

## 三、改造方案详细设计

### 3.1 整体架构（改造后）

```
┌─────────────────────────────────────────────────────────────┐
│                   Reflexion 局外大循环                        │
│                                                             │
│   ┌──────────┐     ┌──────────┐     ┌──────────────────┐   │
│   │  ReAct   │     │Evaluator │     │ Self-Reflection  │   │
│   │ 局内循环  │────▶│  评估器   │────▶│   反思生成器     │   │
│   └──────────┘     └──────────┘     └──────────────────┘   │
│        ▲                                      │             │
│        │                                      ▼             │
│        │              ┌──────────────────────────┐          │
│        └──────────────│   Long-Term Memory (Ω)   │          │
│                       │   长期记忆缓冲池          │          │
│                       └──────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 新增文件规划

```
react_exp/
├── react_agent.py              # [改造] 抽取内循环为独立函数
├── reflexion_agent.py          # [新增] Reflexion 局外大循环控制器
├── evaluator.py                # [新增] 评估器组件
├── memory.py                   # [新增] 长期记忆管理器
├── prompts/
│   ├── react_system.txt        # [改造] 支持动态注入长期记忆
│   └── reflection_system.txt   # [新增] 反思模型专用 Prompt
├── tools.py                    # [不变]
├── logger_config.py            # [不变]
└── logs/
```

---

## 四、各组件详细设计

### 4.1 ReAct Agent 改造（react_agent.py）

**改造要点：**

1. **将 `run_react_agent` 的返回值扩展**：除了返回最终答案，还需返回完整的推理轨迹（trajectory）和结束状态

2. **新增 `long_term_memory` 参数**：允许外部传入反思记忆列表

3. **Prompt 模板化**：在 System Prompt 加载时支持动态插入记忆段落

**改造后函数签名：**

```python
def run_react_agent(
    question: str,
    image_path: str | None = None,
    max_steps: int = 8,
    long_term_memory: list[str] | None = None,  # 新增：长期记忆注入
) -> dict:
    """
    返回:
    {
        "answer": str,           # 最终答案
        "trajectory": str,       # 完整推理轨迹文本
        "steps_used": int,       # 实际使用步数
        "terminated_reason": str # "final_answer" | "max_steps" | "parse_error"
    }
    """
```

**Prompt 动态注入逻辑：**

```python
def _inject_memory_to_prompt(system_prompt: str, memories: list[str]) -> str:
    """在 System Prompt 末尾或 User Message 中注入长期记忆"""
    if not memories:
        return system_prompt
    
    memory_section = "\n\n## 历史反思经验（务必吸取教训）\n\n"
    memory_section += "以下是你之前尝试该类型任务时的失败教训，请在推理时避免重蹈覆辙：\n\n"
    for i, mem in enumerate(memories, 1):
        memory_section += f"教训 {i}: {mem}\n\n"
    
    return system_prompt + memory_section
```

---

### 4.2 评估器组件（evaluator.py）

评估器负责 ReAct 完成后判断结果是否满足任务需求。

**设计方案：多策略评估器**

```python
class Evaluator:
    """评估 ReAct 输出的质量"""
    
    def evaluate(self, task: str, answer: str, trajectory: str) -> EvalResult:
        """
        返回:
        EvalResult(
            success: bool,          # 是否成功
            confidence: float,      # 置信度 [0, 1]
            reason: str,            # 判定理由
            feedback_signal: str    # 反馈信号（传给反思模型）
        )
        """
```

**评估策略分层设计：**

| 策略类型 | 适用场景 | 实现方式 |
|---------|---------|---------|
| 规则评估 | 有标准答案的任务（测试/调试） | Exact Match / Contains / Regex |
| LLM 评估 | 开放域问答 | 用 LLM 判断答案质量和完整性 |
| 工具验证 | 代码生成、计算任务 | 执行代码/计算验证正确性 |
| 人工评估 | 交互模式 | 用户手动确认 Yes/No |
| 启发式评估 | 通用兜底 | 检测轨迹异常模式 |

**启发式异常检测规则（不需要标准答案的场景）：**

```python
FAILURE_HEURISTICS = [
    "达到最大步数但未给出 Final Answer",              # max_steps 超限
    "轨迹中出现重复的 Action 调用（死循环）",          # 相同工具+相同参数 ≥ 2次
    "连续出现 'Error:' 类型的 Observation",            # 工具调用连续失败
    "Final Answer 为空或过短（< 10 字符）",            # 答案质量过低
    "轨迹中 Thought 出现自我矛盾",                    # LLM 评估检测
]
```

**LLM 评估 Prompt 模板：**

```
你是一个严格的任务评估专家。请判断以下智能体的回答是否充分解决了用户的问题。

【用户任务】: {task}
【智能体最终答案】: {answer}

评估标准：
1. 答案是否直接回应了用户的问题核心？
2. 答案是否有实际内容支撑（而非空泛陈述）？
3. 答案中是否存在明显的事实性错误或自相矛盾？

请输出 JSON 格式：
{"success": true/false, "confidence": 0.0~1.0, "reason": "简要判定理由"}
```

---

### 4.3 反思生成器（Self-Reflection）

反思生成器是独立的 LLM 会话，在全局后觉视角下分析失败轨迹。

**反思 Prompt 设计（reflection_system.txt）：**

```
你是一个专业的任务诊断分析师。你需要分析一个 AI 智能体在执行任务时的完整推理和行动轨迹，
找出导致失败的关键错误，并生成简短、具体、可操作的反思总结。

## 输出要求

用第一人称写一段 2-3 句话的反思，必须包含：
1. 【错误定位】：具体指出在哪一步出错，做了什么错误的假设或行动
2. 【根因分析】：为什么会犯这个错误（幻觉？遗漏信息？工具误用？死循环？）
3. 【纠正策略】：下一次遇到同类问题时，具体应该怎么做才能避免

## 反面示例（禁止这样写）

❌ "我很抱歉，我下次会更仔细地思考。"
❌ "我需要更好地利用工具。"
❌ "我应该更加小心地验证信息。"

## 正面示例（必须这样写）

✅ "我在第3步看到 Observation 中没有提到城市人口数据，却直接假设了一个数字进行计算，
   导致最终答案错误。根因是我在缺乏数据时产生了数字幻觉。
   下次在 Observation 中找不到所需数据时，我必须先用 web_search 补充查询，
   而不是凭空编造数据。"

✅ "我在第2步和第4步重复调用了 wikipedia_search('CNN')，浪费了两步却得到相同结果。
   根因是我没有注意到第一次搜索结果中'CNN'被消歧到了电视台页面而非神经网络。
   下次搜索结果不符合预期时，我应立即调整搜索关键词（如加上 'neural network'），
   而不是重复相同的搜索。"
```

**反思生成的调用逻辑：**

```python
def generate_reflection(task: str, trajectory: str, eval_feedback: str) -> str:
    """
    独立 LLM 会话生成反思
    
    Args:
        task: 原始任务描述
        trajectory: 完整失败轨迹
        eval_feedback: 评估器给出的失败原因
    
    Returns:
        2-3句话的具体反思文本
    """
    user_prompt = f"""
【任务目标】: {task}

【评估反馈】: {eval_feedback}

【完整推理轨迹】:
{trajectory}

请根据以上信息，生成一段简短、具体的反思总结。
"""
    # 使用与 ReAct 相同的模型但独立会话
    reflection = call_llm(
        system_prompt=load_reflection_prompt(),
        user_prompt=user_prompt,
        temperature=0.3  # 略带创造性但不过于发散
    )
    return reflection
```

---

### 4.4 长期记忆管理器（memory.py）

**核心设计原则（来自论文最佳实践）：**

1. **有界窗口（Bounded）**：记忆条数限制在 1~3 条（论文推荐），避免上下文超限
2. **滑动窗口策略（FIFO）**：超出限制时淘汰最早的反思
3. **任务关联性**：记忆与具体任务类型绑定，不同类型任务不共享记忆
4. **持久化可选**：支持会话内短期存储和跨会话的文件持久化

**记忆管理器设计：**

```python
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json


@dataclass
class ReflectionEntry:
    """单条反思记忆"""
    reflection: str              # 反思文本
    task: str                    # 关联的始任务
    trial_number: int            # 第几次尝试产生的
    timestamp: str               # 生成时间
    eval_feedback: str = ""      # 评估器反馈
    trajectory_summary: str = "" # 轨迹摘要（可选，用于调试）


class ReflexionMemory:
    """Reflexion 长期记忆管理器"""
    
    def __init__(
        self,
        max_size: int = 3,                    # 论文推荐 1~3
        persist_path: Path | None = None,     # 持久化路径（可选）
        eviction_strategy: str = "fifo",      # 淘汰策略
    ):
        self.max_size = max_size
        self.persist_path = persist_path
        self.eviction_strategy = eviction_strategy
        self._buffer: list[ReflectionEntry] = []
        
        # 若有持久化文件，加载
        if persist_path and persist_path.exists():
            self._load()
    
    def add(self, entry: ReflectionEntry) -> None:
        """添加一条反思，超限时按策略淘汰"""
        self._buffer.append(entry)
        
        while len(self._buffer) > self.max_size:
            if self.eviction_strategy == "fifo":
                self._buffer.pop(0)  # 淘汰最旧的
            elif self.eviction_strategy == "lru":
                # 未来可扩展：淘汰最少被引用的
                self._buffer.pop(0)
        
        if self.persist_path:
            self._save()
    
    def get_memories_for_prompt(self) -> list[str]:
        """返回用于注入 Prompt 的反思文本列表"""
        return [entry.reflection for entry in self._buffer]
    
    def clear(self) -> None:
        """清空记忆（任务成功后可选调用）"""
        self._buffer.clear()
        if self.persist_path:
            self._save()
    
    def size(self) -> int:
        return len(self._buffer)
    
    def _save(self) -> None:
        """持久化到 JSON 文件"""
        data = [
            {
                "reflection": e.reflection,
                "task": e.task,
                "trial_number": e.trial_number,
                "timestamp": e.timestamp,
                "eval_feedback": e.eval_feedback,
            }
            for e in self._buffer
        ]
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.persist_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    
    def _load(self) -> None:
        """文件加载"""
        data = json.loads(self.persist_path.read_text(encoding="utf-8"))
        self._buffer = [
            ReflectionEntry(**item) for item in data
        ]
```

**记忆淘汰策略对比：**

| 策略 | 描述 | 适用场景 | 优缺点 |
|------|------|---------|--------|
| FIFO（先进先出） | 淘汰最早的反思 | 通用场景，论文默认 | 简单高效，但可能丢失重要早期教训 |
| 相关性淘汰 | 淘汰与当前任务相关度最低的 | 多类型任务混合 | 需额外计算相似度，复杂度高 |
| 质量加权 | 保留高质量（导致后续成功的）反思 | 长期运行系统 | 需要正反馈追踪 |
| 压缩合并 | 将多条相似反思合并为一条 | 反复犯同类错误 | 减少冗余，但实现复杂 |

**推荐**：初始实现采用 **FIFO**，`max_size=3`，与论文一致。后续可按需升级。

---

### 4.5 Reflexion 控制器（reflexion_agent.py）

**核心逻辑：局外大循环**

```python
class ReflexionReActAgent:
    """
    Reflexion 增强的 ReAct 智能体
    在 ReAct 局内循环外包裹 Trial→Evaluate→Reflect 局外大循环
    """
    
    def __init__(
        self,
        max_trials: int = 3,           # 最大重试轮数
        max_memory_size: int = 3,      # 长期记忆容量
        evaluator_mode: str = "llm",   # 评估模式
        persist_memory: bool = False,  # 是否持久化记忆
    ):
        self.max_trials = max_trials
        self.memory = ReflexionMemory(
            max_size=max_memory_size,
            persist_path=Path("react_exp/memory/reflections.json") if persist_memory else None,
        )
        self.evaluator = Evaluator(mode=evaluator_mode)
    
    def execute(self, task: str, image_path: str | None = None) -> ReflexionResult:
        """执行带反思的完整任务流程"""
        
        for trial in range(1, self.max_trials + 1):
            logger.info(f"=== Reflexion Trial {trial}/{self.max_trials} ===")
            
            # 步骤 1: 获取长期记忆
            memories = self.memory.get_memories_for_prompt()
            
            # 步骤 2: 运行 ReAct（带记忆注入）
            react_result = run_react_agent(
                question=task,
                image_path=image_path,
                long_term_memory=memories,
            )
            
            # 步骤 3: 评估结果
            eval_result = self.evaluator.evaluate(
                task=task,
                answer=react_result["answer"],
                trajectory=react_result["trajectory"],
            )
            
            if eval_result.success:
                logger.info(f"✅ Trial {trial} 成功!")
                return ReflexionResult(
                    success=True,
                    answer=react_result["answer"],
                    trials_used=trial,
                    reflections=memories,
                )
            
            # 步骤 4: 失败 → 触发反思
            logger.info(f"❌ Trial {trial} 失败: {eval_result.reason}")
            
            reflection = generate_reflection(
                task=task,
                trajectory=react_result["trajectory"],
                eval_feedback=eval_result.feedback_signal,
            )
            
            logger.info(f"💡 反思: {reflection}")
            
            # 步骤 5: 更新长期记忆
            self.memory.add(ReflectionEntry(
                reflection=reflection,
                task=task,
                trial_number=trial,
                timestamp=datetime.now().isoformat(),
                eval_feedback=eval_result.reason,
            ))
        
        # 所有轮次用完仍失败
        return ReflexionResult(
            success=False,
            answer=react_result["answer"],  # 返回最后一轮的答案
            trials_used=self.max_trials,
            reflections=self.memory.get_memories_for_prompt(),
        )
```

---

## 五、Prompt 改造细节

### 5.1 react_system.txt 改造

在文件末尾新增一个**可选段落占位符**，由代码动态注：

```
（原有内容保持不变）

{REFLEXION_MEMORY_SECTION}
```

代码中如果 `long_term_memory` 为空则不注入任何内容，保持向后兼容。

### 5.2 记忆注入的位置选择

| 注入位置 | 优点 | 缺点 |
|---------|------|------|
| System Prompt 末尾 | 权重高，模型更易遵循 | 可能与规则混淆 |
| User Message 前缀 | 语义自然，像"之前的经验" | 对话轮次增加 |
| 独立 System Message | 结构清晰，不污染原有 Prompt | 部分模型不支持多 System |

**推荐方案**：在 **User Message 前缀** 中注入，格式如下：

```
【重要提醒：你之前在类似任务中犯过以下错误，务必避免重蹈覆辙】

教训 1: 我在搜索时使用了过于宽泛的关键词"AI"，得到不相关结果后又重复搜索。
下次应使用更具体的关键词如"transformer architecture NLP"。

教训 2: ...

---

【当前任务】: {原始用户问题}
```

---

## 六、记忆管理策略详解

### 6.1 记忆生命周期

```
┌─────────┐    失败    ┌──────────┐    生成    ┌──────────┐    注入    ┌─────────┐
│  ReAct  │──────────▶│ Evaluator│──────────▶│Reflector │──────────▶│ Memory  │
│  执行   │           │  评估    │           │  反思    │           │  存储   │
└─────────┘           └──────────┘           └──────────┘           └─────────┘
                                                                         │
     ┌───────────────────────────────────────────────────────────────────┘
     │ 下一轮注入
     ▼
┌─────────┐
│  ReAct  │ (带记忆的新一轮)
│  执行   │
└─────────┘
```

### 6.2 记忆清除策略

| 场景 | 操作 |
|------|------|
| 任务成功完成 | 可选清除（成功说明记忆已发挥作用） |
| 用户切换新任务 | 清除旧任务记忆（不同任务的记忆无关联） |
| 会话结束 | 若非持久化模式则自动清除 |
| 同一任务多次成功 | 保留记忆（可能仍有参考价值） |

### 6.3 记忆质量保障

为防止低质量反思污染后续推理，可增加以下过滤机制：

1. **长度过滤**：反思文本不足 20 字符或超过 500 字符时视为无效
2. **格式校验**：必须包含"错误定位"和"纠正策略"两个要素
3. **去重检测**：与已有记忆的语义相似度 > 0.9 时跳过存储
4. **成功验证**：如果某条反思在后续轮次中验证有效（任务成功），给予更高保留优先级

---

## 七、配置设计

建议在 `react_exp/config.py` 中统一管理 Reflexion 相关配置：

```python
# Reflexion 配置
REFLEXION_CONFIG = {
    "enabled": True,                    # 是否启用 Reflexion
    "max_trials": 3,                    # 最大重试轮数
    "max_memory_size": 3,               # 长期记忆池大小 (Ω)
    "evaluator_mode": "heuristic",      # 评估模式: "llm" | "heuristic" | "human"
    "persist_memory": False,            # 是否持久化记忆到文件
    "memory_persist_path": "react_exp/memory/reflections.json",
    "reflection_model": "DeepSeek-V4-Pro",  # 反思模型（可与 ReAct 不同）
    "reflection_temperature": 0.3,      # 反思生成温度
    "min_reflection_length": 20,        # 反思最短长度
    "max_reflection_length": 500,       # 反思最长长度
}
```

---

## 八、向后兼容性保障

改造需确保原有 `run_react_agent()` 的独立使用不受影响：

1. `long_term_memory` 参数默认为 `None`，不传则行为与改造前完全一致
2. 返回值从 `str` 改为 `dict`，但提供兼容包装函数
3. Reflexion 功能通过 `reflexion_agent.py` 独立入口使用
4. 所有新增文件为独立模块，不修改 `tools.py` 和 `logger_config.py`

**兼容包装示例：**

```python
# 保持原有调用方式可用
def run_react_agent_simple(question: str, **kwargs) -> str:
    """向后兼容的简单接口，直接返回答案字符串"""
    result = run_react_agent(question, **kwargs)
    return result["answer"] if isinstance(result, dict) else result
```

---

## 九、实现优先级与分步计划

### Phase 1: 基础框架（MVP）

| 序号 | 任务 | 文件 | 复杂度 |
|------|------|------|--------|
| 1 | 改造 `run_react_agent` 返回 dict（含轨迹） | react_agent.py | 低 |
| 2 | 新增 `long_term_memory` 参数 + 注入逻辑 | react_agent.py | 低 |
| 3 | 实现 `ReflexionMemory` 记忆管理器 | memory.py | 中 |
| 4 | 实现启发式评估器 | evaluator.py | 中 |
| 5 | 编写 `reflection_system.txt` | prompts/ | 低 |
| 6 | 实现 `ReflexionReActAgent` 控制器 | reflexion_agent.py | 中 |

### Phase 2: 增强功能

| 序号 | 任务 | 文件 | 复杂度 |
|------|------|------|--------|
| 7 | LLM 评估器实现 | evaluator.py | 中 |
| 8 | 记忆持久化（JSON 文件） | memory.py | 低 |
| 9 | 配置化管理 | config.py | 低 |
| 10 | 命令行入口支持 Reflexion 模式 | reflexion_agent.py | 低 |

### Phase 3: 高级优化

| 序号 | 任务 | 复杂度 |
|------|------|--------|
| 11 | 记忆去重与相似度过滤 | 高 |
| 12 | 反思质量评分与筛选 | 高 |
| 13 | 多任务记忆隔离 | 中 |
| 14 | 成功轮次正反馈追踪 | 中 |

---

## 十、关键注意事项

### 10.1 模型能力门槛

论文明确指出 Reflexion 是一种涌现能力，在弱模型上几乎无效。当前使用的 **DeepSeek-V4-Pro** 属于强模型，适合应用 Reflexion。但需注意：
- 若后续切换到较弱模型，Reflexion 可能退化
- 反思模型建议使用与 ReAct 相同或更强的模型

### 10.2 上下文窗口管理

- 长期记忆注入会占用上下文空间，需控制每条反思长度 ≤ 500 字符
- 3 条记忆 + 原始 Prompt ≈ 额外增加约 2000 tokens
- 若 ReAct 步数较多（8步），轨迹本身已较长，需注意总 token 不超限

### 10.3 成本控制

每次 Reflexion Trial 包含：
- 1次完整 ReAct 执行（多次 LLM 调用）
- 1次 Evaluator LLM 调用（若使用 LLM 评估）
- 1次 Reflection LLM 调用

`max_trials=3` 意味着最坏情况下 LLM 调用量是原来的 **3倍 + 6次额外调用**。建议：
- 默认 `max_trials=3`，不建议超过 5
- 启发式评估器（`heuristic`）不消耗额外 LLM 调用，优先使用
- 可设置"快速失败"：如果轨迹明显异常（如第1步就报错），跳过反思直接重试

### 10.4 日志与可观测性

所有 Reflexion 相关行为需完整记录日志：
- Trial 编号、开始/结束时间
- 评估结果（success/fail + reason）
- 生成的反思文本
- 记忆池当前状态（size、最新条目）
- 最终结果（成功/失败 + 使用轮次）

---

## 十一、测试方案

### 单元测试

| 组件 | 测试要点 |
|------|---------|
| ReflexionMemory | FIFO 淘汰、持久化存取、max_size 边界 |
| Evaluator | 各策略的判定准确性 |
| Prompt 注入 | 有/无记忆时的 Prompt 格式正确性 |
| 轨迹捕获 | 各种结束条件下轨迹完整性 |

### 集成测试

设计 3-5 个**可复现的失败→反思→成功**的测试用例：

1. **搜索关键词不当**：首次用模糊词搜索失败，反思后用精确词搜索成功
2. **工具选择错误**：首次用 wikipedia 搜不到时事，反思后改用 web_search
3. **计算遗漏**：首次遗漏单位换算，反思后补充换算步骤
4. **死循环检测**：首次陷入重复搜索，反思后改变策略

---

## 十二、总结

本方案的核心改造量集中在 4 个新增文件 + 1 个改造文件，总代码量约 400-600 行。改造后系统具备：

- ✅ 失败自动检测与重试
- ✅ 自然语言反思生成
- ✅ 有界滑动窗口长期记忆
- ✅ 向后兼容原有 ReAct 功能
- ✅ 可配置的评估策略
- ✅ 完整的日志观测

整体改造遵循**最小侵入原则**：改动核心工具链（tools.py），不改变原有 ReAct 的独立使用方式，Reflexion 作为可选增强层叠加在现有架构之上。