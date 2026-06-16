# 我花了三周写路由规则，然后全删了

> 让 LLM 自己决定调哪个工具，比任何规则都准。

---

上周五晚上，我盯着后端日志看了五分钟。

用户问「这张发票金额多少」，系统返回了另一张发票的数据——因为关键词规则把这个问题分到了 RAG_QA 路径。检索了知识库，找到了一条不相关的发票记录，LLM 一本正经地给出了错误答案。

图片路径根本就没被触发。因为 L1 关键词列表里，「金额多少」没有被列进图片直答的分类。

这个 bug 的修复方式是加一句 `if "金额多少" in text`。但我突然不想再加了——这已经是第 17 条关键词了。

**路由规则是 AI 项目早期的幻觉。**

---

## 五条路径，越修越碎的三个星期

最初的架构叫 Direct First, Agent When Needed：

```
图片直答    → VISION_DIRECT   VLM 直接回答，不经过 LLM
字段提取    → VISION_SCHEMA   VLM 输出 JSON，不经过 LLM
文本问答    → RAG_QA          LLM 检索知识库后回答
复杂推理    → RAG_QA+图片     VLM 先感知，LLM 再推理
工具执行    → TOOL_ACT        占位，V1 再做
```

五条路径，十四条节点，两层路由（L1 关键词 + L2 DeepSeek 分类）。

听起来很合理。跑起来全是洞：

- 「这张发票税号多少」——「税号」不在 VISION 关键词列表，被送去 RAG_QA，连图都没看就开始检索
- 「这张发票能报销吗」—— LLM 需要先看图（VLM）再查报销政策（搜索），一条路径根本兜不住

每次修一个 bug 就加一条关键词。三周后关键词列表撑到 30 多条，但边界情况反而更多了。

**为什么？** 因为路由规则本质是在用代码模拟一个分类器。而 LLM 本身就是一个更好的分类器——它是为理解自然语言设计的。你让 LLM 去回答复杂的问题，却不让它决定「这个问题该用什么方式回答」——这就像雇了个博士但只让他做速算题。

我有两个选择：继续加关键词，或者把规则全删了。

---

## 200 行的实验说服了我

项目里有个 `react_exp` 文件夹，是一个纯 Python 的 ReAct Agent：

```
用户问题 → Thought → Action(tool) → Observation → ... → Final Answer
```

没有 LangGraph，没有 FastAPI。就是一个 `while not finished` 循环，搭配 8 个 `@register` 装饰器注册的工具。web_search、wikipedia、calculator、analyze_image 都在里面。

跑通那天我看到了这个：

```
用户：2025年诺贝尔物理学奖得主是谁？
  Step 1 → Thought: 需要最新信息
           Action: web_search("2025年诺贝尔物理学奖")
           Observation: 约翰·霍普菲尔德和杰弗里·辛顿...
  Step 2 → Final Answer
```

我突然意识到：**VLM 感知图片、搜索互联网、查百科、做数学计算——对 LLM 来说没有区别。** 都是「调一个函数，拿一个结果，继续推理」。

那为什么还要给它们分别设计路径？

删掉五条路由规则用了二十分钟。

---

## 双路径：所有请求一视同仁

改完的图只剩两条路：

```text
TOOL_ACT   →  ReAct 循环 (LLM 决策 + 工具执行)
FALLBACK   →  异常兜底
```

LangGraph 节点从 14 个压到 10 个。conditional edge 从 3 个变 2 个。

每个 ReAct 步骤都是独立的图节点：

```python
builder.add_conditional_edges("react_decide", should_call_tool, {
    "execute_tool": "execute_tool",     # 有工具要调
    "react_finish": "react_finish",     # 直接回答
})
builder.add_conditional_edges("execute_tool", should_continue, {
    "react_decide": "react_decide",     # 继续循环
    "react_finish": "react_finish",     # 结束
})
```

**为什么用节点循环而不是 while 循环？** 因为每个 Thought 和 Action 都有独立的 Trace 事件。前端能看到 LLM 是怎么一步步推理的，而不是一个大黑盒。而且 LangGraph 的 `interrupt()` 只能挂起在节点之间——未来要做高风险工具的人工确认，while 循环做不到。

---

## 两个真实的 trace

**场景 1：发票识别。** 上传一张发票图片，问「金额多少」：

```text
━━ Step 1 ━━
  Thought: 用户上传了发票图片，需要先用 analyze_image 识别
  Action: analyze_image(path | 识别发票信息)
  Observation: {invoice_code: "25447...", amount: 2640.28, ...}

━━ Final ━━
  这张发票金额为 2,640.28 元（价税合计 2,983.51 元）
```

首次 VLM 调用 75 秒。但同一张图再问——上传时的 SHA256 缓存命中，不再调 VLM，LLM 直接基于缓存文本回答，降到 ~3 秒。

**场景 2：多工具协作。** 用户问「2025年全球主跨最长的悬索桥比金门大桥长百分之几」：

```text
━━ Step 1 ━━ get_current_time()
            → 2026年6月11日 ✅ 确认2025年已过去

━━ Step 2 ━━ web_search(2025年世界最长悬索桥 主跨)
            → 1915恰纳卡莱大桥，2023米

━━ Step 3 ━━ wikipedia_search(金门大桥)
            → 主跨1280.2米

━━ Step 4 ━━ calculator((2023-1280.2)/1280.2×100)
            → 58.02%

━━ Final ━━ 长58.02%
```

**LLM 自主规划了四个不同工具，没有一行 if/else 路由规则。**

---

## 三个关键设计

**1. DeepSeek V4 Pro 做决策，不用 Flash。**

试过 Flash——更快，但压不住场景。它会在被问到「2025年诺贝尔奖得主」时直接凭训练数据编一个答案，不会意识到需要搜索。而 Pro 会先调 web_search。

**2. Thinking 模式按步切换。**

```
首步 + 工具失败后   → thinking=enabled   需要深度推理
工具成功后          → thinking=disabled  直接基于结果回答
```

不从头到尾开着 thinking——没必要在已有搜索结果时还深度推理。

**3. max_steps 只算 ReAct 迭代。**

一开始用全局 `step_count`——normalize、load_context、route_task 三个前置节点占了 3 步，ReAct 只剩 3 步可用。后来改成用 `len(tool_results)` 计数——`max_steps=6` 就是纯 6 轮 Thought→Action→Observation，前置节点不占配额。

---

## 学到的三件事

1. **路由规则是伪需求。** 你永远写不完所有边界情况，而 LLM 天然就是做决策的那一层。删掉规则只用了 20 分钟，得到的准确率比三周的关键词列表高。

2. **图片可以是个普通的工具。** VLM 识别和 web_search 对 LLM 来说没有本质区别——调一个函数，拿一个结果，继续推理。单开路径是过度设计。

3. **先写实验，再写工程。** react_exp 那个 200 行的 while 循环验证了整个 ReAct 方案。如果没有它，我可能还在加第 18 条关键词。

---

> 代码开源：`<repo-url>`
>
> 讨论欢迎：GitHub Issues / PR
