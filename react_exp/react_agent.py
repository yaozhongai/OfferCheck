"""
ReAct Agent 实验入口

基于 ReAct（Reasoning + Acting）框架的多工具智能体，使用 OpenAI 原生 SDK
调用 DeepSeek V4 Pro 作为推理核心，支持网页搜索、百科查询、图片分析（端侧/云端）、
数学计算和时间查询。

用法::

    # 纯文字问答
    python -m react_exp.react_agent "2025年诺贝尔物理学奖得主是谁？"

    # 携带图片
    python -m react_exp.react_agent "分析这张图里的设备状态" --image data/device.jpg

    # 限制步数
    python -m react_exp.react_agent "北京今天天气怎么样" --max-steps 5

工作原理::

    [用户问题]
        ↓
    Thought: 分析当前情况，决定调用哪个工具
    Action:  tool_name(arguments)
        ↓ 系统执行工具
    Observation: 工具返回结果
        ↓
    （重复，直到输出 Final Answer 或达到 max_steps）
        ↓
    Final Answer: 最终回答
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import List, Optional, Tuple

# 加载 .env
try:
    from dotenv import load_dotenv

    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _dotenv_path = os.path.join(_project_root, ".env")
    if os.path.exists(_dotenv_path):
        load_dotenv(_dotenv_path)
except ImportError:
    pass

from react_exp.logger_config import get_logger
from react_exp.tools import (
    execute_tool, TOOLS,
    get_session_extracts, clear_session_extracts, write_extract_to_disk,
)
from react_exp.config import get_model_for_role, DYNAMIC_UPGRADE_THRESHOLD

logger = get_logger("react_agent")

# ==========================================================================
# 配置
# ==========================================================================

# DeepSeek API 配置
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", os.environ.get("KIMI_API_KEY", ""))
LLM_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", os.environ.get("KIMI_BASE_URL", "https://api.deepseek.com"))
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-pro")

# 如果 LLM_MODEL 是 flash，ReAct 实验默认升级为 pro（更好的推理能力）
if "flash" in LLM_MODEL.lower():
    LLM_MODEL = "deepseek-v4-pro"
    logger.info("检测到 LLM_MODEL 为 flash 版本，ReAct 实验自动切换为 %s", LLM_MODEL)

DEFAULT_MAX_STEPS = 10

# System Prompt 路径
_SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "react_system.txt")
_CURATION_PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "curation.txt")


# ==========================================================================
# System Prompt 加载
# ==========================================================================

def load_system_prompt() -> str:
    """从文件加载 System Prompt"""
    if os.path.isfile(_SYSTEM_PROMPT_PATH):
        with open(_SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    logger.warning("System Prompt 文件不存在: %s，使用内置 Prompt", _SYSTEM_PROMPT_PATH)
    return _builtin_system_prompt()


def _builtin_system_prompt() -> str:
    """内置的简化 System Prompt（兜底）"""
    from react_exp.tools import get_tools_description

    tools_desc = get_tools_description()
    return f"""你是一个 ReAct（推理 + 行动）智能体。通过交替执行 Thought → Action → Observation 解决问题。

## 可用工具

{tools_desc}

## 输出格式

每一步必须严格遵守：

Thought: <分析当前情况>
Action: <tool_name>(<arguments>)
Observation: <工具返回结果>

...可重复...

Thought: 我有足够信息了。
Final Answer: <最终答案>

## 规则
1. 每次行动前先写 Thought
2. 每步只能调用一个工具
3. 根据 Observation 指导下一步
4. 信息足够立刻输出 Final Answer
5. 不得捏造 Observation
"""


# ==========================================================================
# LLM 调用
# ==========================================================================

def _get_llm_client():
    """获取 OpenAI 客户端实例"""
    from openai import OpenAI

    return OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        timeout=120.0,
    )


def call_llm(
    messages: List[dict],
    enable_thinking: bool = True,
    max_tokens: int = 4096,
    model: Optional[str] = None,
) -> Tuple[str, int, int]:
    """调用 DeepSeek LLM

    Args:
        messages: 对话消息列表
        enable_thinking: 是否启用 DeepSeek 推理模式
        max_tokens: 最大输出 token 数
        model: 可选模型名覆盖，为 None 则使用全局 LLM_MODEL

    Returns:
        (response_text, prompt_tokens, completion_tokens)
    """
    client = _get_llm_client()
    actual_model = model or LLM_MODEL

    # 构建 API 参数
    kwargs = {
        "model": actual_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
        "stop": ["Observation:"],  # 防止模型自行捏造 Observation
    }

    # DeepSeek thinking 模式控制（仅 Pro 模型支持 thinking）
    if enable_thinking and "pro" in actual_model.lower():
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    elif "deepseek" in actual_model.lower():
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    t0 = time.time()
    response = client.chat.completions.create(**kwargs)
    elapsed_ms = (time.time() - t0) * 1000

    choice = response.choices[0]
    content = choice.message.content or ""

    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    completion_tokens = response.usage.completion_tokens if response.usage else 0
    total_tokens = response.usage.total_tokens if response.usage else 0

    logger.info(
        "LLM 调用完成 model=%s elapsed=%.0fms tokens(in=%d out=%d total=%d) thinking=%s",
        actual_model, elapsed_ms, prompt_tokens, completion_tokens, total_tokens,
        "on" if enable_thinking and "pro" in actual_model.lower() else "off",
    )

    return content, prompt_tokens, completion_tokens


# ==========================================================================
# 响应解析
# ==========================================================================

def parse_llm_response(text: str) -> dict:
    """解析 LLM 响应，提取 Thought、Action 或 Final Answer

    ReAct 格式:
        Thought: <思考>
        Action: tool_name(arguments)

    或:
        Thought: <思考>
        Final Answer: <答案>

    Returns:
        {
            "thought": str | None,
            "action": str | None,       # 工具名
            "action_args": str | None,  # 工具参数
            "final_answer": str | None,
        }
    """
    result = {
        "thought": None,
        "action": None,
        "action_args": None,
        "final_answer": None,
    }

    # 提取 Final Answer
    fa_match = re.search(r"Final\s+Answer\s*[:：]\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if fa_match:
        result["final_answer"] = fa_match.group(1).strip()
        # 截断到 Final Answer 之前（取前面的 Thought）
        text_before_fa = text[: fa_match.start()]
    else:
        text_before_fa = text

    # 提取 Thought（取最后一个，因为可能有多个 Thought-Action 对）
    thought_matches = re.findall(r"Thought\s*[:：]\s*(.*?)(?=\n(?:Action|Final|Thought)\s*[:：]|\Z)",
                                 text_before_fa, re.DOTALL | re.IGNORECASE)
    if thought_matches:
        result["thought"] = thought_matches[-1].strip()

    # 提取 Action（只取 Final Answer 之前的最后一个 Action）
    if not result["final_answer"]:
        action_match = re.search(r"Action\s*[:：]\s*(.*)", text_before_fa, re.IGNORECASE)
        if action_match:
            action_text = action_match.group(1).strip()
            # 解析 tool_name(arguments)
            tool_match = re.match(r"(\w+)\s*\(\s*(.*?)\s*\)\s*$", action_text, re.DOTALL)
            if tool_match:
                result["action"] = tool_match.group(1)
                args = tool_match.group(2).strip()
                # 去掉 LLM 可能添加的首尾引号（URL常被错误包裹）
                if len(args) >= 2 and args[0] == args[-1] and args[0] in ('"', "'"):
                    args = args[1:-1]
                result["action_args"] = args
            else:
                logger.warning("无法解析 Action 格式: %s", action_text[:100])
                result["action"] = action_text

    return result


# ==========================================================================
# 用户消息构建
# ==========================================================================

def build_user_message(user_query: str, image_path: Optional[str] = None) -> dict:
    """构建发给 LLM 的用户消息

    当有图片时，告知 LLM 图片路径，由 LLM 自行决定调用哪个图片分析工具。
    不将图片数据直接传给 LLM（LLM 不承担视觉感知）。

    Args:
        user_query: 用户问题
        image_path: 可选的图片路径

    Returns:
        OpenAI 格式的消息 dict
    """
    if image_path:
        abs_path = os.path.abspath(image_path)
        content = (
            f"用户问题: {user_query}\n\n"
            f"注意: 用户上传了一张图片，路径为: {abs_path}\n"
            f"如果需要分析这张图片，请使用 analyze_image（端侧快速）或 "
            f"analyze_image_cloud（云端深度理解）工具。"
            f"参数格式为: 图片路径 | 分析提示词"
        )
    else:
        content = user_query

    return {"role": "user", "content": content}


# ==========================================================================
# ReAct 主循环
# ==========================================================================

def react_loop(
    user_query: str,
    image_path: Optional[str] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    verbose: bool = True,
    long_term_memory: Optional[list[str]] = None,
) -> dict:
    """ReAct 主循环：Thought → Action → Observation

    Args:
        user_query: 用户问题
        image_path: 可选的图片路径
        max_steps: 最大步数
        verbose: 是否打印中间过程
        long_term_memory: 可选的长期记忆反思列表，注入到 User Message 前缀

    Returns:
        {
            "answer": str,              # 最终答案
            "trajectory": str,          # 完整推理轨迹文本
            "steps_used": int,          # 实际使用步数
            "terminated_reason": str,   # "final_answer" | "max_steps" | "parse_error" | "llm_error"
            "total_prompt_tokens": int,
            "total_completion_tokens": int,
        }
    """
    # 加载 System Prompt
    system_prompt = load_system_prompt()
    messages = [{"role": "system", "content": system_prompt}]

    # 添加用户消息（含长期记忆注入）
    user_msg = build_user_message(user_query, image_path)

    # 如果有长期记忆，注入到 User Message 前缀
    if long_term_memory:
        memory_prefix = _build_memory_prefix(long_term_memory)
        user_msg["content"] = memory_prefix + user_msg["content"]

    messages.append(user_msg)

    step_count = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    trajectory_parts: list[str] = []  # 收集完整推理轨迹

    # ── P0: 信用分配 — 步骤效用追踪 ──
    step_utilities: list[dict] = []   # [{"step": 1, "action": "...", "args": "...", "utility": 0.5}, ...]
    action_history: list[tuple] = []  # [(tool_name, tool_args), ...] 用于检测重复调用

    # ── P0: 模型路由 — 动态升级追踪 ──
    consecutive_parse_failures = 0
    use_strong_model = False  # 动态升级标志

    # 工具调用状态（用于控制 DeepSeek thinking 模式）
    last_tool_success = None  # None=首次, True=成功, False=失败

    print(f"\n{'='*60}")
    print(f"🚀 ReAct Agent 启动")
    if image_path:
        print(f"🖼️  附带图片: {image_path}")
    print(f"🔧 可用工具: {', '.join(TOOLS.keys())}")
    print(f"🤖 推理模型: 首步={get_model_for_role('react_first')}, 后续={get_model_for_role('react_main')}")
    print(f"📏 最大步数: {max_steps}")
    print(f"{'='*60}\n")

    # 清空上次会话的 extract 缓存
    clear_session_extracts()

    while step_count < max_steps:
        step_count += 1

        # ── P0: 模型路由 — 选择当前步骤使用的模型 ──
        if use_strong_model:
            step_model = get_model_for_role("react_first")  # 动态升级用强模型
        elif step_count == 1:
            step_model = get_model_for_role("react_first")  # 首步规划用强模型
        else:
            step_model = get_model_for_role("react_main")   # 后续步骤用快模型

        # 决定是否启用 thinking（仅 Pro 模型有效）
        if last_tool_success is None:
            enable_thinking = True   # 首步：需要规划
        elif last_tool_success:
            enable_thinking = False  # 上一步成功：直接汇总
        else:
            enable_thinking = True   # 上一步失败：重新推理

        print(f"--- Step {step_count}/{max_steps} [模型: {step_model}] ---")
        logger.info("Step %d: 调用 LLM (model=%s thinking=%s, history=%d messages)",
                     step_count, step_model, "on" if enable_thinking else "off", len(messages))

        # 调用 LLM（使用路由选择的模型）
        try:
            # 首步 thinking=on 时 max_tokens 加倍，防止推理输出被截断
            step_max_tokens = 8192 if enable_thinking else 4096
            response_text, prompt_tok, completion_tok = call_llm(
                messages,
                enable_thinking=enable_thinking,
                model=step_model,
                max_tokens=step_max_tokens,
            )
            total_prompt_tokens += prompt_tok
            total_completion_tokens += completion_tok
        except Exception as exc:
            logger.error("LLM 调用失败 step=%d: %s", step_count, exc, exc_info=True)
            print(f"\n❌ LLM 调用失败: {exc}")
            trajectory = "\n".join(trajectory_parts) if trajectory_parts else "(空轨迹)"
            return {
                "answer": f"[错误] 推理模型调用失败 (step {step_count}): {exc}",
                "trajectory": trajectory,
                "steps_used": step_count,
                "terminated_reason": "llm_error",
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "step_utilities": step_utilities,
                "critical_step": _find_critical_step(step_utilities),
            }

        # 记录轨迹
        trajectory_parts.append(f"### Step {step_count}\n{response_text}")

        # 解析响应
        parsed = parse_llm_response(response_text)

        # 打印 Thought
        if parsed["thought"] and verbose:
            print(f"💭 Thought: {parsed['thought'][:200]}{'...' if len(parsed['thought']) > 200 else ''}")

        # 检查是否到达 Final Answer
        if parsed["final_answer"]:
            # ── P0: 动态升级复位 ──
            consecutive_parse_failures = 0
            use_strong_model = False

            final_answer = parsed["final_answer"]
            print(f"\n✅ Final Answer:\n{final_answer}")
            logger.info("ReAct 完成 step=%d final_answer_len=%d", step_count, len(final_answer))
            _print_summary(step_count, total_prompt_tokens, total_completion_tokens)
            _curation_step(verbose=verbose)
            trajectory = "\n".join(trajectory_parts)
            return {
                "answer": final_answer,
                "trajectory": trajectory,
                "steps_used": step_count,
                "terminated_reason": "final_answer",
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "step_utilities": step_utilities,
                "critical_step": _find_critical_step(step_utilities),
            }

        # 检查是否有 Action
        if not parsed["action"]:
            # ── P0: 动态升级 — 追踪连续解析失败 ──
            consecutive_parse_failures += 1
            if consecutive_parse_failures >= DYNAMIC_UPGRADE_THRESHOLD and not use_strong_model:
                use_strong_model = True
                logger.warning(
                    "连续 %d 次解析失败，触发动态升级 → strong 模型 + thinking",
                    consecutive_parse_failures,
                )
                print(f"⚡ 动态升级: 切换为强模型 (连续 {consecutive_parse_failures} 次解析失败)")

            # 既没有 Final Answer 也没有 Action —— 可能是格式错误
            logger.warning("Step %d: LLM 响应中未找到 Action 或 Final Answer", step_count)
            print(f"⚠️  LLM 响应格式不正确，未找到 Action 或 Final Answer")
            if verbose:
                print(f"原始响应: {response_text[:300]}...")
            # 将响应作为 Observation 注入，让 LLM 自我纠正
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": (
                    "Observation: [系统提示] 你的上一条回复格式不正确。"
                    "请严格按照以下格式输出：\n"
                    "Thought: <分析>\n"
                    "Action: <tool_name>(<arguments>)\n"
                    "或者如果已有足够信息：\n"
                    "Thought: 我有足够信息\n"
                    "Final Answer: <答案>"
                ),
            })

            # 记录信用分配：解析失败
            step_utilities.append({
                "step": step_count, "action": "_parse_error", "args": "",
                "utility": -0.5,
            })

            last_tool_success = False
            continue

        # 执行工具
        tool_name = parsed["action"]
        tool_args = parsed["action_args"] or ""

        print(f"🔧 Action: {tool_name}({tool_args[:100]}{'...' if len(tool_args) > 100 else ''})")

        # ── P0: 动态升级复位（成功解析出 Action）──
        if consecutive_parse_failures > 0:
            consecutive_parse_failures = 0
            if use_strong_model:
                logger.info("解析恢复成功，退出动态升级模式")
                use_strong_model = False

        observation = execute_tool(tool_name, tool_args)
        last_tool_success = not observation.startswith("[错误]")

        # ── P0: 信用分配 — 计算本步效用 ──
        step_utility = _compute_step_utility(
            tool_name=tool_name,
            tool_args=tool_args,
            observation=observation,
            action_history=action_history,
        )
        step_utilities.append({
            "step": step_count,
            "action": tool_name,
            "args": tool_args[:100],
            "utility": step_utility,
        })
        # 记录到历史（用于检测重复调用）
        action_history.append((tool_name, tool_args.strip()))

        # 截断过长的 Observation
        max_obs_len = 2000
        if len(observation) > max_obs_len:
            observation = observation[:max_obs_len] + f"\n...(Observation 过长，已截断至 {max_obs_len} 字符)"

        print(f"👁️  Observation: {observation[:300]}{'...' if len(observation) > 300 else ''}")

        # 记录 Observation 到轨迹
        trajectory_parts.append(f"Observation: {observation[:500]}")

        # 将 Assistant 回复和 Observation 注入对话历史
        # assistant 消息保留完整响应（含 Thought 和 Action）
        messages.append({"role": "assistant", "content": response_text})
        # Observation 由系统以 user 角色注入
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    # 达到 max_steps：触发兜底汇总
    logger.warning("达到 max_steps=%d，触发兜底汇总", max_steps)
    print(f"\n⚠️  达到最大步数 {max_steps}，触发兜底汇总...")

    fallback_prompt = (
        "你已经达到了最大步数限制。请基于以上所有 Observation 中的信息，"
        "直接给出对用户问题的最终答案。不要再调用任何工具。\n"
        "格式要求: Final Answer: <你的答案>"
    )
    messages.append({"role": "user", "content": fallback_prompt})

    try:
        response_text, prompt_tok, completion_tok = call_llm(
            messages,
            enable_thinking=False,
            model=get_model_for_role("react_main"),
            max_tokens=6144,
        )
        total_prompt_tokens += prompt_tok
        total_completion_tokens += completion_tok
        trajectory_parts.append(f"### 兜底汇总\n{response_text}")

        parsed = parse_llm_response(response_text)
        if parsed["final_answer"]:
            final_answer = parsed["final_answer"]
            print(f"\n✅ (兜底) Final Answer:\n{final_answer}")
            _print_summary(step_count, total_prompt_tokens, total_completion_tokens)
            _curation_step(verbose=verbose)
            trajectory = "\n".join(trajectory_parts)
            return {
                "answer": final_answer,
                "trajectory": trajectory,
                "steps_used": step_count,
                "terminated_reason": "max_steps",
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "step_utilities": step_utilities,
                "critical_step": _find_critical_step(step_utilities),
            }
        else:
            # 尝试提取完整响应作为答案
            print(f"\n⚠️  兜底汇总未找到 Final Answer 标记，使用完整响应")
            _print_summary(step_count, total_prompt_tokens, total_completion_tokens)
            _curation_step(verbose=verbose)
            trajectory = "\n".join(trajectory_parts)
            return {
                "answer": response_text.strip(),
                "trajectory": trajectory,
                "steps_used": step_count,
                "terminated_reason": "max_steps",
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "step_utilities": step_utilities,
                "critical_step": _find_critical_step(step_utilities),
            }

    except Exception as exc:
        logger.error("兜底汇总 LLM 调用失败: %s", exc, exc_info=True)
        _curation_step(verbose=verbose)
        trajectory = "\n".join(trajectory_parts) if trajectory_parts else "(空轨迹)"
        return {
            "answer": f"[错误] 兜底汇总失败: {exc}",
            "trajectory": trajectory,
            "steps_used": step_count,
            "terminated_reason": "llm_error",
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "step_utilities": step_utilities,
            "critical_step": _find_critical_step(step_utilities),
        }


def _compute_step_utility(
    tool_name: str,
    tool_args: str,
    observation: str,
    action_history: list[tuple],
) -> float:
    """计算单个步骤的效用值 [-1.0, +1.0]

    效用规则:
        - 搜索/百科返回有效信息: +0.5
        - 搜索/百科无结果: 0.0
        - 重复搜索相同 query: -0.5
        - fetch/extract 成功获取数据 (>500 chars): +1.0
        - fetch/extract 超时/无内容: -0.3
        - 工具报错: -0.5
        - 计算成功: +0.3
        - 其他默认: 0.0
    """
    obs_lower = observation.lower()
    tool_lower = tool_name.lower()
    args_clean = tool_args.strip()
    has_error = observation.startswith("[错误]") or observation.lower().startswith("[error]")
    has_no_result = any(kw in obs_lower for kw in ["未找到", "无结果", "no result", "not found"])

    # 工具错误
    if has_error:
        return -0.5

    # 重复检测：相同工具 + 相同参数
    if (tool_name, args_clean) in action_history:
        return -0.5

    # tavily_extract / content extraction
    if tool_name in ("tavily_extract",):
        if len(observation) > 500:
            return 1.0
        return -0.3

    # web_search / wikipedia_search
    if tool_name in ("web_search", "wikipedia_search"):
        if has_no_result:
            return 0.0
        return 0.5

    # 计算器
    if tool_name == "calculator":
        if has_error:
            return -0.5
        return 0.3

    # 图片分析
    if tool_name in ("analyze_image", "analyze_image_cloud"):
        if has_error:
            return -0.5
        return 0.3

    # 保存内容
    if tool_name == "save_content":
        if has_error:
            return -0.5
        return 0.3

    # 时间查询
    if tool_name == "get_current_time":
        return 0.1

    return 0.0


def _find_critical_step(step_utilities: list[dict]) -> Optional[dict]:
    """找出效用值最低的步骤作为 critical_step

    Args:
        step_utilities: 步骤效用列表

    Returns:
        效用最低的步骤信息，如果列表为空则返回 None
    """
    if not step_utilities:
        return None

    critical = min(step_utilities, key=lambda x: x["utility"])
    # 仅当有负效用步骤时才返回
    if critical["utility"] < 0:
        return critical
    return None


def _build_memory_prefix(memories: list[str]) -> str:
    """构建记忆注入前缀（User Message 前缀形式）

    Args:
        memories: 反思文本列表

    Returns:
        注入到 User Message 开头的记忆前缀字符串
    """
    if not memories:
        return ""

    prefix = "【重要提醒：你之前在类似任务中犯过以下错误，务必避免重蹈覆辙】\n\n"
    for i, mem in enumerate(memories, 1):
        prefix += f"教训 {i}: {mem}\n\n"
    prefix += "---\n\n"
    return prefix


def _print_summary(steps: int, prompt_tokens: int, completion_tokens: int) -> None:
    """打印执行摘要"""
    print(f"\n{'='*60}")
    print(f"📊 执行摘要: {steps} 步, "
          f"输入 {prompt_tokens} tokens, 输出 {completion_tokens} tokens, "
          f"合计 {prompt_tokens + completion_tokens} tokens")
    print(f"{'='*60}")


# ==========================================================================
# 策展步骤（ReAct 结束后独立执行）
# ==========================================================================

def _curation_step(verbose: bool = True) -> None:
    """ReAct 循环结束后，由独立 LLM 调用判断哪些 extract 值得保存

    使用独立的 curation prompt，不影响主 System Prompt。
    无缓存时直接跳过，不产生额外 API 调用。
    """
    extracts = get_session_extracts()
    if not extracts:
        return

    logger.info("策展步骤启动 extracts=%d", len(extracts))

    # 加载策展 prompt
    curation_prompt = ""
    if os.path.isfile(_CURATION_PROMPT_PATH):
        with open(_CURATION_PROMPT_PATH, "r", encoding="utf-8") as f:
            curation_prompt = f.read()
    else:
        logger.warning("策展 prompt 文件不存在: %s，跳过", _CURATION_PROMPT_PATH)
        return

    # 构建待判断的内容摘要
    items_text = []
    for i, ext in enumerate(extracts):
        title = ext.get("title", "无标题")
        url = ext.get("url", "")
        length = len(ext.get("raw_content", ""))
        # 取前 500 字供判断
        preview = ext.get("raw_content", "")[:500]
        items_text.append(
            f"### [{i}] {title}\n"
            f"来源: {url}\n"
            f"字符数: {length}\n"
            f"内容预览: {preview}...\n"
        )
    items_block = "\n".join(items_text)

    curation_messages = [
        {"role": "system", "content": curation_prompt},
        {"role": "user", "content": f"请判断以下 {len(extracts)} 条提取内容是否值得保存：\n\n{items_block}"},
    ]

    try:
        response_text, _, _ = call_llm(
            curation_messages, enable_thinking=False, max_tokens=1024,
        )
    except Exception as exc:
        logger.error("策展 LLM 调用失败: %s", exc)
        return

    # 解析 SAVE / SKIP 行
    saved = 0
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("SAVE:") or line.startswith("SAVE："):
            # 格式: SAVE: filename | reason
            body = line.split(":", 1)[1].strip() if ":" in line else ""
            if "|" in body:
                fname = body.split("|")[0].strip()
            else:
                fname = body.strip()
            if fname and saved < len(extracts):
                extract = extracts[saved]  # 按顺序对应
                try:
                    filepath = write_extract_to_disk(extract, fname)
                    logger.info("策展: SAVE → %s", filepath)
                    if verbose:
                        print(f"📥 策展保存: {fname}.md ({len(extract.get('raw_content', ''))} 字符)")
                    saved += 1
                except Exception as exc:
                    logger.error("策展写入失败 %s: %s", fname, exc)

    if verbose and saved > 0:
        print(f"📥 策展完成: 共保存 {saved}/{len(extracts)} 条\n")
    elif verbose:
        print(f"📥 策展完成: 无内容需保存 ({len(extracts)} 条均跳过)\n")


# ==========================================================================
# CLI 入口
# ==========================================================================

def main():
    global LLM_MODEL

    parser = argparse.ArgumentParser(
        description="ReAct Agent — 基于 ReAct 框架的多工具智能体实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m react_exp.react_agent "2025年诺贝尔物理学奖得主是谁？"
  python -m react_exp.react_agent "分析这张图里的设备状态" --image data/device.jpg
  python -m react_exp.react_agent "北京今天天气怎么样" --max-steps 5
        """,
    )
    parser.add_argument(
        "query",
        type=str,
        help="用户问题",
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="图片路径（相对或绝对路径）",
    )
    parser.add_argument(
        "--max-steps", "-s",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"最大步数（默认: {DEFAULT_MAX_STEPS}）",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式，不打印中间过程",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"覆盖推理模型（默认: {LLM_MODEL}）",
    )

    args = parser.parse_args()

    # 覆盖模型
    if args.model:
        LLM_MODEL = args.model

    # 验证 API Key
    if not LLM_API_KEY:
        print("❌ 错误: 未配置 DEEPSEEK_API_KEY 或 KIMI_API_KEY")
        print("   请在 .env 文件中设置 API Key")
        sys.exit(1)

    # 验证图片路径
    image_path = None
    if args.image:
        if os.path.isfile(args.image):
            image_path = args.image
        else:
            # 尝试相对项目根
            alt_path = os.path.join(_project_root, args.image)
            if os.path.isfile(alt_path):
                image_path = alt_path
            else:
                print(f"❌ 错误: 图片文件不存在: {args.image}")
                print(f"   也尝试过: {alt_path}")
                sys.exit(1)

    # 运行 ReAct 循环
    result = react_loop(
        user_query=args.query,
        image_path=image_path,
        max_steps=args.max_steps,
        verbose=not args.quiet,
    )

    # 提取答案（兼容新的 dict 返回格式）
    answer = result["answer"] if isinstance(result, dict) else result
    if not args.quiet:
        print(f"\n{'='*60}")
        print(f"📊 终止原因: {result.get('terminated_reason', 'N/A') if isinstance(result, dict) else 'N/A'}")
        print(f"{'='*60}")

    return answer


if __name__ == "__main__":
    main()
