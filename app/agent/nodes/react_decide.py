"""
react_decide — LLM 推理决策节点

调用 LLM，输出 Thought + Action 或 Final Answer。
"""

from __future__ import annotations

import json
import os
import re
import time

from app.agent.state import (
    AgentState, StepStatus, ModelCallRecord, ToolCallRecord, ToolCallStatus,
    trace_patch,
)
from app.api.deps import get_llm_client
from app.llm.client import LLMMessage
from app.tools import get_tools_description
from app.utils.logger_config import get_logger

logger = get_logger("node.react_decide")

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "tools", "prompts", "react_system.txt")


def _load_system_prompt() -> str:
    tools = get_tools_description()
    if os.path.isfile(_PROMPT_PATH):
        with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().replace("{tools_description}", tools)
    return f"""你是 ReAct 智能体。可用工具:\n{tools}\n\n格式: Thought → Action → Observation → Final Answer"""


def _parse_response(text: str) -> dict:
    """解析 LLM 响应中的 Thought / Action / Final Answer"""
    result = {"thought": None, "action": None, "action_args": None, "final_answer": None}

    fa_match = re.search(r"Final\s+Answer\s*[:：]\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if fa_match:
        result["final_answer"] = fa_match.group(1).strip()
        text_before = text[:fa_match.start()]
    else:
        text_before = text

    thought_matches = re.findall(
        r"Thought\s*[:：]\s*(.*?)(?=\n(?:Action|Final|Thought|Observation)\s*[:：]|\Z)",
        text_before, re.DOTALL | re.IGNORECASE,
    )
    if thought_matches:
        result["thought"] = thought_matches[-1].strip()

    if not result["final_answer"]:
        action_match = re.search(r"Action\s*[:：]\s*(.*)", text_before, re.IGNORECASE)
        if action_match:
            action_text = action_match.group(1).strip()
            tool_match = re.match(r"(\w+)\s*\(\s*(.*?)\s*\)\s*$", action_text, re.DOTALL)
            if tool_match:
                result["action"] = tool_match.group(1)
                args = tool_match.group(2).strip()
                # 去掉 LLM 可能添加的首尾引号
                if len(args) >= 2 and args[0] == args[-1] and args[0] in ('"', "'"):
                    args = args[1:-1]
                result["action_args"] = args
            else:
                result["action"] = action_text

    return result


def react_decide(state: AgentState) -> dict:
    t0 = time.time()
    step = state.get("step_count", 0) + 1
    user_input = state.get("user_input", "")

    # 构建消息
    system_prompt = _load_system_prompt()
    messages = [LLMMessage.system(system_prompt)]

    # 注入历史观察
    for tr in state.get("tool_results", []):
        obs = tr.tool_output.get("observation", "") if tr.tool_output else ""
        messages.append(LLMMessage.user(f"Observation: {obs[:1500]}"))

    # 用户消息（含图片引用）
    image_refs = state.get("image_refs", [])
    if image_refs and image_refs[0].path:
        content = f"用户问题: {user_input}\n\n注意: 有一张图片 {image_refs[0].path}，如需分析请用 analyze_image(路径 | 提示词)"
    else:
        content = user_input
    messages.append(LLMMessage.user(content))

    # 调用 LLM（ReAct 决策统一用 Pro，缓存单例）
    from app.llm.client import create_llm_client as _create_llm
    if not hasattr(react_decide, "_llm"):
        react_decide._llm = _create_llm("deepseek", model="deepseek-v4-pro")
    llm = react_decide._llm

    # 思考模式控制：首步 + 工具失败后 → 启用；工具成功后 → 关闭
    tool_results = state.get("tool_results", [])
    last_failed = tool_results and tool_results[-1].status.value == "failed" if tool_results else False
    enable_thinking = (len(tool_results) == 0 or last_failed)
    kwargs = {}
    if hasattr(llm, "_model") and "deepseek" in getattr(llm, "_model", ""):
        kwargs["extra_body"] = {"thinking": {"type": "enabled" if enable_thinking else "disabled"}}

    try:
        resp = llm.chat(messages, temperature=0.0, max_tokens=2048, **kwargs)
        elapsed_ms = int((time.time() - t0) * 1000)
    except Exception as exc:
        elapsed_ms = int((time.time() - t0) * 1000)
        logger.error("react_decide LLM 失败: %s", exc)
        return {
            "final_answer": f"[ReAct LLM 调用失败: {exc}]",
            "react_finished": True,
            "model_calls": [ModelCallRecord(
                model_name="unknown", node="react_decide", success=False,
                error_message=str(exc), latency_ms=elapsed_ms,
            )],
            **trace_patch(step=step, node="react_decide", action="decide",
                          status=StepStatus.FAILED, error_message=str(exc), latency_ms=elapsed_ms),
            "step_count": step,
        }

    parsed = _parse_response(resp.content)

    # 清晰的 ReAct 步骤日志
    react_step = len(state.get("tool_results", [])) + 1
    thought = (parsed.get("thought") or "")[:120]
    if parsed["action"]:
        logger.info("━━ Step %d ━━\n  Thought: %s\n  Action: %s(%s)",
                     react_step, thought, parsed["action"], parsed["action_args"] or "")
    elif parsed["final_answer"]:
        logger.info("━━ Step %d (Final) ━━\n  Thought: %s\n  Final Answer: %s",
                     react_step, thought, parsed["final_answer"][:200])

    result: dict = {
        "react_decision_summary": (parsed.get("thought") or "")[:200],
        "model_calls": [ModelCallRecord(
            provider=resp.model, model_name=resp.model, node="react_decide",
            input_summary=user_input[:100],
            output_summary=resp.content[:200],
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            total_tokens=resp.total_tokens,
            latency_ms=elapsed_ms, success=True,
        )],
        **trace_patch(step=step, node="react_decide", action="decide",
                      status=StepStatus.SUCCESS,
                      output_summary=f"thought={(parsed.get('thought') or '')[:80]}",
                      latency_ms=elapsed_ms),
        "step_count": step,
    }

    if parsed["final_answer"]:
        result["final_answer"] = parsed["final_answer"]
        result["react_finished"] = True
    elif parsed["action"]:
        result["pending_tool_call"] = ToolCallRecord(
            tool_call_id=f"tc_{step}",
            tool_name=parsed["action"],
            tool_input={"args": parsed["action_args"] or ""},
            status=ToolCallStatus.PLANNED,
        )
    else:
        # 无 Action 无 Final Answer — 格式错误，注入纠正提示
        messages.append(LLMMessage.assistant(resp.content))
        messages.append(LLMMessage.user(
            "Observation: [系统提示] 上一条回复格式不正确。请输出 Thought + Action 或 Final Answer。"
        ))
        result["tool_results"] = [ToolCallRecord(
            tool_call_id=f"tc_format_{step}",
            tool_name="_format_error",
            tool_input={"raw": resp.content[:200]},
            status=ToolCallStatus.FAILED,
            error_message="LLM 响应格式错误",
            tool_output={"observation": "[系统提示] 格式不正确，请输出 Thought + Action 或 Final Answer"},
        )]

    return result
