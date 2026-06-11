"""
react_execute + react_observe — 工具执行 + Observation 注入节点

执行 pending_tool_call 中的工具，结果写入 tool_results。
"""

from __future__ import annotations

import time

from app.agent.state import (
    AgentState, StepStatus, ToolCallRecord, ToolCallStatus, trace_patch,
)
from app.tools import execute_tool
from app.utils.logger_config import get_logger

logger = get_logger("node.react_execute")

MAX_OBSERVATION_CHARS = 2000


def execute_tool_call(state: AgentState) -> dict:
    """执行 pending_tool_call 中的工具，返回 Observation"""
    t0 = time.time()
    step = state.get("step_count", 0) + 1
    pending = state.get("pending_tool_call")

    if not pending:
        return {
            "tool_results": [ToolCallRecord(
                tool_call_id=f"tc_empty_{step}",
                tool_name="_no_pending",
                tool_input={},
                status=ToolCallStatus.FAILED,
                error_message="无待执行工具",
                tool_output={"observation": "[系统提示] 无待执行的工具调用，请重新输出 Action。"},
            )],
            **trace_patch(step=step, node="execute_tool", action="execute",
                          status=StepStatus.FAILED, reason="no pending tool"),
            "step_count": step,
        }

    tool_name = pending.tool_name
    tool_args = pending.tool_input.get("args", "") if pending.tool_input else ""

    try:
        result_str = execute_tool(tool_name, tool_args)
        status = ToolCallStatus.FAILED if result_str.startswith("[错误]") else ToolCallStatus.SUCCESS
    except Exception as exc:
        result_str = f"[错误] 工具执行异常: {exc}"
        status = ToolCallStatus.FAILED

    elapsed_ms = int((time.time() - t0) * 1000)

    # 截断过长结果
    if len(result_str) > MAX_OBSERVATION_CHARS:
        result_str = result_str[:MAX_OBSERVATION_CHARS] + "\n...(截断)"

    observation = f"Observation: ({tool_name}) {result_str}"

    # 清晰的 Observation 日志
    obs_preview = result_str[:150].replace("\n", " ")
    logger.info("  Observation: %s(%s) → %s (%dms)",
                 tool_name, tool_args[:40], obs_preview, elapsed_ms)

    return {
        "tool_results": [ToolCallRecord(
            tool_call_id=pending.tool_call_id,
            tool_name=tool_name,
            tool_input=pending.tool_input or {},
            status=status,
            tool_output={"observation": observation},
            latency_ms=elapsed_ms,
            error_message=None if status == ToolCallStatus.SUCCESS else result_str,
        )],
        "pending_tool_call": None,
        **trace_patch(step=step, node="execute_tool", action="execute",
                      status=StepStatus.SUCCESS if status == ToolCallStatus.SUCCESS else StepStatus.FAILED,
                      reason=f"{tool_name}({tool_args[:60]})",
                      output_summary=result_str[:200],
                      latency_ms=elapsed_ms),
        "step_count": step,
    }


def react_finish(state: AgentState) -> dict:
    """ReAct 子图完成节点 — 写回 final_answer。无答案但有待观察时，兜底汇总。"""
    step = state.get("step_count", 0) + 1
    answer = state.get("final_answer") or ""

    # 兜底汇总: 有工具结果但无 Final Answer → 再调一次 LLM
    if not answer and state.get("tool_results"):
        answer = _fallback_summary(state)

    if not answer:
        answer = "抱歉，暂时无法处理该请求。"

    logger.info("━━ ReAct 完成 ━━\n  Final Answer (%d chars): %s",
                 len(answer), answer[:200])

    return {
        "final_answer": answer,
        "react_finished": True,
        **trace_patch(step=step, node="react_finish", action="finish",
                      status=StepStatus.SUCCESS, output_summary=answer[:200]),
        "step_count": step,
    }


def _fallback_summary(state: AgentState) -> str:
    """基于所有 Observation 强制生成 Final Answer"""
    try:
        from app.llm.client import create_llm_client, LLMMessage
        llm = create_llm_client("deepseek", model="deepseek-v4-flash")

        obs_text = []
        for tr in state.get("tool_results", []):
            obs_text.append(tr.tool_output.get("observation", ""))

        messages = [
            LLMMessage.system("你是一个问答助手。基于以下工具返回的信息，回答用户问题。简洁直接。"),
            LLMMessage.user(
                f"用户问题: {state.get('user_input', '')}\n\n"
                + "--- 工具返回信息 ---\n" + "\n".join(obs_text[-3:])
                + "\n\n请基于以上信息给出最终答案。"
            ),
        ]
        resp = llm.chat(messages, temperature=0.0, max_tokens=1024)
        return resp.content or ""
    except Exception as exc:
        logger.warning("兜底汇总失败: %s", exc)
        return ""
