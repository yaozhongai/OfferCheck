"""
ReAct 子图条件路由

should_call_tool: react_decide 后走 execute 还是 finish
should_continue: execute_tool 后继续循环还是结束

步数限制只算 ReAct 迭代次数（tool_results 长度），
不包含 normalize/load_context/route_task 等前置节点。
"""

from __future__ import annotations

from app.agent.state import AgentState

def should_call_tool(state: AgentState) -> str:
    if state.get("react_finished"):
        return "react_finish"
    if state.get("pending_tool_call"):
        return "execute_tool"
    return "react_finish"


def should_continue(state: AgentState) -> str:
    if state.get("react_finished"):
        return "react_finish"
    react_steps = len(state.get("tool_results", []))
    if react_steps >= state.get("max_steps", 6):
        return "react_finish"
    return "react_decide"
