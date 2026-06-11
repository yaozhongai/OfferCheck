"""
LangGraph conditional edge 路由 — TOOL_ACT + FALLBACK
"""

from app.agent.state import AgentState


def route_after_task(state: AgentState) -> str:
    route = state.get("route_result")
    if route is None:
        return "fallback"
    # 所有有效路由统一进入 TOOL_ACT (ReAct)
    return "react_decide"
