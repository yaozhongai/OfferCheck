"""
LangGraph 主图 — TOOL_ACT (ReAct) + FALLBACK

原则:
- LangGraph 是唯一图执行引擎
- 所有请求统一走 TOOL_ACT → react_decide ⇄ execute_tool → finish
- 异常兜底走 FALLBACK
- 不再有 VISION_DIRECT / VISION_SCHEMA / RAG_QA 独立路径
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.agent.state import AgentState
from app.agent.routers import route_after_task
from app.agent.nodes.normalize import normalize_input
from app.agent.nodes.route import route_task
from app.agent.nodes.load_context import load_short_term_context
from app.agent.nodes.respond import respond
from app.agent.nodes.memory import update_memory
from app.agent.nodes.fallback import fallback
from app.agent.nodes.react_decide import react_decide
from app.agent.nodes.react_execute import execute_tool_call, react_finish
from app.agent.nodes.react_routers import should_call_tool, should_continue
from app.utils.logger_config import get_logger

logger = get_logger("agent_graph")


def build_agent_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("normalize_input", normalize_input)
    builder.add_node("load_short_term_context", load_short_term_context)
    builder.add_node("route_task", route_task)
    builder.add_node("react_decide", react_decide)
    builder.add_node("execute_tool", execute_tool_call)
    builder.add_node("react_finish", react_finish)
    builder.add_node("respond", respond)
    builder.add_node("update_memory", update_memory)
    builder.add_node("fallback", fallback)

    builder.set_entry_point("normalize_input")
    builder.add_edge("normalize_input", "load_short_term_context")
    builder.add_edge("load_short_term_context", "route_task")

    # TOOL_ACT → ReAct 循环
    builder.add_conditional_edges("route_task", route_after_task, {
        "react_decide": "react_decide",
        "fallback": "fallback",
    })
    builder.add_conditional_edges("react_decide", should_call_tool, {
        "execute_tool": "execute_tool",
        "react_finish": "react_finish",
    })
    builder.add_conditional_edges("execute_tool", should_continue, {
        "react_decide": "react_decide",
        "react_finish": "react_finish",
    })

    builder.add_edge("react_finish", "respond")
    builder.add_edge("respond", "update_memory")
    builder.add_edge("update_memory", END)
    builder.add_edge("fallback", "respond")

    return builder.compile(checkpointer=MemorySaver())


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_agent_graph()
        logger.info("LangGraph 主图已编译 (TOOL_ACT + FALLBACK)")
    return _compiled_graph
