"""结构化会话字段 → 引擎任务串的服务端组装（评审 3.4）。

多轮记忆此前由**前端**把「最近 3 轮对话窗口 + 用户材料 + 早前阶段结论」拼成单条
user prompt，并靠 `[追问/补充信息]` / `[本阶段任务]` 等中文标记与 stage_router、
引擎耦合（跨层字符串契约）。本模块把拼串逻辑从前端收回 server：请求改以
`followup_context`（history/materials/...）+ `carryover` 等**结构化字段**传入，
server 权威组装成引擎/stage_router 期望的**同一串格式**（冻结标记不破）。

收益：
  ① 消灭跨层字符串契约——前端不再需要知道 `[追问/补充信息]` 这类标记；
  ② 客户端提供的历史结论由 server 统一框定为「供参考、未经本轮独立核实」，收窄
     「客户端伪造已取证结论」的信任面（完整可信会话存储属 SPEC L3，另议）。

组装严格沿用前端旧格式，确保引擎多轮行为零回归；长度上限统一在 server 侧施加
（不再信任客户端自觉截断）。
"""

from __future__ import annotations

import json
from typing import Any, Optional

# 长度上限（与前端旧 buildFollowupInput 一致；由 server 强制，防客户端超发）
_MATERIAL_CAP = 1500
_TASK_SUMMARY_CAP = 400
_MAX_TURNS = 3
_MAX_PRIOR_SOURCES = 8


def assemble_followup_input(
    question: str,
    history: Optional[list[dict]] = None,
    materials: Optional[dict] = None,
    original_task: str = "",
    prior_sources: Optional[list[str]] = None,
) -> str:
    """把结构化 followup 上下文组装成引擎任务串。

    产出格式与前端旧 buildFollowupInput 一致：
        [对话上下文 - 供参考]
        {ctx JSON}

        [追问/补充信息]
        {question}
    `[追问/补充信息]` 标记是 stage_router 的冻结契约（extract_followup_question 依赖它）。
    """
    ctx: dict[str, Any] = {
        "original_task": (original_task or "")[:_TASK_SUMMARY_CAP],
        "conversation_history": (history or [])[-_MAX_TURNS:],
    }
    if prior_sources:
        ctx["prior_sources"] = prior_sources[:_MAX_PRIOR_SOURCES]
    mat: dict[str, str] = {}
    if materials:
        resume = (materials.get("resume") or "").strip()
        jd = (materials.get("jd") or "").strip()
        if resume:
            mat["resume"] = resume[:_MATERIAL_CAP]
        if jd:
            mat["jd"] = jd[:_MATERIAL_CAP]
    if mat:
        ctx["user_materials"] = mat

    return (
        "[对话上下文 - 供参考]\n"
        + json.dumps(ctx, ensure_ascii=False, indent=2)
        + f"\n\n[追问/补充信息]\n{question}"
    )


def assemble_initial_input(task: str, carryover: Optional[list[dict]] = None) -> str:
    """把本案早前阶段已取证结论作为参考前缀拼到本阶段任务前。

    产出格式与前端旧 buildCrossStageContext + startInitial 前缀一致；接地不弱化：
    明示「新裁定仍须独立取证核实」，evidence gate / Verifier 照常。
    """
    if not carryover:
        return task
    block = (
        "[本案早前阶段的已取证结论 - 供参考，新裁定仍须独立取证核实]\n"
        + json.dumps(carryover, ensure_ascii=False, indent=2)
    )
    return f"{block}\n\n[本阶段任务]\n{task}"


def resolve_task_input(req) -> str:
    """按请求的结构化字段决定最终引擎任务串（run_stage 与 stream 共用）。

    - `followup_context` 存在（结构化 followup）→ server 组装 [对话上下文]/[追问/补充信息]；
    - 否则 `carryover` 存在（结构化 initial 跨阶段携带）→ 组装 [本阶段任务] 前缀；
    - 都没有 → 原样用 `req.input`（单轮 / legacy 客户端，向后兼容）。
    """
    fc = getattr(req, "followup_context", None)
    if fc is not None:
        return assemble_followup_input(
            req.input,
            history=fc.history,
            materials=fc.materials,
            original_task=fc.original_task or "",
            prior_sources=fc.prior_sources,
        )
    if getattr(req, "carryover", None):
        return assemble_initial_input(req.input, carryover=req.carryover)
    return req.input
