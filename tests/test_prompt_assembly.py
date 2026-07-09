"""结构化会话字段 → 引擎任务串的服务端组装测试（评审 3.4）。

拼串逻辑从前端收回 server 后，必须保证：
  ① 冻结标记 `[追问/补充信息]` / `[本阶段任务]` 仍在，且 stage_router 能解析；
  ② 长度上限由 server 强制（不信任客户端截断）；
  ③ 无结构化字段时原样透传 input（向后兼容）。
全部离线，不发网络。
"""

from server.api.prompt_assembly import (
    assemble_followup_input, assemble_initial_input, resolve_task_input,
)
from server.api.schemas import RunStageRequest, FollowupContext
from nexa_agent.stage_router import extract_followup_question


def test_followup_preserves_marker_and_question():
    out = assemble_followup_input(
        "这个 offer 的竞业条款正常吗？",
        history=[{"user": "字节的 offer", "assistant": {"verdict": "存疑"}}],
        original_task="字节跳动 offer 证伪",
    )
    assert "[对话上下文 - 供参考]" in out
    assert "[追问/补充信息]" in out
    # stage_router 的冻结契约：extract_followup_question 必须能从组装串里取回追问正文
    assert extract_followup_question(out) == "这个 offer 的竞业条款正常吗？"


def test_followup_caps_materials_server_side():
    huge = "x" * 5000
    out = assemble_followup_input("q", materials={"resume": huge, "jd": huge})
    # server 强制 1500 上限（前端截断不可信）
    assert "x" * 1500 in out
    assert "x" * 1501 not in out


def test_followup_omits_empty_optional_blocks():
    out = assemble_followup_input("q", materials={"resume": "  "}, prior_sources=[])
    # 空材料 / 空 sources 不应污染上下文 JSON
    assert "user_materials" not in out
    assert "prior_sources" not in out


def test_initial_carryover_marker():
    out = assemble_initial_input(
        "Offer / contract:\n年薪 80w",
        carryover=[{"stage": "① Research", "verdict": "靠谱"}],
    )
    assert "[本案早前阶段的已取证结论 - 供参考，新裁定仍须独立取证核实]" in out
    assert "[本阶段任务]" in out
    assert out.rstrip().endswith("年薪 80w")


def test_initial_no_carryover_is_passthrough():
    assert assemble_initial_input("原始任务", carryover=None) == "原始任务"
    assert assemble_initial_input("原始任务", carryover=[]) == "原始任务"


# ── resolve_task_input：请求字段 → 组装分派 ──────────────────────────────

def test_resolve_prefers_followup_context():
    req = RunStageRequest(
        input="来源可靠吗",
        stage="offercheck_stage4",
        followup_context=FollowupContext(
            history=[{"user": "上一问", "assistant": "上一答"}],
            original_task="字节 offer",
        ),
    )
    out = resolve_task_input(req)
    assert "[追问/补充信息]" in out
    assert extract_followup_question(out) == "来源可靠吗"


def test_resolve_carryover_when_no_followup():
    req = RunStageRequest(
        input="核实这份 offer",
        stage="offercheck_stage4",
        carryover=[{"stage": "① Research", "verdict": "靠谱"}],
    )
    out = resolve_task_input(req)
    assert "[本阶段任务]" in out and "核实这份 offer" in out


def test_resolve_plain_input_passthrough():
    # 无 followup_context / carryover：单轮 / legacy 客户端，原样透传
    req = RunStageRequest(input="字节跳动招聘真实吗", stage="offercheck_stage1")
    assert resolve_task_input(req) == "字节跳动招聘真实吗"
