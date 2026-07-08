"""Evaluator judge rubric 结构测试（评审 2.6）

判 prompt 本身是否带上了明确 rubric 锚点——防止 prompt 被无意改回"偏乐观"的弱版。
（judge 的实际行为需 live LLM 验证，此处只做结构护栏。）
"""

import pytest

from nexa_agent.evaluator import Evaluator


@pytest.fixture
def ev():
    return Evaluator(mode="llm")


def test_generic_prompt_has_rubric_anchors(ev):
    prompt = ev._build_llm_eval_prompt(
        task="X 公司 2025 年融资多少",
        answer="该公司多轮获得融资，具体金额需进一步查证",
        trajectory="",
        stage=None,
    )
    # 明确的 A–D 分档 + 宁严勿松 + 正反例
    assert "宁严勿松" in prompt
    assert "premature_answer" in prompt
    assert "反例" in prompt and "正例" in prompt


def test_stage4_prompt_keeps_three_state(ev):
    prompt = ev._build_llm_eval_prompt(
        task="核实这份 offer",
        answer="[Verdict] 存疑",
        trajectory="",
        stage="stage4",
    )
    for label in ("靠谱", "存疑", "大概率有坑"):
        assert label in prompt
