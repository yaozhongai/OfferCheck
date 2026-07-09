"""Scratchpad 跨 Trial 事实抗污染框定测试（评审 2.5）

上一轮记录的"事实"可能源自被注入/幻觉污染的观察；注入下一轮时必须保留可反驳性，
不得当作裁定的免检前提，否则错误会跨 Trial 传播且不再复核。
"""

from nexa_agent.reflexion_agent import ReflexionReActAgent


def test_block_contains_facts():
    block = ReflexionReActAgent._build_scratchpad_block(
        ["字节跳动官方域名 bytedance.com", "[无效] read_xlsx 无法取背景色"]
    )
    assert "bytedance.com" in block
    assert "[无效] read_xlsx 无法取背景色" in block


def test_block_preserves_refutability():
    block = ReflexionReActAgent._build_scratchpad_block(["某事实"])
    # 必须传达：非本轮核实、不作免检前提、以新证据为准
    assert "非本轮独立核实" in block
    assert "免检前提" in block
    assert "以新证据为准" in block


def test_block_drops_dangerous_framing():
    block = ReflexionReActAgent._build_scratchpad_block(["某事实"])
    # 旧框定「无需重新搜索 / 直接使用 / 已确认的事实数据」是污染传播的根源，不得再出现
    for bad in ("无需重新搜索", "直接使用", "已确认的事实数据"):
        assert bad not in block
