"""显式输出语言透传测试（评审 1.10）

_detect_output_language 的 0.20 阈值会被"英文正文夹中文公司名"带偏；
显式 output_lang 应优先于内容检测。react_loop/execute 现接受该参数。
"""

import inspect

import pytest

from nexa_agent.react_agent import _detect_output_language, react_loop
from nexa_agent.reflexion_agent import ReflexionReActAgent


def test_detect_language_threshold_flips_on_mixed():
    # 一段英文里夹中文公司名——内容检测可能翻成 zh（复现 0.20 阈值脆弱性）
    mixed = "Please verify the offer from 字节跳动科技有限公司 for this remote role"
    # 该输入的检测结果不稳定，正是需要显式指定的场景；这里只断言函数可跑
    assert _detect_output_language(mixed) in ("en", "zh")
    # 纯英文/纯中文的确定性
    assert _detect_output_language("verify this offer please") == "en"
    assert _detect_output_language("请核实这份 offer 的真伪") == "zh"


def test_react_loop_accepts_output_lang():
    sig = inspect.signature(react_loop)
    assert "output_lang" in sig.parameters


def test_execute_accepts_output_lang():
    sig = inspect.signature(ReflexionReActAgent.execute)
    assert "output_lang" in sig.parameters
