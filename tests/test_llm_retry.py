"""共享 LLM 重试 helper 测试（评审 1.9）

验证：瞬时错误分类正确、瞬时错误会退避重试、4xx/非瞬时错误立即抛、on_retry 回调。
用 base_delay=0 避免真实 sleep。
"""

import pytest

from nexa_agent.util.llm_retry import is_transient_llm_error, call_with_retry


class _FakeStatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def test_transient_classification():
    assert is_transient_llm_error(Exception("Connection reset by peer")) is True
    assert is_transient_llm_error(Exception("Read timed out")) is True
    assert is_transient_llm_error(_FakeStatusError(503)) is True
    assert is_transient_llm_error(_FakeStatusError(429)) is True
    # 4xx 参数/协议错误：不重试
    assert is_transient_llm_error(_FakeStatusError(400)) is False
    assert is_transient_llm_error(_FakeStatusError(422)) is False
    assert is_transient_llm_error(ValueError("bad arg")) is False


# ── GMI DeepSeek 推理模型 reasoning_content 400 特判（P0 线上修复）─────────────

# 线上实测的原始错误串（run_20260709_082117 step 6）
_GMI_REASONING_400 = (
    "Error code: 400 - {'error': {'message': 'Backend request failed with status 400', "
    "'type': 'backend_error', 'code': 400, 'details': '{\"error\":{\"message\":\"The "
    "`reasoning_content` in the thinking mode must be passed back to the API.\","
    "\"type\":\"invalid_request_error\",\"param\":\"\",\"code\":\"invalid_request_error\"}}'}}"
)


def test_gmi_reasoning_content_400_is_retryable():
    # 这条特定 400（间歇性后端坑）判为可重试
    assert is_transient_llm_error(Exception(_GMI_REASONING_400)) is True
    # 带 status_code=400 的真实 SDK 风格异常也命中（走文本特征，不看 status）
    e = _FakeStatusError(400)
    e.args = (_GMI_REASONING_400,)
    assert is_transient_llm_error(e) is True


def test_other_400_still_not_retryable():
    # 其它 400（真参数错）不因本特判被放宽——避免无谓重试 3 次
    assert is_transient_llm_error(Exception(
        "Error code: 400 - invalid 'messages': missing 'role'")) is False
    assert is_transient_llm_error(Exception(
        "Error code: 400 - context length exceeded")) is False


def test_gmi_reasoning_400_retries_then_succeeds():
    # 端到端：撞上该 400 → 退避重试 → 后续 attempt 自愈
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise Exception(_GMI_REASONING_400)
        return "ok"

    out = call_with_retry(flaky, max_retries=3, base_delay=0)
    assert out == "ok" and calls["n"] == 2


def test_retries_then_succeeds():
    calls = {"n": 0}
    retried = []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("connection error")
        return "ok"

    out = call_with_retry(
        flaky, max_retries=3, base_delay=0,
        on_retry=lambda attempt, exc: retried.append(attempt),
    )
    assert out == "ok"
    assert calls["n"] == 3
    assert retried == [1, 2]  # 前两次失败各触发一次 on_retry


def test_non_transient_raises_immediately():
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise _FakeStatusError(400)

    with pytest.raises(_FakeStatusError):
        call_with_retry(bad, max_retries=3, base_delay=0)
    assert calls["n"] == 1  # 未重试


def test_exhausts_and_raises():
    calls = {"n": 0}

    def always_flaky():
        calls["n"] += 1
        raise Exception("timeout")

    with pytest.raises(Exception):
        call_with_retry(always_flaky, max_retries=2, base_delay=0)
    assert calls["n"] == 3  # 初次 + 2 次重试
