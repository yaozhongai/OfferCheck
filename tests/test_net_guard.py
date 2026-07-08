"""SSRF 防护测试（评审 2.4）

字面 IP / scheme 校验不发网络；localhost 解析用系统 hosts（离线可判）。
"""

import pytest

from nexa_agent.util.net_guard import is_public_url, reject_reason_if_unsafe


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/",  # 云元数据端点（链路本地）
    "http://127.0.0.1:8000/admin",               # 环回
    "http://10.0.0.5/internal",                  # 私网 A
    "http://192.168.1.1/",                       # 私网 C
    "http://172.16.0.1/",                        # 私网 B
    "http://[::1]/",                             # IPv6 环回
    "http://0.0.0.0/",                          # 未指定
])
def test_private_literals_rejected(url):
    ok, reason = is_public_url(url)
    assert ok is False
    assert reason


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "gopher://127.0.0.1/",
    "dict://localhost:11211/",
])
def test_non_http_schemes_rejected(url):
    ok, _ = is_public_url(url)
    assert ok is False


def test_localhost_resolves_to_loopback_rejected():
    # localhost 走 DNS/hosts → 127.0.0.1 → 拒绝
    ok, reason = is_public_url("http://localhost/whatever")
    assert ok is False


def test_public_hostname_passes_without_resolution():
    # resolve=False 只校验 scheme/字面 IP，不发 DNS（离线稳定）
    ok, reason = is_public_url("https://api.example.com/v1", resolve=False)
    assert ok is True
    assert reason == ""


def test_reject_reason_wrapper():
    msg = reject_reason_if_unsafe("http://169.254.169.254/")
    assert msg.startswith("[错误]")
    assert "拒绝" in msg
