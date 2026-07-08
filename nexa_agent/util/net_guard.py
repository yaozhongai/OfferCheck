"""SSRF 防护：拒绝抓取解析到非公网地址的 URL（评审 2.4）。

OfferCheck 的工具（web_fetch / read_pdf / read_xlsx / 图片下载）会按模型/用户
给出的 URL 直接发起下载。公网部署后，攻击者可诱导 agent 抓取内网地址或云元
数据端点（169.254.169.254），构成典型 SSRF。本模块在下载前解析目标主机的所有
IP，命中私网/环回/链路本地/保留段即拒绝。

设计为纯函数 + 无第三方依赖，便于单测（不实际发请求）。
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse
from typing import Tuple

# 允许的协议——只放行 http/https，挡掉 file:// gopher:// dict:// 等 SSRF 常用跳板
_ALLOWED_SCHEMES = ("http", "https")


def _ip_is_disallowed(ip: str) -> bool:
    """该 IP 是否属于禁止访问的非公网段。"""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True  # 解析不出合法 IP → 保守拒绝
    return (
        addr.is_private       # 10/8、172.16/12、192.168/16、fc00::/7 等
        or addr.is_loopback   # 127/8、::1
        or addr.is_link_local # 169.254/16（含云元数据端点）、fe80::/10
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified # 0.0.0.0、::
    )


def is_public_url(url: str, *, resolve: bool = True) -> Tuple[bool, str]:
    """判断 URL 是否可安全抓取（解析到公网单播地址）。

    Args:
        url: 目标 URL。
        resolve: 是否做 DNS 解析（默认 True）。测试可传 False 只校验字面 IP/scheme。

    Returns:
        (ok, reason)——ok=False 时 reason 说明拒绝原因（供拼进错误 observation）。
    """
    if not url or not isinstance(url, str):
        return False, "空 URL"
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        return False, f"不支持的协议 '{parsed.scheme}'（仅允许 http/https）"
    host = parsed.hostname
    if not host:
        return False, "URL 无有效主机名"

    # 主机本身就是 IP 字面量 → 直接判定
    try:
        ipaddress.ip_address(host)
        return (not _ip_is_disallowed(host),
                "" if not _ip_is_disallowed(host)
                else f"目标是非公网地址 {host}（疑似 SSRF 探测内网/云元数据）")
    except ValueError:
        pass  # 不是 IP 字面量，走 DNS 解析

    if not resolve:
        return True, ""

    try:
        infos = socket.getaddrinfo(host, parsed.port or None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        return False, f"域名解析失败：{exc}"
    except Exception as exc:  # noqa: BLE001
        return False, f"域名解析异常：{exc}"

    resolved = {info[4][0] for info in infos}
    if not resolved:
        return False, "域名未解析到任何地址"
    for ip in resolved:
        if _ip_is_disallowed(ip):
            return False, f"目标 {host} 解析到非公网地址 {ip}（疑似 SSRF 探测内网/云元数据）"
    return True, ""


def reject_reason_if_unsafe(url: str) -> str:
    """便捷封装：不安全时返回一条标准错误 observation；安全时返回空串。"""
    ok, reason = is_public_url(url)
    if ok:
        return ""
    return f"[错误] 出于安全考虑拒绝抓取该 URL：{reason}。请改用 web_search 获取公开信息。"
