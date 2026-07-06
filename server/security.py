"""
公网防滥用护栏（Tier-1）—— 无账号鉴权，只堵滥用与破坏性操作。

设计前提：单 Railway 实例、单进程 → 用进程内内存计数即可，不引入 Redis/slowapi。
三层护栏：
  1) 按 IP 滑窗限流（best-effort，XFF 可伪造）；
  2) 全局每小时 run 上限（硬顶，伪造 IP 也绕不过 → GMI 花费的真正保险）；
  3) 并发 run 上限（保护 512MB 实例不被同时的重任务打爆）。

另有：
  - require_admin：破坏性端点（reset / memory 写）需 ADMIN_TOKEN 请求头；
  - validate_image_path：image_path 必须落在 data/uploads 内 → 关闭任意文件读(LFI)；
  - clamp_run_limits：服务端强制 max_steps/max_trials 上限，防止放大花费。

所有阈值可用环境变量覆盖。
"""

from __future__ import annotations

import collections
import os
import secrets
import threading
import time
from typing import Optional

from fastapi import HTTPException, Request


def _int_env(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, "") or default)
    except ValueError:
        return default


# ── 可调阈值（环境变量覆盖）─────────────────────────────────────────────
RUN_PER_MIN_PER_IP = _int_env("ABUSE_RUN_PER_MIN", 6)      # 每 IP 每分钟发起 run 次数
RUN_PER_HOUR_GLOBAL = _int_env("ABUSE_RUN_PER_HOUR", 60)   # 全局每小时 run 次数（硬顶）
UPLOAD_PER_MIN_PER_IP = _int_env("ABUSE_UPLOAD_PER_MIN", 10)
MAX_CONCURRENT_RUNS = _int_env("ABUSE_MAX_CONCURRENT_RUNS", 3)
MAX_STEPS_CAP = _int_env("ABUSE_MAX_STEPS", 12)
MAX_TRIALS_CAP = _int_env("ABUSE_MAX_TRIALS", 2)
UPLOAD_MAX_BYTES = _int_env("ABUSE_UPLOAD_MAX_MB", 10) * 1024 * 1024
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")


# ── 滑动窗口限流器（线程安全）───────────────────────────────────────────
class _SlidingWindow:
    def __init__(self, max_events: int, window_sec: int):
        self.max = max_events
        self.window = window_sec
        self._hits: dict[str, collections.deque] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            dq = self._hits.get(key)
            if dq is None:
                dq = collections.deque()
                self._hits[key] = dq
            cutoff = now - self.window
            while dq and dq[0] <= cutoff:
                dq.popleft()
            if len(dq) >= self.max:
                return False
            dq.append(now)
            return True


_run_ip_limiter = _SlidingWindow(RUN_PER_MIN_PER_IP, 60)
_run_global_limiter = _SlidingWindow(RUN_PER_HOUR_GLOBAL, 3600)
_upload_ip_limiter = _SlidingWindow(UPLOAD_PER_MIN_PER_IP, 60)


# ── 并发 run 护栏 ───────────────────────────────────────────────────────
class _ConcurrencyGuard:
    def __init__(self, limit: int):
        self.limit = limit
        self._n = 0
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            if self._n >= self.limit:
                return False
            self._n += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._n > 0:
                self._n -= 1


RUN_CONCURRENCY = _ConcurrencyGuard(MAX_CONCURRENT_RUNS)


# ── 客户端 IP（Railway 经 X-Forwarded-For 透传）─────────────────────────
def client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ── 对外护栏函数 ────────────────────────────────────────────────────────
def enforce_run_quota(request: Request) -> None:
    """run 端点入口调用：按 IP 限流 + 全局每小时硬顶。超限抛 429。"""
    ip = client_ip(request)
    if not _run_ip_limiter.allow(ip):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试（每 IP 限流）")
    if not _run_global_limiter.allow("global"):
        raise HTTPException(status_code=429, detail="演示实例当前调用量已达上限，请稍后再试")


def enforce_upload_quota(request: Request) -> None:
    ip = client_ip(request)
    if not _upload_ip_limiter.allow(ip):
        raise HTTPException(status_code=429, detail="上传过于频繁，请稍后再试")


def require_admin(request: Request) -> None:
    """破坏性端点依赖：需正确的 ADMIN_TOKEN 请求头；未配置 token 时端点整体禁用。"""
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="该端点已禁用（未配置 ADMIN_TOKEN）")
    got = request.headers.get("x-admin-token", "")
    if not secrets.compare_digest(got, ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="需要有效的管理员令牌")


def validate_image_path(path: Optional[str], project_root: str) -> Optional[str]:
    """image_path 必须落在 <project_root>/data/uploads 内，否则拒绝 —— 关闭任意文件读。"""
    if not path:
        return None
    uploads = os.path.realpath(os.path.join(project_root, "data", "uploads"))
    resolved = os.path.realpath(path)
    if resolved != uploads and not resolved.startswith(uploads + os.sep):
        raise HTTPException(status_code=400, detail="非法的 image_path（仅允许引用已上传文件）")
    if not os.path.isfile(resolved):
        raise HTTPException(status_code=400, detail="image_path 指向的文件不存在")
    return resolved


def clamp_run_limits(max_steps: Optional[int], max_trials: Optional[int]) -> tuple[Optional[int], Optional[int]]:
    """服务端强制上限，忽略客户端传入的超大值（防放大 GMI 花费）。"""
    steps = min(max_steps, MAX_STEPS_CAP) if max_steps else MAX_STEPS_CAP
    trials = min(max_trials, MAX_TRIALS_CAP) if max_trials else MAX_TRIALS_CAP
    return steps, trials
