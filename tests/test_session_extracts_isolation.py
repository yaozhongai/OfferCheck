"""会话提取缓存的并发隔离测试（评审 1.6）

server 每条调查跑在独立线程；ContextVar 使各线程拿到隔离的提取缓存，
并发调查不再互相 clear/污染。旧的模块全局 + react_loop 开头 clear 会串扰。
"""

import threading

from nexa_agent import tools


def _worker(tag, barrier, out):
    # 每个线程模拟一条调查：clear → 追加自己的 extract → 屏障处等对方也追加 → 读回
    tools.clear_session_extracts()
    tools._extracts().append({"title": tag})
    barrier.wait(timeout=5)          # 确保两线程在读回前都写过 → 若串扰会互相看见
    out[tag] = [e["title"] for e in tools.get_session_extracts()]


def test_extracts_isolated_across_threads():
    barrier = threading.Barrier(2)
    out = {}
    threads = [
        threading.Thread(target=_worker, args=(tag, barrier, out))
        for tag in ("A", "B")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    # 每个线程只应看到自己的 tag，互不串扰
    assert out["A"] == ["A"]
    assert out["B"] == ["B"]
