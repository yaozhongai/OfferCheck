#!/usr/bin/env python3
"""SearXNG 后端冒烟测试 / 健康检查。

用途：验证自建 SearXNG 的 JSON API 是否可用、多引擎是否聚合正常。
运行：python searxng/smoke_test.py
退出码：0 = 健康；非 0 = 异常（可用于 CI / 启动前探活）。
"""
from __future__ import annotations

import sys
from collections import Counter
from urllib.parse import urlencode
from urllib.request import urlopen
import json

BASE_URL = "http://localhost:8888"
TIMEOUT = 15


def search(query: str, language: str = "auto") -> list[dict]:
    params = urlencode({"q": query, "format": "json", "language": language})
    with urlopen(f"{BASE_URL}/search?{params}", timeout=TIMEOUT) as resp:
        data = json.load(resp)
    return data.get("results", [])


def main() -> int:
    cases = [
        ("DeepSeek Harness engineer", "auto"),
        ("深度求索 招聘", "zh"),
        ("GAIA benchmark level 1 accuracy", "auto"),
    ]
    all_ok = True
    for query, lang in cases:
        try:
            results = search(query, lang)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] '{query}' 请求异常: {exc}")
            all_ok = False
            continue

        if not results:
            print(f"[FAIL] '{query}' 返回 0 条结果")
            all_ok = False
            continue

        engines = Counter(r.get("engine", "?") for r in results)
        has_content = sum(1 for r in results if r.get("content"))
        print(f"[OK]  '{query}': {len(results)} 条 | 引擎 {dict(engines)} | 含摘要 {has_content}")

    print("-" * 60)
    if all_ok:
        print("SearXNG 后端健康 ✅")
        return 0
    print("SearXNG 后端异常 ❌ —— 检查容器: docker compose -f searxng/docker-compose.yml ps")
    return 1


if __name__ == "__main__":
    sys.exit(main())
