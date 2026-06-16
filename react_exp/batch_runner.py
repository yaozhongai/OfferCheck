#!/usr/bin/env python
"""
从 TXT 文件读取问题，执行 ReflexionReActAgent。

用法::

    python -m react_exp.batch_runner --input question.txt
    python -m react_exp.batch_runner --input question.txt --max-trials 3 --max-steps 12
    python -m react_exp.batch_runner --input question.txt --output result.json
    python -m react_exp.batch_runner --input question.txt --quiet

输入文件格式::

    # 以 # 开头的行为注释，会被过滤掉
    这是问题的第一段...

    这是问题的第二段...
    （空行保留，成为问题的一部分）

"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv

    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _dotenv_path = os.path.join(_project_root, ".env")
    if os.path.exists(_dotenv_path):
        load_dotenv(_dotenv_path)
except ImportError:
    pass

from react_exp.reflexion_agent import ReflexionReActAgent
from react_exp.config import (
    REFLEXION_CONFIG, REACT_CONFIG, get_config_summary, get_model_for_role,
)


def load_question(filepath: str) -> str:
    """从 TXT 文件读取一个问题（过滤 # 注释行，保留其余全部内容）

    Args:
        filepath: TXT 文件路径

    Returns:
        问题文本

    Raises:
        FileNotFoundError, ValueError
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip() for line in f.readlines()
                 if not line.lstrip().startswith("#")]

    text = "\n".join(lines).strip()
    if not text:
        raise ValueError(f"文件中没有有效内容: {filepath}")

    print(f"📄 从 {filepath} 加载问题 ({len(text)} 字符)")
    return text


def main():
    parser = argparse.ArgumentParser(
        description="从 TXT 文件读取问题，执行 ReflexionReActAgent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m react_exp.batch_runner --input question.txt
  python -m react_exp.batch_runner --input question.txt --max-trials 3 --max-steps 16
  python -m react_exp.batch_runner --input question.txt --output result.json
  python -m react_exp.batch_runner --input question.txt --quiet
        """,
    )

    parser.add_argument("--input", "-i", type=str, required=True,
                        help="问题文件路径（TXT）")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出 JSON 文件路径")
    parser.add_argument("--max-trials", "-t", type=int,
                        default=REFLEXION_CONFIG["max_trials"],
                        help=f"最大重试轮数（默认: {REFLEXION_CONFIG['max_trials']}）")
    parser.add_argument("--max-steps", "-s", type=int,
                        default=REACT_CONFIG["max_steps"],
                        help=f"每次 ReAct 最大步数（默认: {REACT_CONFIG['max_steps']}）")
    parser.add_argument("--eval-mode", "-e", type=str,
                        default=REFLEXION_CONFIG["evaluator_mode"],
                        choices=["heuristic", "llm", "hybrid"],
                        help=f"评估模式（默认: {REFLEXION_CONFIG['evaluator_mode']}）")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="静默模式")
    parser.add_argument("--config", action="store_true",
                        help="打印配置摘要后退出")

    args = parser.parse_args()

    if args.config:
        print(get_config_summary())
        return

    from react_exp.react_agent import LLM_API_KEY
    if not LLM_API_KEY:
        print("❌ 未配置 DEEPSEEK_API_KEY")
        sys.exit(1)

    # 加载问题
    try:
        question = load_question(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"❌ {exc}")
        sys.exit(1)

    # 创建 Agent
    agent = ReflexionReActAgent(
        max_trials=args.max_trials,
        max_memory_size=REFLEXION_CONFIG["max_memory_size"],
        evaluator_mode=args.eval_mode,
        max_steps=args.max_steps,
    )

    if not args.quiet:
        print(get_config_summary())

    # 执行
    t0 = time.time()
    try:
        result = agent.execute(task=question, verbose=not args.quiet)
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
        sys.exit(130)
    elapsed = time.time() - t0

    record = {
        "question": question,
        "success": result.success,
        "answer": result.answer,
        "trials_used": result.trials_used,
        "elapsed_seconds": round(elapsed, 1),
        "trial_details": result.trial_details,
        "timestamp": datetime.now().isoformat(),
        "model_routing": {
            "react_first": get_model_for_role("react_first"),
            "react_main": get_model_for_role("react_main"),
            "reflection": get_model_for_role("reflection"),
        },
    }

    if not args.quiet:
        print(result.summary())

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"📝 结果已写入: {args.output}")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
