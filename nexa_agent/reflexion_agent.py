"""
Reflexion 增强的 ReAct 智能体控制器

在 ReAct 局内循环（Thought → Action → Observation）之外包裹一层
局外大循环（Trial → Evaluate → Reflect → Retry），实现自我反思和
从错误中学习的能力。

论文: Reflexion: Language Agents with Verbal Reinforcement Learning
      (Shinn et al., 2023)

用法::

    # 命令行
    python -m nexa_agent.reflexion_agent "2024年全球GDP排名前十的国家有哪些？"

    # 代码调用
    from nexa_agent.reflexion_agent import ReflexionReActAgent

    agent = ReflexionReActAgent(max_trials=3, evaluator_mode="hybrid")
    result = agent.execute("2025年诺贝尔物理学奖得主是谁？")
    print(result.answer)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

# 加载 .env
try:
    from dotenv import load_dotenv

    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _dotenv_path = os.path.join(_project_root, ".env")
    if os.path.exists(_dotenv_path):
        load_dotenv(_dotenv_path)
except ImportError:
    pass

from nexa_agent.logger import get_logger, start_run_log, stop_run_log
from nexa_agent.react_agent import react_loop, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
from nexa_agent.memory import ReflexionMemory, ReflectionEntry, _jaccard_similarity
from nexa_agent.evaluator import Evaluator, EvalResult, create_evaluator
from nexa_agent.verifier import VerifierAgent, should_trigger_verifier
from nexa_agent.config import (
    REFLEXION_CONFIG, REACT_CONFIG, MEMORY_CONFIG, PATH_CONFIG,
    thinking_extra_body, get_config_summary, get_model_for_role,
)
from nexa_agent.util.llm_retry import call_with_retry

logger = get_logger("reflexion_agent")

# System Prompt 路径
_REFLECTION_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "prompts", "reflection_system.txt",
)


# ==========================================================================
# 数据结构
# ==========================================================================

@dataclass
class ReflexionResult:
    """Reflexion 执行结果

    Attributes:
        success: 最终是否成功
        answer: 最终答案
        trials_used: 使用的 Trial 数
        trial_details: 每轮 Trial 的详细信息
        reflections: 最终记忆池中的反思列表
    """
    success: bool
    answer: str
    trials_used: int
    trial_details: list[dict] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)

    @property
    def total_llm_calls(self) -> int:
        """估算总 LLM 调用次数"""
        return self.trials_used * 2  # 每轮: 1次ReAct + 1次Reflection(失败时)

    def summary(self) -> str:
        """生成结果摘要"""
        lines = [
            "=" * 60,
            f"Reflexion 执行结果: {'✅ 成功' if self.success else '❌ 失败'}",
            f"使用 Trial 数: {self.trials_used}",
            f"答案: {self.answer[:200]}{'...' if len(self.answer) > 200 else ''}",
        ]
        for i, detail in enumerate(self.trial_details, 1):
            status = "✅" if detail.get("success") else "❌"
            lines.append(
                f"  Trial {i}: {status} "
                f"steps={detail.get('steps_used', '?')} "
                f"reason={detail.get('terminated_reason', '?')}"
            )
            if detail.get("failure_mode"):
                lines.append(f"    失败模式: {detail['failure_mode']}")
            if detail.get("reflection"):
                ref = detail["reflection"]
                lines.append(f"    反思: {ref[:120]}{'...' if len(ref) > 120 else ''}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ==========================================================================
# ReflexionReActAgent
# ==========================================================================

class ReflexionReActAgent:
    """Reflexion 增强的 ReAct 智能体

    在 ReAct 局内循环外包裹 Trial → Evaluate → Reflect 局外大循环。

    Usage::

        agent = ReflexionReActAgent(max_trials=3)
        result = agent.execute("2024年全球GDP排名？")
        print(result.summary())
    """

    def __init__(
        self,
        max_trials: int = 3,
        max_memory_size: int = 3,
        evaluator_mode: str = "hybrid",
        persist_memory: bool = False,
        max_steps: int = 16,
    ):
        """初始化 Reflexion 控制器

        Args:
            max_trials: 最大重试轮数（推荐 3）
            max_memory_size: 长期记忆池大小（推荐 3）
            evaluator_mode: 评估模式 — "heuristic" | "llm" | "hybrid"
            persist_memory: 是否持久化记忆到文件
            max_steps: 每次 ReAct 的最大步数
        """
        self.max_trials = max_trials
        self.max_steps = max_steps

        # 记忆管理器
        persist_path = Path(PATH_CONFIG["reflections_file"]) if persist_memory else None
        self.memory = ReflexionMemory(
            max_size=max_memory_size,
            persist_path=persist_path,
            eviction_strategy="fifo",
        )

        # 评估器
        self.evaluator = create_evaluator(mode=evaluator_mode)

        # ── 方案 B: Verifier Agent（事实网关）──
        self.verifier = VerifierAgent()

        # 反思 Prompt
        self.reflection_prompt = self._load_reflection_prompt()

        # ── P0: 教训计数器（用于去重和晋升追踪）──
        self.lesson_counter: dict[str, int] = {}

        logger.info(
            "ReflexionReActAgent 初始化 max_trials=%d max_steps=%d eval=%s memory=%d persist=%s",
            max_trials, max_steps, evaluator_mode, max_memory_size, persist_memory,
        )

    # ── 主入口 ──

    def execute(
        self,
        task: str,
        image_path: Optional[str] = None,
        verbose: bool = True,
        max_steps: Optional[int] = None,
        stage: Optional[str] = None,
        on_event: Optional[Callable[[dict], None]] = None,
        answer_mode: bool = False,
        output_lang: Optional[str] = None,
    ) -> ReflexionResult:
        """执行带反思的完整任务流程

        Args:
            task: 用户任务描述
            image_path: 可选的图片路径
            verbose: 是否打印详细过程
            stage: 可选的 OfferCheck 阶段标识（如 "offercheck_stage1"）。
                   透传给 react_loop，为该阶段加载任务定义 prompt。
            on_event: 可选的结构化事件回调，透传给 react_loop 并在 Trial
                      边界发射 trial_start/trial_end，供 server 转 SSE 流。

        Returns:
            ReflexionResult
        """
        def _emit(event_type: str, **payload) -> None:
            if on_event is None:
                return
            try:
                on_event({"type": event_type, **payload})
            except Exception:  # noqa: BLE001 — 可观测钩子绝不影响主流程
                logger.debug("on_event 回调异常，已忽略", exc_info=True)

        effective_max_steps = max_steps if max_steps is not None else self.max_steps

        # 每次运行开启独立日志文件
        run_log_path = start_run_log(tag="reflexion")
        logger.info("Run log: %s", run_log_path)
        logger.info("Task: %s", task[:200])

        if verbose:
            print(f"\n{'='*60}")
            print(f"🔄 ReflexionReActAgent 启动")
            print(f"📝 任务: {task}")
            print(f"🔁 最大 Trial 数: {self.max_trials}")
            print(f"📏 每次最大步数: {effective_max_steps}")
            print(f"🧠 记忆池容量: {self.memory.max_size}")
            print(f"🔍 评估模式: {self.evaluator.mode}")
            print(f"📄 日志: {run_log_path}")
            print(f"{'='*60}\n")

        trial_details = []

        # 每次新任务清空教训计数器，防止跨任务污染
        self.lesson_counter.clear()

        # 跨 Trial 的已访问 URL 追踪
        visited_urls: set[str] = set()

        # 跨 Trial 的事实草稿板（Scratchpad）
        scratchpad_facts: list[str] = []

        for trial in range(1, self.max_trials + 1):
            t0 = time.time()
            _emit("trial_start", trial=trial, max_trials=self.max_trials)

            if verbose:
                memory_state = f"({self.memory.size()} 条记忆)" if not self.memory.is_empty() else "(空)"
                print(f"\n{'─'*40}")
                print(f"🔄 Trial {trial}/{self.max_trials} {memory_state}")
                print(f"{'─'*40}")

            # 阶段 1: 检索长期记忆 + Scratchpad 事实 + 已访问 URL 约束
            memories = self.memory.get_memories_for_prompt()

            # 注入 scratchpad 事实（最高优先级，放在最前面）
            if scratchpad_facts:
                facts_block = (
                    "以下是之前 Trial 已确认的事实数据，无需重新搜索，直接使用：\n"
                    + "\n".join(f"- {f}" for f in scratchpad_facts)
                )
                memories = [facts_block] + (memories or [])
                if verbose:
                    print(f"📋 Scratchpad: 注入 {len(scratchpad_facts)} 条已确认事实")

            if visited_urls:
                url_constraint = (
                    "下次不要再访问以下 URL（之前的 Trial 已访问过，内容无用或无法解析）: "
                    + ", ".join(sorted(visited_urls)[:5])
                )
                memories = (memories or []) + [url_constraint]
                if verbose:
                    print(f"🚫 排除 {len(visited_urls)} 个已访问 URL")

            if memories and verbose:
                print(f"🧠 注入 {len(memories)} 条历史教训:")
                for i, m in enumerate(memories, 1):
                    print(f"   教训 {i}: {m[:100]}{'...' if len(m) > 100 else ''}")

            # 阶段 2: 运行 ReAct
            react_result = react_loop(
                user_query=task,
                image_path=image_path,
                max_steps=effective_max_steps,
                verbose=verbose,
                long_term_memory=memories if memories else None,
                stage=stage,
                on_event=on_event,
                answer_mode=answer_mode,
                output_lang=output_lang,
            )

            # 从轨迹中提取本轮访问过的 URL，累积到 visited_urls
            self._extract_visited_urls(react_result.get("trajectory", ""), visited_urls)

            answer = react_result["answer"]
            trajectory = react_result["trajectory"]
            steps_used = react_result["steps_used"]
            terminated_reason = react_result["terminated_reason"]
            trial_elapsed = time.time() - t0

            # 快速失败: API 余额不足/认证失败等不可恢复错误，直接终止所有 Trial
            if terminated_reason == "llm_error" and any(
                code in answer for code in ["402", "401", "Insufficient Balance", "Authentication"]
            ):
                logger.error("不可恢复的 API 错误，跳过剩余 Trial: %s", answer[:100])
                if verbose:
                    print(f"\n🛑 API 不可恢复错误，终止重试: {answer[:100]}")
                stop_run_log()
                return ReflexionResult(
                    success=False,
                    answer=answer,
                    trials_used=trial,
                    trial_details=[{
                        "trial": trial, "success": False, "answer": answer,
                        "steps_used": steps_used, "terminated_reason": terminated_reason,
                        "failure_mode": "api_error", "eval_reason": answer[:200],
                        "elapsed_seconds": round(trial_elapsed, 1), "reflection": None,
                    }],
                    reflections=[],
                )

            # 阶段 3: 评估（传入结构化行动日志，替代截断 trajectory）
            action_log = {
                "action_history": react_result.get("action_history", []),
                "seen_urls": react_result.get("seen_urls", []),
                "successful_retrievals": react_result.get("successful_retrievals", 0),
                "steps_used": steps_used,
                "terminated_reason": terminated_reason,
            }
            eval_result = self.evaluator.evaluate(
                task=task,
                answer=answer,
                trajectory=trajectory,
                terminated_reason=terminated_reason,
                stage=stage,
                action_log=action_log,
            )

            trial_info = {
                "trial": trial,
                "success": eval_result.success,
                "answer": answer,
                "steps_used": steps_used,
                "terminated_reason": terminated_reason,
                "failure_mode": eval_result.failure_mode,
                "eval_reason": eval_result.reason,
                "elapsed_seconds": round(trial_elapsed, 1),
                "reflection": None,
            }
            trial_details.append(trial_info)
            _emit("trial_evaluated", trial=trial, success=eval_result.success,
                  failure_mode=eval_result.failure_mode, reason=eval_result.reason[:300])

            if eval_result.success:
                # ── 方案 B: Verifier 节点 — 行为通过后进行事实核查 ──
                verdict = None
                if should_trigger_verifier(task, trial_details, trial):
                    if verbose:
                        print(f"\n🔍 Verifier 介入: 事实核查中...")
                    _emit("verifier_start", trial=trial)
                    t_v = time.time()
                    verdict = self.verifier.verify(answer=answer, task=task, stage=stage)
                    logger.info("Verifier trial=%d status=%s elapsed=%.1fs",
                                trial, verdict.status, time.time() - t_v)
                    _emit("verifier_result", trial=trial, passed=verdict.passed,
                          status=verdict.status, reason=verdict.reason[:300])

                    if not verdict.passed:
                        # Verifier 驳回 → 转化为 unreliable_source 失败
                        if verbose:
                            print(f"❌ Verifier 驳回: {verdict.reason}")
                            print(f"   → {verdict.feedback[:200]}")
                        trial_info["failure_mode"] = "unreliable_source"
                        trial_info["eval_reason"] = verdict.reason
                        trial_info["success"] = False
                        # 继续走下面的反思流程
                    else:
                        if verbose:
                            print(f"✅ Verifier 通过: {verdict.reason}")
                else:
                    if verbose:
                        print(f"⏭️  Verifier 跳过（不满足触发条件）")

                # Verifier 通过或跳过 → 成功返回
                if verdict is None or verdict.passed:
                    if verbose:
                        print(f"\n✅ Trial {trial} 成功!")
                    logger.info("Reflexion 成功 trial=%d elapsed=%.1fs",
                                trial, trial_elapsed)
                    stop_run_log()
                    return ReflexionResult(
                        success=True,
                        answer=answer,
                        trials_used=trial,
                        trial_details=trial_details,
                        reflections=self.memory.get_memories_for_prompt(),
                    )

            # 阶段 4: 失败 → 生成反思
            failure_mode = trial_info.get("failure_mode", eval_result.failure_mode)
            if verbose:
                print(f"\n❌ Trial {trial} 失败: {trial_info.get('eval_reason', eval_result.reason)}")
                print(f"   失败模式: {failure_mode}")

            logger.info("Trial %d 失败 mode=%s", trial, failure_mode)

            # 获取信用分配的关键步骤
            critical_step = react_result.get("critical_step")

            # 构建评估反馈（优先使用 Verifier 的具象反馈）
            if failure_mode == "unreliable_source" and verdict:
                eval_feedback = verdict.feedback  # Verifier 的操作指引
            else:
                eval_feedback = eval_result.feedback_signal

            reflection = self._generate_reflection(
                task=task,
                trajectory=trajectory,
                eval_feedback=eval_feedback,
                failure_mode=failure_mode or "unknown",
                critical_step=critical_step,
            )

            # ── P0: 有界教训提取 ──
            lessons = self._extract_lessons(reflection)

            trial_info["reflection"] = reflection
            trial_info["lessons"] = lessons

            if verbose:
                print(f"💡 反思: {reflection}")
                if lessons:
                    print(f"📝 教训: {'; '.join(lessons)}")

            # 阶段 5: 更新长期记忆（含教训）
            self.memory.add_with_quality_check(
                ReflectionEntry(
                    reflection=reflection,
                    task=task,
                    trial_number=trial,
                    timestamp=datetime.now().isoformat(),
                    eval_feedback=eval_result.reason,
                    trajectory_summary=trajectory[:500],
                    lessons=lessons,
                ),
                min_length=REFLEXION_CONFIG["min_reflection_length"],
                max_length=REFLEXION_CONFIG["max_reflection_length"],
            )

            if verbose:
                print(f"🧠 记忆已更新: {self.memory.size()}/{self.memory.max_size} 条")

            # 阶段 6: 从轨迹中提取已确认的事实到 Scratchpad
            new_facts = self._extract_scratchpad_facts(task, answer, trajectory)
            if new_facts:
                for fact in new_facts:
                    if fact not in scratchpad_facts:
                        scratchpad_facts.append(fact)
                if verbose:
                    print(f"📋 Scratchpad 更新: +{len(new_facts)} 条事实 (共 {len(scratchpad_facts)} 条)")
                logger.info("Scratchpad 更新: +%d facts, total=%d", len(new_facts), len(scratchpad_facts))

        # 所有 Trial 用尽
        last_answer = trial_details[-1]["answer"] if trial_details else ""
        logger.warning("Reflexion 所有 %d 轮 Trial 均失败", self.max_trials)

        if verbose:
            print(f"\n{'='*60}")
            print(f"❌ 所有 {self.max_trials} 轮 Trial 均失败")
            print(f"   最后答案: {last_answer[:200]}")
            print(f"{'='*60}")

        stop_run_log()
        return ReflexionResult(
            success=False,
            answer=last_answer,
            trials_used=self.max_trials,
            trial_details=trial_details,
            reflections=self.memory.get_memories_for_prompt(),
        )

    # ── 反思生成 ──

    def _generate_reflection(
        self,
        task: str,
        trajectory: str,
        eval_feedback: str,
        failure_mode: str,
        critical_step: Optional[dict] = None,
    ) -> str:
        """使用独立 LLM 会话生成反思

        Args:
            task: 原始任务描述
            trajectory: 完整失败轨迹
            eval_feedback: 评估器反馈
            failure_mode: 失败模式
            critical_step: 信用分配定位的关键步骤（可选）

        Returns:
            2-3 句具体的反思文本
        """
        # 三明治截断：首 1000 + 末 1940 字符
        traj_summary = self._sandwich_truncate(trajectory, head_chars=1000, tail_chars=1940)

        # 构建关键步骤提示
        critical_info = ""
        if critical_step:
            critical_info = (
                f"【关键步骤定位】: 第 {critical_step['step']} 步 "
                f"({critical_step['action']}) 效用值最低 ({critical_step['utility']:.1f})，"
                f"请重点分析这一步的问题。\n\n"
            )

        user_prompt = f"""【任务目标】: {task}

【失败模式】: {failure_mode}

【评估反馈】: {eval_feedback}

{critical_info}【推理轨迹】:
{traj_summary}

请根据以上信息，生成一段简短、具体的反思总结。"""

        try:
            reflection = self._call_reflection_llm(user_prompt)
            return reflection.strip()
        except Exception as exc:
            logger.error("反思生成 LLM 调用失败: %s", exc)
            # 降级：用评估反馈作为简易反思
            return f"任务执行失败（{failure_mode}）。{eval_feedback[:200]}"

    def _call_reflection_llm(self, user_prompt: str) -> str:
        """调用 LLM 生成反思（独立会话，使用 fast 模型）"""
        from openai import OpenAI

        client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            timeout=120.0,
        )

        model = get_model_for_role("reflection")
        temperature = REFLEXION_CONFIG["reflection_temperature"]

        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.reflection_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 512,
            "temperature": temperature,
            "stream": False,
        }

        # 反思不需要深度推理 → 关闭思考（GMI Flash 生效；DeepSeek 官方走 thinking.disabled）
        _eb = thinking_extra_body(model, enable_thinking=False)
        if _eb:
            kwargs["extra_body"] = _eb

        t0 = time.time()
        response = call_with_retry(lambda: client.chat.completions.create(**kwargs))  # 评审 1.9
        elapsed = (time.time() - t0) * 1000

        content = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0

        logger.info("反思生成完成 model=%s elapsed=%.0fms tokens=%d len=%d",
                    model, elapsed, tokens, len(content))
        return content

    # ── P0: 有界教训提取 ──

    def _extract_lessons(self, reflection: str) -> list[str]:
        """从反思文本中提取 1-3 条标准化教训

        用 LLM 将反思归纳为具体可操作的教训，每条 ≤ 50 字，以"下次..."开头。

        Args:
            reflection: 完整反思文本

        Returns:
            教训字符串列表（1-3条）
        """
        from openai import OpenAI

        lesson_prompt = f"""请将以下反思归纳为 1-3 条标准化教训。要求：

- 每条不超过 50 字
- 以"下次..."或"下次遇到..."开头
- 必须包含具体工具名或操作步骤
- 具有动作导向性（能直接指导下一步行动）
- 与已有教训去重（不要输出重复的教训）

已有教训:
{self._format_existing_lessons()}

反思:
{reflection}

请只输出教训，每条一行。例如：
下次搜索无结果时，立即用更精确的关键词重新 web_search，不要重复同一查询。"""

        try:
            client = OpenAI(
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
                timeout=60.0,
            )

            model = get_model_for_role("lesson_extract")
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "你是一个知识蒸馏专家。请只输出简洁的教训文本，每条一行。"},
                    {"role": "user", "content": lesson_prompt},
                ],
                "max_tokens": 256,
                "temperature": 0.1,
                "stream": False,
            }
            _eb = thinking_extra_body(model, enable_thinking=False)  # 评审 1.2
            if _eb:
                kwargs["extra_body"] = _eb

            response = call_with_retry(lambda: client.chat.completions.create(**kwargs))  # 评审 1.9
            content = response.choices[0].message.content or ""

            # 解析教训行
            lessons = []
            for line in content.strip().split("\n"):
                line = line.strip()
                # 去掉编号前缀
                line = line.lstrip("0123456789. -、)")
                if len(line) >= 8 and len(line) <= 80:
                    lessons.append(line)

            # 去重 × 去重
            unique = []
            for lesson in lessons[:3]:  # 最多 3 条
                if not self._is_duplicate_lesson(lesson):
                    unique.append(lesson)
                    # 更新 counter
                    key = self._lesson_key(lesson)
                    self.lesson_counter[key] = self.lesson_counter.get(key, 0) + 1

            logger.info("教训提取完成 lessons=%d unique=%d", len(lessons), len(unique))
            return unique

        except Exception as exc:
            logger.error("教训提取失败: %s", exc)
            # 降级：从反思中提取最后一句话
            sentences = reflection.replace("。", "。\n").split("\n")
            last = [s.strip() for s in sentences if len(s.strip()) > 10]
            return last[:1] if last else []

    def _format_existing_lessons(self) -> str:
        """格式化已有教训列表"""
        if not self.lesson_counter:
            return "（无）"
        lines = []
        for lesson, count in sorted(self.lesson_counter.items(),
                                     key=lambda x: x[1], reverse=True)[:10]:
            lines.append(f"- [{count}次] {lesson}")
        return "\n".join(lines)

    def _is_duplicate_lesson(self, lesson: str) -> bool:
        """检查教训是否与已有教训重复（简单关键词重叠检测）"""
        key = self._lesson_key(lesson)
        if key in self.lesson_counter:
            return True
        # Jaccard 检测
        for existing in self.lesson_counter:
            if _jaccard_similarity(lesson, existing) > 0.7:
                return True
        return False

    @staticmethod
    def _lesson_key(lesson: str) -> str:
        """生成教训的关键词键（去标点、统一化）"""
        import re
        key = re.sub(r"[^\w一-鿿]", "", lesson.lower())
        return key[:30]

    def _extract_scratchpad_facts(
        self, task: str, answer: str, trajectory: str,
    ) -> list[str]:
        """从本轮 Trial 轨迹中提取已确认的关键事实数据

        不是教训/策略，是具体的数据点（数值、名称、日期等），
        下一轮 Trial 可以直接使用而无需重新搜索。
        """
        from openai import OpenAI

        # 只在有实质性答案或较长轨迹时提取（避免对空轨迹浪费 API）
        if len(trajectory) < 500 and len(answer) < 50:
            return []

        traj_tail = trajectory[-3000:] if len(trajectory) > 3000 else trajectory
        answer_snippet = answer[:500]

        prompt = f"""从以下 Agent 执行轨迹中提取关键信息，供下一轮 Trial 使用。

提取两类内容（合计最多 5 条）：
1. 已确认的正向事实 — 从工具返回中确认的具体数值、名称、日期等，每条 ≤80 字
2. 已证明无效的工具路径 — 以"[无效]"开头，如"[无效] read_xlsx 只能读取单元格值，无法获取背景颜色"
   仅在工具明确返回错误或被证明无法满足需求时才记录，避免误判

格式：每条一行，如果完全没有可提取的内容，输出"无"

任务: {task[:200]}

Agent 最终答案: {answer_snippet}

轨迹末尾:
{traj_tail}

请只输出列表，每条一行："""

        try:
            client = OpenAI(
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
                timeout=60.0,
            )
            model = get_model_for_role("lesson_extract")
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "你是一个数据提取专家。只输出从工具返回中确认的事实数据，每条一行。"},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 300,
                "temperature": 0.0,
                "stream": False,
            }
            _eb = thinking_extra_body(model, enable_thinking=False)  # 评审 1.2
            if _eb:
                kwargs["extra_body"] = _eb

            response = call_with_retry(lambda: client.chat.completions.create(**kwargs))  # 评审 1.9
            content = response.choices[0].message.content or ""

            if "无" in content and len(content) < 20:
                return []

            facts = []
            for line in content.strip().split("\n"):
                line = line.strip().lstrip("0123456789.-、) ")
                if len(line) >= 10 and len(line) <= 120:
                    facts.append(line)

            logger.info("Scratchpad 事实提取完成: %d 条", len(facts))
            return facts[:5]

        except Exception as exc:
            logger.error("Scratchpad 事实提取失败: %s", exc)
            return []

    @staticmethod
    def _extract_visited_urls(trajectory: str, visited_urls: set) -> None:
        """从轨迹中提取已访问的 URL，用于跨 Trial 去重"""
        import re as _re
        fetch_tools = {"web_fetch", "tavily_extract", "read_pdf"}
        # 匹配 tool calling 格式和传统 Action 格式中的 URL
        urls = _re.findall(r"https?://[^\s\)\]\},\"'<>]+", trajectory)
        for url in urls:
            base = url.split("?")[0].rstrip("/")
            if base.endswith((".pdf", ".doc", ".docx", ".xls", ".xlsx")):
                visited_urls.add(base)
            elif any(tool in trajectory[:trajectory.find(url) + 1] for tool in fetch_tools if url in trajectory):
                visited_urls.add(base)

    @staticmethod
    def _sandwich_truncate(text: str, head_chars: int = 1000, tail_chars: int = 1940) -> str:
        """三明治截断：保留首尾，中间省略

        失败通常发生在轨迹末尾，三明治策略确保反思模型看到:
        - 任务理解和初始规划（开头）
        - 失败现场和最后操作（结尾）
        """
        if len(text) <= head_chars + tail_chars + 50:
            return text

        head = text[:head_chars]
        tail = text[-tail_chars:]
        return f"{head}\n\n...（中间省略 {len(text) - head_chars - tail_chars} 字符）...\n\n{tail}"

    @staticmethod
    def _load_reflection_prompt() -> str:
        """加载反思 System Prompt"""
        if os.path.isfile(_REFLECTION_PROMPT_PATH):
            with open(_REFLECTION_PROMPT_PATH, "r", encoding="utf-8") as f:
                return f.read()

        logger.warning("反思 Prompt 文件不存在: %s，使用内置 Prompt", _REFLECTION_PROMPT_PATH)
        return "你是一个任务诊断分析师。分析失败原因并给出具体的纠正策略。"

    # ── 便捷方法 ──

    def reset_memory(self) -> None:
        """重置记忆池"""
        self.memory.clear()

    def get_memory_state(self) -> dict:
        """获取记忆系统状态"""
        entries = self.memory.get_all_entries()
        return {
            "size": self.memory.size(),
            "max_size": self.memory.max_size,
            "entries": [
                {
                    "task": e.task[:80],
                    "trial": e.trial_number,
                    "reflection": e.reflection[:100],
                    "timestamp": e.timestamp,
                }
                for e in entries
            ],
        }


# ==========================================================================
# CLI 入口
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ReflexionReActAgent — 带自我反思能力的 ReAct 智能体",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m nexa_agent.reflexion_agent "2024年全球GDP排名前十的国家有哪些？"
  python -m nexa_agent.reflexion_agent "北京到上海的直线距离是多少？" --max-trials 2
  python -m nexa_agent.reflexion_agent "这张图里有什么？" --image data/photo.jpg
  python -m nexa_agent.reflexion_agent "最新的苹果公司市值" --eval-mode heuristic
        """,
    )
    parser.add_argument(
        "task",
        type=str,
        nargs="?",
        default=None,
        help="用户任务描述（直接输入文本，或用 --file 从文件读取）",
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="从 TXT 文件读取问题（# 开头的行为注释）",
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="图片路径（相对或绝对路径）",
    )
    parser.add_argument(
        "--max-trials", "-t",
        type=int,
        default=REFLEXION_CONFIG["max_trials"],
        help=f"最大重试轮数（默认: {REFLEXION_CONFIG['max_trials']}）",
    )
    parser.add_argument(
        "--max-steps", "-s",
        type=int,
        default=REACT_CONFIG["max_steps"],
        help=f"每次 ReAct 最大步数（默认: {REACT_CONFIG['max_steps']}）",
    )
    parser.add_argument(
        "--eval-mode", "-e",
        type=str,
        default=REFLEXION_CONFIG["evaluator_mode"],
        choices=["heuristic", "llm", "hybrid"],
        help=f"评估模式（默认: {REFLEXION_CONFIG['evaluator_mode']}）",
    )
    parser.add_argument(
        "--memory-size", "-m",
        type=int,
        default=REFLEXION_CONFIG["max_memory_size"],
        help=f"记忆池容量（默认: {REFLEXION_CONFIG['max_memory_size']}）",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="OfferCheck 阶段标识（stage1=选岗调研 | stage4=offer证伪，加载对应任务定义 prompt）",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="持久化记忆到文件",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式",
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="打印配置摘要后退出",
    )

    args = parser.parse_args()

    # 打印配置
    if args.config:
        print(get_config_summary())
        return

    # 确定任务文本: --file 优先，其次 positional arg
    task_text = None
    if args.file:
        fpath = Path(args.file)
        if not fpath.exists():
            alt = Path(_project_root) / args.file
            fpath = alt if alt.exists() else fpath
        if not fpath.exists():
            print(f"❌ 文件不存在: {args.file}")
            sys.exit(1)
        with open(fpath, "r", encoding="utf-8") as f:
            lines = [l.rstrip() for l in f if not l.lstrip().startswith("#")]
        task_text = "\n".join(lines).strip()
        if not task_text:
            print(f"❌ 文件中没有有效内容: {fpath}")
            sys.exit(1)
        print(f"📄 从 {fpath} 加载问题 ({len(task_text)} 字符)")
    elif args.task:
        task_text = args.task
    else:
        print("❌ 请提供任务文本或使用 --file 指定问题文件")
        parser.print_help()
        sys.exit(1)

    # 验证 API Key
    if not LLM_API_KEY:
        print("❌ 错误: 未配置 DEEPSEEK_API_KEY 或 KIMI_API_KEY")
        print("   请在 .env 文件中设置 API Key")
        sys.exit(1)

    # 验证图片路径
    image_path = None
    if args.image:
        if os.path.isfile(args.image):
            image_path = args.image
        else:
            alt_path = os.path.join(_project_root, args.image)
            if os.path.isfile(alt_path):
                image_path = alt_path
            else:
                print(f"❌ 错误: 图片文件不存在: {args.image}")
                sys.exit(1)

    # 创建 Agent
    agent = ReflexionReActAgent(
        max_trials=args.max_trials,
        max_memory_size=args.memory_size,
        evaluator_mode=args.eval_mode,
        persist_memory=args.persist,
        max_steps=args.max_steps,
    )

    if not args.quiet:
        print(get_config_summary())

    # 执行
    result = agent.execute(
        task=task_text,
        image_path=image_path,
        verbose=not args.quiet,
        stage=args.stage,
    )

    # 输出结果
    if not args.quiet:
        print(result.summary())
    else:
        print(result.answer)

    # 返回退出码
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
