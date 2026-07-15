"""
OfferCheck 阶段执行接口 — server 瘦调用 nexa_agent 核心引擎

Walking Skeleton（Slice 0）：把 HTTP 请求瘦转发给核心
ReflexionReActAgent.execute()，按 stage 加载阶段任务定义 prompt，返回裁定结果。

当前为非流式（阻塞执行后一次性返回），证明 server↔引擎↔前端全栈打通；
后续 Slice（7/3 trace 发射钩子）升级为 SSE 实时推送每步调查轨迹。

端点定义为同步 def，由 FastAPI 自动放入线程池执行，避免阻塞事件循环。
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from server.api.schemas import RunStageRequest, RunStageResponse
from server.api.prompt_assembly import resolve_task_input
from server.trace_store.recorder import TraceRecorder, new_trace_id
from server.api.deps import get_config
from server.security import (
    enforce_run_quota, validate_image_path, clamp_run_limits, RUN_CONCURRENCY,
)
from nexa_agent.logger import get_logger
from nexa_agent.run_metrics import build_run_metrics

logger = get_logger("api_run_stage")
router = APIRouter(prefix="/api/v0", tags=["run_stage"])


def _build_agent(request: RunStageRequest):
    """按请求参数构造核心引擎实例（run_stage 与 stream 共用）。"""
    from nexa_agent.reflexion_agent import ReflexionReActAgent
    from nexa_agent.config import REFLEXION_CONFIG

    # 服务端强制上限（忽略客户端传入的超大值，防放大 GMI 花费）
    max_steps, max_trials = clamp_run_limits(request.max_steps, request.max_trials)
    agent = ReflexionReActAgent(
        max_trials=max_trials,
        max_memory_size=REFLEXION_CONFIG["max_memory_size"],
        evaluator_mode=REFLEXION_CONFIG["evaluator_mode"],
        persist_memory=False,
        max_steps=max_steps,
    )
    return agent


@router.post("/run_stage", response_model=RunStageResponse,
             summary="执行 OfferCheck 阶段（瘦调用核心引擎，非流式）")
def run_stage(request: RunStageRequest, http_request: Request) -> RunStageResponse:
    """瘦转发到 nexa_agent 核心 ReflexionReActAgent.execute()（阻塞返回）。"""
    enforce_run_quota(http_request)
    image_path = validate_image_path(request.image_path, get_config().project_root)
    if not RUN_CONCURRENCY.try_acquire():
        raise HTTPException(status_code=429, detail="并发调查数已达上限，请稍后再试")
    # 结构化会话字段 → 引擎任务串（评审 3.4：拼串逻辑从前端收回 server）
    task_input = resolve_task_input(request)
    logger.info("run_stage 请求 stage=%s input_len=%d", request.stage, len(task_input))

    try:
        agent = _build_agent(request)
        t0 = time.time()
        result = agent.execute(
            task=task_input,
            image_path=image_path,
            verbose=False,
            stage=request.stage,
            output_lang=request.output_lang,
        )
        latency_ms = (time.time() - t0) * 1000
    finally:
        RUN_CONCURRENCY.release()

    logger.info("run_stage 完成 stage=%s success=%s trials=%d latency=%.0fms",
                request.stage, result.success, result.trials_used, latency_ms)

    return RunStageResponse(
        success=result.success,
        stage=request.stage,
        answer=result.answer,
        trials_used=result.trials_used,
        trial_details=result.trial_details,
        reflections=result.reflections,
        latency_ms=latency_ms,
        verdict=result.verdict,
        usage={
            "prompt_tokens": result.total_prompt_tokens,
            "completion_tokens": result.total_completion_tokens,
            "total_tokens": result.total_tokens,
        },
        metrics=build_run_metrics(result, latency_ms),
    )


@router.post("/run_stage/stream", summary="执行 OfferCheck 阶段（SSE 实时 trace 流）")
async def run_stage_stream(request: RunStageRequest, http_request: Request) -> StreamingResponse:
    """SSE 流式执行：引擎在后台线程跑，on_event 事件经线程安全队列实时推给前端。

    事件类型（data 行 JSON 的 type 字段）：
      trial_start / step_start / action / observation / correction /
      verifier_start / verifier_result / trial_evaluated / usage（每 Trial token 用量）/
      final_answer / done（终止，携带最终裁定 + 累计 usage）/ error（异常）
    """
    enforce_run_quota(http_request)
    image_path = validate_image_path(request.image_path, get_config().project_root)
    if not RUN_CONCURRENCY.try_acquire():
        raise HTTPException(status_code=429, detail="并发调查数已达上限，请稍后再试")
    # 结构化会话字段 → 引擎任务串（评审 3.4）；stage_router 与 execute 共用同一串，
    # [追问/补充信息] 标记契约不破。
    task_input = resolve_task_input(request)
    # 同源轨迹持久化（评审 3.3）：一次 run 一个 trace_id + recorder（缓冲，结束落盘）
    trace_id = new_trace_id()
    recorder = TraceRecorder(trace_id, stage=request.stage)
    logger.info("run_stage/stream 请求 stage=%s input_len=%d trace_id=%s",
                request.stage, len(task_input), trace_id)

    event_q: "queue.Queue[dict]" = queue.Queue()
    _SENTINEL = {"type": "__end__"}

    def _worker():
        agent = _build_agent(request)
        t0 = time.time()
        try:
            # followup 轻量 stage 路由：追问明显属于其他阶段能力时切换 stage prompt
            # （关键词门零成本快筛 + fast 层 LLM 确认；失败安全回落当前 stage）
            effective_stage = request.stage
            if request.auto_route and request.stage:
                from nexa_agent.stage_router import route_stage_for_followup
                routed, reason = route_stage_for_followup(task_input, request.stage)
                if routed:
                    logger.info("stage 路由 %s → %s（%s）", request.stage, routed, reason)
                    event_q.put({
                        "type": "stage_routed",
                        "from_stage": request.stage,
                        "to_stage": routed,
                        "reason": reason,
                    })
                    effective_stage = routed

            result = agent.execute(
                task=task_input,
                image_path=image_path,
                verbose=False,
                stage=effective_stage,
                on_event=lambda e: event_q.put(e),
                answer_mode=bool(request.answer_mode),
                output_lang=request.output_lang,
            )
            latency_ms = round((time.time() - t0) * 1000)
            event_q.put({
                "type": "done",
                "success": result.success,
                "answer": result.answer,
                "trials_used": result.trials_used,
                "reflections": result.reflections,
                "latency_ms": latency_ms,
                "stage": effective_stage,
                "verdict": result.verdict,  # 结构化裁定（评审 3.2），additive
                "usage": {                   # Token 用量（评审 3.6），additive
                    "prompt_tokens": result.total_prompt_tokens,
                    "completion_tokens": result.total_completion_tokens,
                    "total_tokens": result.total_tokens,
                },
                "metrics": build_run_metrics(result, latency_ms),
            })
        except Exception as exc:  # noqa: BLE001
            logger.error("run_stage/stream 引擎异常: %s", exc, exc_info=True)
            event_q.put({"type": "error", "message": str(exc)[:500]})
        finally:
            RUN_CONCURRENCY.release()
            event_q.put(_SENTINEL)

    async def _event_stream():
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        loop = asyncio.get_running_loop()
        # 起始事件，让前端立刻有反馈；trace_id additive 透出供关联持久化轨迹
        started_evt = {"type": "started", "stage": request.stage, "trace_id": trace_id}
        recorder.record(started_evt)
        yield f"data: {json.dumps(started_evt, ensure_ascii=False)}\n\n"
        final_success = None
        final_latency = None
        while True:
            try:
                # 带超时的 get：每 20s 若无事件则发一次 SSE keepalive 注释，
                # 防止 Next.js dev proxy / Nginx 在长时间无数据后断开连接
                evt = await loop.run_in_executor(None, lambda: event_q.get(timeout=20))
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if evt.get("type") == "__end__":
                break
            # 同源持久化（评审 3.3）：落库的就是发给浏览器的同一条事件（best-effort，
            # 缓冲不落盘、绝不打断流）。终止事件顺带抓 success/latency 供 finalize 摘要。
            recorder.record(evt)
            if evt.get("type") in ("done", "error"):
                final_success = evt.get("success")
                final_latency = evt.get("latency_ms")
            try:
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
            except (TypeError, ValueError) as exc:
                logger.warning("事件序列化失败 type=%s: %s", evt.get("type"), exc)
        # 流结束：整条 trace 单次落盘（OTel JSON）
        recorder.finalize(success=final_success, latency_ms=final_latency)

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
