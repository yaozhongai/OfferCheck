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
from server.api.deps import get_config
from server.security import (
    enforce_run_quota, validate_image_path, clamp_run_limits, RUN_CONCURRENCY,
)
from nexa_agent.logger import get_logger

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
    logger.info("run_stage 请求 stage=%s input_len=%d", request.stage, len(request.input))

    try:
        agent = _build_agent(request)
        t0 = time.time()
        result = agent.execute(
            task=request.input,
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
    )


@router.post("/run_stage/stream", summary="执行 OfferCheck 阶段（SSE 实时 trace 流）")
async def run_stage_stream(request: RunStageRequest, http_request: Request) -> StreamingResponse:
    """SSE 流式执行：引擎在后台线程跑，on_event 事件经线程安全队列实时推给前端。

    事件类型（data 行 JSON 的 type 字段）：
      trial_start / step_start / action / observation / correction /
      verifier_start / verifier_result / trial_evaluated / final_answer /
      done（终止，携带最终裁定）/ error（异常）
    """
    enforce_run_quota(http_request)
    image_path = validate_image_path(request.image_path, get_config().project_root)
    if not RUN_CONCURRENCY.try_acquire():
        raise HTTPException(status_code=429, detail="并发调查数已达上限，请稍后再试")
    logger.info("run_stage/stream 请求 stage=%s input_len=%d",
                request.stage, len(request.input))

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
                routed, reason = route_stage_for_followup(request.input, request.stage)
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
                task=request.input,
                image_path=image_path,
                verbose=False,
                stage=effective_stage,
                on_event=lambda e: event_q.put(e),
                answer_mode=bool(request.answer_mode),
                output_lang=request.output_lang,
            )
            event_q.put({
                "type": "done",
                "success": result.success,
                "answer": result.answer,
                "trials_used": result.trials_used,
                "reflections": result.reflections,
                "latency_ms": round((time.time() - t0) * 1000),
                "stage": effective_stage,
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
        # 起始事件，让前端立刻有反馈
        yield f"data: {json.dumps({'type': 'started', 'stage': request.stage}, ensure_ascii=False)}\n\n"
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
            try:
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
            except (TypeError, ValueError) as exc:
                logger.warning("事件序列化失败 type=%s: %s", evt.get("type"), exc)

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
