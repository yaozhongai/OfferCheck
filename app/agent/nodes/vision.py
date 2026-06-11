"""vision_direct / vision_schema / vision_perceive — VLM 节点"""

from __future__ import annotations

import time
import json
from typing import Optional
from app.agent.state import (
    AgentState, StepStatus, Observation, ObservationSource, ModelCallRecord, trace_patch,
)
from app.api.deps import get_extraction_pipeline, get_llm_client
from app.llm.client import LLMMessage
from app.utils.logger_config import get_logger

logger = get_logger("node.vision")


def _get_image_path(state: AgentState) -> Optional[str]:
    refs = state.get("image_refs", [])
    return refs[0].path if refs else None


def _get_cached_image_analysis(state: AgentState) -> tuple[str, dict]:
    """从 ImageAnalysisCache 表查询已缓存的 VLM 识别结果（按文件 SHA256）"""
    image_path = _get_image_path(state)
    if not image_path:
        return "", {}
    try:
        from app.storage.database import get_session
        from app.storage.models import ImageAnalysisCache
        import hashlib
        with open(image_path, "rb") as f:
            file_sha256 = hashlib.sha256(f.read()).hexdigest()
        s = get_session()
        try:
            row = s.query(ImageAnalysisCache).filter_by(
                file_sha256=file_sha256, status="success",
            ).first()
            if row and row.vlm_text:
                logger.info("VLM 缓存命中 sha256=%s path=%s", file_sha256[:16], image_path)
                return row.vlm_text.strip(), row.structured_data
        finally:
            s.close()
    except Exception as exc:
        logger.debug("缓存查询失败: %s", exc)
    return "", {}


def _answer_from_cached_analysis(question: str, image_text: str) -> tuple[str, ModelCallRecord | None]:
    if not question.strip():
        return image_text, None

    llm = get_llm_client()
    t0 = time.time()
    try:
        resp = llm.chat([
            LLMMessage.system(
                "你是票据/文档问答助手。只能基于已识别的图片内容回答；"
                "如果识别内容中没有相关信息，请明确说明未在图片识别内容中找到。"
            ),
            LLMMessage.user(
                f"【已识别图片内容】\n{image_text}\n\n【用户问题】\n{question}"
            ),
        ])
        elapsed = int((time.time() - t0) * 1000)
        return resp.content, ModelCallRecord(
            provider=resp.model,
            model_name=resp.model,
            node="vision_direct",
            input_summary=question[:100],
            output_summary=resp.content[:200],
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            total_tokens=resp.total_tokens,
            latency_ms=elapsed,
            success=True,
        )
    except Exception as exc:
        elapsed = int((time.time() - t0) * 1000)
        logger.warning("缓存图片内容问答失败，回退返回识别文本: %s", exc)
        return image_text, ModelCallRecord(
            model_name="unknown",
            node="vision_direct",
            input_summary=question[:100],
            output_summary=image_text[:200],
            latency_ms=elapsed,
            success=False,
            error_message=str(exc),
        )


def vision_direct(state: AgentState) -> dict:
    """VISION_DIRECT — VLM 直接回答"""
    step = state.get("step_count", 0) + 1
    user_input = state.get("user_input", "")
    cached_text, _ = _get_cached_image_analysis(state)
    if cached_text:
        answer, call = _answer_from_cached_analysis(user_input, cached_text)
        patch = {
            "final_answer": answer,
            "observations": [Observation(
                source=ObservationSource.VLM,
                content=cached_text,
                node="vision_direct",
                source_id="cached_image_analysis",
            )],
            **trace_patch(step=step, node="vision_direct", action="use_cached_image_analysis",
                          status=StepStatus.SUCCESS, reason="used cached image analysis",
                          input_summary=user_input[:100],
                          output_summary=answer[:200],
                          latency_ms=call.latency_ms if call else 0),
            "step_count": step,
        }
        if call:
            patch["model_calls"] = [call]
        return patch

    pipeline = get_extraction_pipeline()
    image_path = _get_image_path(state)

    t0 = time.time()
    if pipeline.vlm_available and image_path:
        vlm = pipeline._vlm_engine
        direct_prompt = f"请用自然语言直接回答以下问题，不要输出 JSON。\n问题：{user_input or '请描述图片内容'}"
        result = vlm.analyze_image(image_path, prompt=direct_prompt)
        answer = result.response
    else:
        answer = "[VLM 不可用]"

    elapsed = int((time.time() - t0) * 1000)

    return {
        "final_answer": answer,
        "observations": [Observation(
            source=ObservationSource.VLM, content=answer, node="vision_direct",
        )],
        "model_calls": [ModelCallRecord(
            model_name="vlm", node="vision_direct", latency_ms=elapsed,
            output_summary=answer[:200], success=True,
        )],
        **trace_patch(step=step, node="vision_direct", action="vlm_direct_answer",
                      status=StepStatus.SUCCESS, output_summary=answer[:200],
                      latency_ms=elapsed),
        "step_count": step,
    }


def vision_schema(state: AgentState) -> dict:
    """VISION_SCHEMA — VLM 结构化提取"""
    step = state.get("step_count", 0) + 1
    cached_text, structured = _get_cached_image_analysis(state)
    if cached_text:
        answer = json.dumps(structured, ensure_ascii=False, indent=2) if structured else cached_text
        return {
            "final_answer": answer,
            "structured_output": structured,
            "observations": [Observation(
                source=ObservationSource.VLM,
                content=cached_text,
                structured_data=structured,
                node="vision_schema",
                source_id="cached_image_analysis",
            )],
            **trace_patch(step=step, node="vision_schema", action="use_cached_image_analysis",
                          status=StepStatus.SUCCESS, reason="used cached image analysis",
                          output_summary=answer[:200], latency_ms=0),
            "step_count": step,
        }

    pipeline = get_extraction_pipeline()
    image_path = _get_image_path(state)

    t0 = time.time()
    if pipeline.vlm_available and image_path:
        result = pipeline.extract(image_path=image_path, text_input=state.get("user_input", ""))
        answer = result.raw_text
        structured = result.structured_data
    else:
        answer = "[VLM 不可用]"
        structured = {}

    elapsed = int((time.time() - t0) * 1000)

    return {
        "final_answer": answer,
        "structured_output": structured,
        "observations": [Observation(
            source=ObservationSource.VLM, content=answer,
            structured_data=structured, node="vision_schema",
        )],
        "model_calls": [ModelCallRecord(
            model_name="vlm", node="vision_schema", latency_ms=elapsed,
            output_summary=answer[:200], success=True,
        )],
        **trace_patch(step=step, node="vision_schema", action="vlm_schema_extract",
                      status=StepStatus.SUCCESS, output_summary=answer[:200],
                      latency_ms=elapsed),
        "step_count": step,
    }


def vision_perceive(state: AgentState) -> dict:
    """VISION_REASON 第一阶段 — VLM 感知"""
    step = state.get("step_count", 0) + 1
    cached_text, structured = _get_cached_image_analysis(state)
    if cached_text:
        return {
            "observations": [Observation(
                source=ObservationSource.VLM,
                content=cached_text,
                structured_data=structured,
                node="vision_perceive",
                source_id="cached_image_analysis",
            )],
            **trace_patch(step=step, node="vision_perceive", action="use_cached_image_analysis",
                          status=StepStatus.SUCCESS, reason="used cached image analysis",
                          output_summary=cached_text[:200], latency_ms=0),
            "step_count": step,
        }

    pipeline = get_extraction_pipeline()
    image_path = _get_image_path(state)

    t0 = time.time()
    if pipeline.vlm_available and image_path:
        result = pipeline.extract(image_path=image_path, text_input="")
        content = result.raw_text
    else:
        content = ""

    elapsed = int((time.time() - t0) * 1000)

    return {
        "observations": [Observation(
            source=ObservationSource.VLM, content=content, node="vision_perceive",
        )],
        "model_calls": [ModelCallRecord(
            model_name="vlm", node="vision_perceive", latency_ms=elapsed,
            output_summary=content[:200], success=True,
        )],
        **trace_patch(step=step, node="vision_perceive", action="vlm_perceive",
                      status=StepStatus.SUCCESS, output_summary=content[:200],
                      latency_ms=elapsed),
        "step_count": step,
    }
