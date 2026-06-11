"""
文件上传接口 — V0

支持票据图片（JPG/PNG/PDF）上传，保存后返回路径供对话接口引用。

日志统一使用 logger_config.get_logger。
"""

from __future__ import annotations

import os
import uuid
import time
import hashlib
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api.deps import (
    UploadResponse, ImageAnalysisRequest, ImageAnalysisResponse,
    get_config, get_extraction_pipeline,
)
from app.storage.database import get_session, init_db
from app.storage.models import ImageAnalysisCache
from app.pipeline.vlm import IMAGE_ANALYSIS_PROMPT
from app.utils.logger_config import get_logger

logger = get_logger("api_upload")

router = APIRouter(prefix="/api/v0", tags=["upload"])

# 允许的图片类型
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".pdf"}
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_to_response(row: ImageAnalysisCache, cached: bool) -> ImageAnalysisResponse:
    return ImageAnalysisResponse(
        session_id=row.session_id,
        file_id=row.file_id,
        file_sha256=row.file_sha256,
        file_path=row.image_path,
        filename=row.filename,
        content_type=row.content_type,
        status=row.status,
        cached=cached,
        model_name=row.model_name or "",
        vlm_text=row.vlm_text or "",
        structured_data=row.structured_data,
        latency_ms=row.latency_ms or 0,
        error_message=row.error_message,
    )


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="上传票据图片",
    description="上传发票/票据图片，返回服务器端路径。支持 JPG/PNG/PDF。",
)
async def upload_file(
    file: UploadFile = File(..., description="图片文件"),
    session_id: str = Form("", description="关联的会话ID（可选）"),
) -> UploadResponse:
    """上传文件到服务器本地存储"""
    config = get_config()

    # 校验扩展名
    ext = Path(file.filename or "unknown").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning("不支持的文件类型: %s", ext)
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext}。允许: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # 生成唯一文件名
    sid = session_id or uuid.uuid4().hex[:8]
    file_id = f"{sid}_{uuid.uuid4().hex[:8]}"
    safe_name = f"{file_id}{ext}"
    upload_dir = os.path.join(config.project_root, "data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    dest_path = os.path.join(upload_dir, safe_name)

    # 写入
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"文件大小超过限制 ({MAX_FILE_SIZE // 1024 // 1024}MB)")
    file_sha256 = _sha256_bytes(content)

    with open(dest_path, "wb") as f:
        f.write(content)

    logger.info("文件上传成功 session=%s file=%s size=%d path=%s",
                sid, file.filename, len(content), dest_path)

    return UploadResponse(
        session_id=sid,
        filename=file.filename or "unknown",
        file_path=dest_path,
        file_size=len(content),
        content_type=file.content_type,
        file_id=file_id,
        file_sha256=file_sha256,
    )


@router.post(
    "/files/analyze",
    response_model=ImageAnalysisResponse,
    summary="一次性识别上传图片",
    description="对上传后的图片执行一次 VLM 识别，并缓存识别结果供后续会话复用。",
)
async def analyze_uploaded_file(request: ImageAnalysisRequest) -> ImageAnalysisResponse:
    """上传后预识别图片内容；后续 chat 可复用 vlm_text，避免二次 VLM。"""
    init_db()
    path = request.file_path
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="文件不存在，无法识别")

    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return ImageAnalysisResponse(
            session_id=request.session_id,
            file_id=request.file_id or Path(path).stem,
            file_sha256=request.file_sha256 or _sha256_file(path),
            file_path=path,
            filename=request.filename,
            content_type=request.content_type,
            status="failed",
            error_message="当前 VLM 预识别仅支持图片文件，PDF 将在后续流程中按文件处理。",
        )

    file_sha256 = request.file_sha256 or _sha256_file(path)
    file_id = request.file_id or Path(path).stem

    s = get_session()
    try:
        cached = s.query(ImageAnalysisCache).filter_by(file_sha256=file_sha256).first()
        if cached and cached.status == "success":
            cached.session_id = request.session_id
            cached.image_path = path
            cached.filename = request.filename or cached.filename
            cached.content_type = request.content_type or cached.content_type
            s.add(cached)
            s.commit()
            logger.info("图片识别命中缓存 file_sha256=%s", file_sha256[:12])
            return _cache_to_response(cached, cached=True)

        pipeline = get_extraction_pipeline()
        t0 = time.time()
        if not pipeline.vlm_available:
            row = cached or ImageAnalysisCache(file_id=file_id, file_sha256=file_sha256)
            row.session_id = request.session_id
            row.image_path = path
            row.filename = request.filename
            row.content_type = request.content_type
            row.model_name = ""
            row.vlm_text = ""
            row.status = "failed"
            row.latency_ms = 0
            row.error_message = "VLM 不可用"
            s.add(row)
            s.commit()
            return _cache_to_response(row, cached=False)

        result = pipeline.extract(image_path=path, text_input=IMAGE_ANALYSIS_PROMPT)
        elapsed = int((time.time() - t0) * 1000)
        row = cached or ImageAnalysisCache(file_id=file_id, file_sha256=file_sha256)
        row.session_id = request.session_id
        row.image_path = path
        row.filename = request.filename
        row.content_type = request.content_type
        engine = getattr(pipeline, "_vlm_engine", None)
        row.model_name = getattr(engine, "model_name", None) or getattr(engine, "_model", "") or "vlm"
        row.vlm_text = result.raw_text or ""
        row.status = "success"
        row.latency_ms = elapsed
        row.error_message = None
        row.structured_data = result.structured_data or {}
        s.add(row)
        s.commit()
        sd_len = len(result.structured_data.get("raw_output", "")) if result.structured_data else 0
        logger.info("图片识别完成 session=%s file=%s latency=%dms vlm_text_len=%d structured_raw_len=%d",
                    request.session_id, request.filename, elapsed, len(row.vlm_text), sd_len)
        return _cache_to_response(row, cached=False)
    except Exception as exc:
        s.rollback()
        logger.error("图片识别失败: %s", exc)
        row = s.query(ImageAnalysisCache).filter_by(file_sha256=file_sha256).first()
        row = row or ImageAnalysisCache(file_id=file_id, file_sha256=file_sha256)
        row.session_id = request.session_id
        row.image_path = path
        row.filename = request.filename
        row.content_type = request.content_type
        row.model_name = "vlm"
        row.vlm_text = ""
        row.status = "failed"
        row.latency_ms = 0
        row.error_message = str(exc)
        try:
            s.add(row)
            s.commit()
        except Exception:
            s.rollback()
        return _cache_to_response(row, cached=False)
    finally:
        s.close()
