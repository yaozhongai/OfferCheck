"""
文件上传接口 — V0

支持票据图片（JPG/PNG/PDF）上传，保存后返回路径供对话接口引用。

日志统一使用 nexa_agent.logger.get_logger。
"""

from __future__ import annotations

import contextlib
import os
import uuid
import time
import hashlib
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from server.api.deps import (
    UploadResponse, ImageAnalysisRequest, ImageAnalysisResponse,
    get_config,
)
from server.security import enforce_upload_quota, UPLOAD_MAX_BYTES
from server.persistence.database import get_session, init_db
from server.persistence.models import ImageAnalysisCache
from nexa_agent.logger import get_logger

logger = get_logger("api_upload")

router = APIRouter(prefix="/api/v0", tags=["upload"])

# 允许的图片类型
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".pdf"}
MAX_FILE_SIZE = UPLOAD_MAX_BYTES  # 默认 10MB（server.security，可经 ABUSE_UPLOAD_MAX_MB 调整）


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
    http_request: Request,
    file: UploadFile = File(..., description="图片文件"),
    session_id: str = Form("", description="关联的会话ID（可选）"),
) -> UploadResponse:
    """上传文件到服务器本地存储"""
    enforce_upload_quota(http_request)
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

    # 有界读取：最多读 限额+1 字节，避免超大文件把整块塞进内存（内存 DoS）
    content = await file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"文件大小超过限制 ({MAX_FILE_SIZE // 1024 // 1024}MB)")
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
    """上传后预识别图片内容（云端 VLM，SHA-256 去重缓存）。

    直接调用 nexa_agent 核心工具 analyze_image_cloud（Moonshot 官方 Kimi K2.6，
    Kimi 回落）；同 file_sha256 的重复请求直接命中 ImageAnalysisCache，
    不再调 VLM。识别结果供后续会话/调查复用。
    """
    t0 = time.time()

    # 1) 解析文件：优先 file_sha256 / file_id 查缓存表拿路径，否则用 file_path
    file_path = request.file_path
    file_sha256 = request.file_sha256 or ""
    file_id = request.file_id or ""

    with contextlib.closing(get_session()) as db:
        if not file_sha256 and file_id:
            row = db.query(ImageAnalysisCache).filter_by(file_id=file_id).first()
            if row:
                file_sha256 = row.file_sha256
        if not file_sha256:
            if not os.path.isfile(file_path):
                raise HTTPException(status_code=404, detail=f"文件不存在: {file_path}")
            file_sha256 = _sha256_file(file_path)

        # 2) 缓存命中 → 直接返回
        cached_row = db.query(ImageAnalysisCache).filter_by(
            file_sha256=file_sha256).first()
        if cached_row and cached_row.status == "success" and cached_row.vlm_text:
            logger.info("files/analyze 缓存命中 sha256=%s", file_sha256[:12])
            return _cache_to_response(cached_row, cached=True)

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"文件不存在: {file_path}")

    # 3) 调引擎云端 VLM 工具（同步阻塞，放线程池避免卡 event loop）
    import asyncio
    from nexa_agent.tools import analyze_image_cloud

    prompt = "提取图中所有文字信息，并简要描述图片主要内容。"
    vlm_text = await asyncio.to_thread(
        analyze_image_cloud, f"{file_path} | {prompt}")
    latency_ms = int((time.time() - t0) * 1000)

    is_error = vlm_text.startswith("[错误]")
    status = "error" if is_error else "success"
    model_name = "analyze_image_cloud"
    if is_error:
        logger.warning("files/analyze VLM 识别失败: %s", vlm_text[:200])

    # 4) 写缓存（含失败结果——避免对同一坏文件反复打 VLM）
    with contextlib.closing(get_session()) as db:
        row = db.query(ImageAnalysisCache).filter_by(
            file_sha256=file_sha256).first()
        if row is None:
            row = ImageAnalysisCache(
                file_id=file_id or f"analyze_{uuid.uuid4().hex[:12]}",
                file_sha256=file_sha256,
                session_id=request.session_id,
                image_path=file_path,
            )
            db.add(row)
        row.filename = request.filename or row.filename
        row.content_type = request.content_type or row.content_type
        row.model_name = model_name
        row.vlm_text = "" if is_error else vlm_text
        row.status = status
        row.latency_ms = latency_ms
        row.error_message = vlm_text if is_error else None
        db.commit()
        db.refresh(row)
        return _cache_to_response(row, cached=False)
