"""
文件上传接口 — V0

支持票据图片（JPG/PNG/PDF）上传，保存后返回路径供对话接口引用。

日志统一使用 nexa_agent.logger.get_logger。
"""

from __future__ import annotations

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
    """上传后预识别图片内容。

    TODO(7/4 server 变薄): 旧 app/pipeline（未实现的 VLM 抽象）已删除。
    图片识别改由 nexa_agent 核心工具 analyze_image / analyze_image_cloud 承担；
    本路由待改为直接调用核心工具，并沿用 ImageAnalysisCache 做去重缓存。
    在核心 wiring 完成前，此端点返回 501。
    """
    raise HTTPException(
        status_code=501,
        detail=(
            "图片预识别待接入 nexa_agent.tools.analyze_image（旧 app/pipeline 已移除，"
            "见 7/4 server 变薄计划）"
        ),
    )
