"""
analyze_image — 端侧 VLM 图片分析工具

复用现有 LlamaCppVLMEngine，不重复实现 base64/API 调用。
"""

from __future__ import annotations

import os

from app.tools import register
from app.api.deps import get_extraction_pipeline
from app.utils.logger_config import get_logger

logger = get_logger("tools.image")


def _parse_image_args(param: str) -> tuple[str, str]:
    """解析 | 分隔参数: "path/to/img.jpg | 提示词" """
    if "|" in param:
        parts = param.split("|", 1)
        return parts[0].strip(), parts[1].strip()
    return param.strip(), "请描述这张图片的内容"


def _resolve_image_path(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("./"):
        raw = raw[2:]
    if os.path.isabs(raw):
        if os.path.isfile(raw):
            return raw
        raise FileNotFoundError(f"图片不存在: {raw}")

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates = [
        os.path.join(project_root, raw),
        os.path.join(project_root, "data", raw),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)
    raise FileNotFoundError(f"图片不存在，尝试: {candidates}")


@register(
    name="analyze_image",
    description="使用端侧本地 MiniCPM-V 视觉模型分析图片，速度快，适合简单描述或轻量感知。参数格式: 图片路径 | 分析提示词",
    signature="analyze_image(image_path | prompt)",
    examples=["analyze_image(./data/photo.jpg | 描述图中的主要内容)"],
)
def analyze_image(param: str) -> str:
    param = param.strip()
    if not param:
        return "[错误] analyze_image: 参数为空，格式为 '图片路径 | 提示词'"

    image_path, prompt = _parse_image_args(param)
    logger.info("analyze_image path=%s", image_path)

    try:
        resolved = _resolve_image_path(image_path)
    except FileNotFoundError as e:
        return f"[错误] analyze_image: {e}"

    try:
        pipeline = get_extraction_pipeline()
        if not pipeline.vlm_available:
            return "[错误] analyze_image: VLM 引擎不可用"
        result = pipeline._vlm_engine.analyze_image(resolved, prompt=prompt)
        return result.response or "(VLM 返回空内容)"
    except Exception as exc:
        return f"[错误] analyze_image 执行失败: {exc}"
