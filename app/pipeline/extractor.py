"""
信息提取管线 — V0

整合 VLM 识别结果，进行信息提取和后处理。
当前提供提取管线框架和接口，具体 VLM 引擎通过依赖注入接入。

日志统一使用 logger_config.get_logger。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.utils.logger_config import get_logger

logger = get_logger("extractor")


# ---------------------------------------------------------------------------
# 统一提取结果
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """统一信息提取结果"""
    source: str = ""                           # "vlm" | "text"
    raw_text: str = ""
    structured_data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    elapsed_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "raw_text": self.raw_text,
            "structured_data": self.structured_data,
            "confidence": self.confidence,
            "elapsed_ms": self.elapsed_ms,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# 提取管线
# ---------------------------------------------------------------------------

class ExtractionPipeline:
    """信息提取管线

    组合文本 / VLM 两条路线，根据输入特点自动选择。
    当前仅搭建管线框架，具体 VLM 引擎通过依赖注入接入。

    用法::

        pipeline = ExtractionPipeline()
        result = pipeline.extract(image_path="/path/to/invoice.jpg")
    """

    def __init__(self):
        self._vlm_engine: Optional[Any] = None
        self._prefer_mode: str = "vlm"         # "vlm" | "auto"
        logger.info("ExtractionPipeline 初始化完成")

    # ------------------------------------------------------------------
    # 引擎注册
    # ------------------------------------------------------------------

    def set_vlm_engine(self, engine) -> None:
        """注入 VLM 引擎实例"""
        self._vlm_engine = engine
        logger.info("VLM 引擎已注册: %s", type(engine).__name__)

    def set_prefer_mode(self, mode: str) -> None:
        """设置优先路线"""
        if mode not in ("vlm", "auto"):
            raise ValueError(f"无效模式: {mode}，可选值: vlm/auto")
        self._prefer_mode = mode
        logger.info("优先路线已设置: %s", mode)

    @property
    def vlm_available(self) -> bool:
        return self._vlm_engine is not None and self._vlm_engine.is_available()

    # ------------------------------------------------------------------
    # 提取
    # ------------------------------------------------------------------

    def extract(
        self,
        image_path: str = "",
        text_input: str = "",
        mode: str = "",
    ) -> ExtractionResult:
        """执行信息提取

        Args:
            image_path: 图片路径
            text_input: 纯文本输入（不需要 VLM 时）
            mode: 强制使用某条路线，覆盖全局设置

        Returns:
            ExtractionResult
        """
        mode = mode or self._prefer_mode

        # 纯文本 → 跳过 VLM
        if not image_path and text_input:
            logger.debug("纯文本输入，跳过识别")
            return ExtractionResult(
                source="text",
                raw_text=text_input,
                confidence=1.0,
            )

        # 按路线执行
        if mode == "vlm" or (mode == "auto" and self.vlm_available):
            return self._extract_vlm(image_path, prompt=text_input)
        else:
            logger.warning("无可用提取路线 image_path=%s", image_path)
            return ExtractionResult(
                source="none",
                confidence=0.0,
                metadata={"error": "无可用引擎"},
            )

    def _extract_vlm(self, image_path: str, prompt: str = "") -> ExtractionResult:
        """VLM 端到端提取"""
        if not self.vlm_available:
            raise RuntimeError("VLM 引擎未注册或不可用")

        import time
        t0 = time.time()
        vlm_result = self._vlm_engine.analyze_image(image_path, prompt=prompt)
        elapsed = (time.time() - t0) * 1000

        return ExtractionResult(
            source="vlm",
            raw_text=vlm_result.response,
            structured_data=vlm_result.structured_data,
            confidence=vlm_result.confidence,
            elapsed_ms=elapsed,
            metadata={
                "model": vlm_result.model_name,
                "prompt_tokens": vlm_result.prompt_tokens,
                "completion_tokens": vlm_result.completion_tokens,
            },
        )
