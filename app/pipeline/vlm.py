"""
VLM 视觉语言模型管线接口 — 路线 B

提供 VLM 端到端识别的抽象接口，不包含具体实现（非 mock）。
下游调用方依赖此接口编程，实际 VLM 模型在后续接入（MiniCPM-V / Qwen-VL）。

日志统一使用 logger_config.get_logger。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.utils.logger_config import get_logger

logger = get_logger("vlm_pipeline")


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class VLMResult:
    """VLM 识别结果"""
    response: str = ""                         # 模型自然语言回答
    structured_data: Dict[str, Any] = field(default_factory=dict)  # 结构化抽取结果
    confidence: float = 0.0
    model_name: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_ms: float = 0.0
    raw_output: str = ""                       # 模型原始输出（调试用）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response": self.response,
            "structured_data": self.structured_data,
            "confidence": self.confidence,
            "model_name": self.model_name,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "elapsed_ms": self.elapsed_ms,
        }


# ---------------------------------------------------------------------------
# VLM 接口
# ---------------------------------------------------------------------------

class BaseVLMEngine(ABC):
    """VLM 引擎抽象基类

    子类需实现 analyze_image() 方法。
    当前阶段仅定义接口，不做 mock 实现。
    """

    @abstractmethod
    def analyze_image(
        self,
        image_path: str,
        prompt: str = "",
        **kwargs,
    ) -> VLMResult:
        """对图片进行端到端理解

        Args:
            image_path: 图片文件路径
            prompt: 引导提示词。若为空则使用默认 prompt
            **kwargs: 引擎特定参数

        Returns:
            VLMResult
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """检查引擎是否可用"""
        ...


# ---------------------------------------------------------------------------
# VLM Prompt 模板
# ---------------------------------------------------------------------------

INVOICE_EXTRACTION_PROMPT = """你是一个专业的票据识别助手。请分析这张票据图片，提取以下信息并以 JSON 格式返回：

{
    "invoice_code": "发票号码",
    "invoice_date": "开票日期 (YYYY-MM-DD)",
    "amount": "金额（数字，不含单位）",
    "tax_number": "税号",
    "seller_name": "销售方名称",
    "buyer_name": "购买方名称",
    "items": [
        {"name": "商品名称", "quantity": "数量", "unit_price": "单价", "amount": "金额"}
    ],
    "notes": "备注或补充信息"
}

注意：
- 若某项信息无法识别，填写空字符串或 null
- 确保金额为纯数字
- 日期统一为 YYYY-MM-DD 格式
- 只输出 JSON，不要输出任何解释性文字"""

GENERAL_IMAGE_PROMPT = """请详细描述这张图片的内容，以 JSON 格式返回：

{
    "image_type": "截图/照片/文档/图表/其他",
    "summary": "一句话概括图片内容",
    "text_content": "图片中所有可见文字的完整转录",
    "objects": ["物体1", "物体2"],
    "layout": "图片的整体布局和结构描述",
    "notes": "补充信息"
}

注意：
- 只输出 JSON，不要输出任何解释性文字
- text_content 尽可能完整地转录所有可见文字
- 若某项无法识别，填写空字符串或 null"""

IMAGE_ANALYSIS_PROMPT = """你是图片分析助手。先判断图片类型，再以对应 JSON 格式输出。

## 判断规则
- 票据类：发票、收据、小票、电子发票截图 → 使用票据格式
- 其他：截图、照片、文档、图表等 → 使用通用格式

## 票据格式
{
    "image_type": "invoice",
    "invoice_code": "发票号码",
    "invoice_date": "开票日期 (YYYY-MM-DD)",
    "amount": "金额（纯数字）",
    "tax_number": "税号",
    "seller_name": "销售方名称",
    "buyer_name": "购买方名称",
    "items": [
        {"name": "商品名称", "quantity": "数量", "unit_price": "单价", "amount": "金额"}
    ],
    "notes": "备注"
}

## 通用格式
{
    "image_type": "general",
    "summary": "一句话概括",
    "text_content": "所有可见文字完整转录",
    "objects": ["物体列表"],
    "layout": "布局描述",
    "notes": "补充信息"
}

注意：
- 先判断 image_type 是 "invoice" 还是 "general"
- 若某项无法识别，填写空字符串或 null
- 确保金额为纯数字，日期为 YYYY-MM-DD
- 只输出 JSON，不要输出任何解释性文字"""
