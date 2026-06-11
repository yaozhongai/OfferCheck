"""
VLM 客户端 — V0

通过 llama.cpp server 的 OpenAI 兼容 API 调用视觉语言模型（MiniCPM-V 等），
支持多模态图片输入。

实现 BaseVLMEngine 接口，可直接注入 ExtractionPipeline。

日志统一使用 app.utils.logger_config。
"""

from __future__ import annotations

import base64
import time
from typing import Any, Dict

from app.pipeline.vlm import BaseVLMEngine, VLMResult, INVOICE_EXTRACTION_PROMPT
from app.utils.logger_config import get_logger

logger = get_logger("vlm_engine")


class LlamaCppVLMEngine(BaseVLMEngine):
    """VLM 引擎 — 通过 llama.cpp OpenAI 兼容 API 调用

    用法::

        engine = LlamaCppVLMEngine(model="minicpm-v", base_url="http://127.0.0.1:8080/v1")
        engine.is_available()  # → True/False
        result = engine.analyze_image("invoice.jpg", prompt="提取发票信息")

    前置条件:
        llama.cpp server 已启动，监听 127.0.0.1:8080
    """

    def __init__(
        self,
        model: str = "minicpm-v",
        base_url: str = "http://127.0.0.1:8080/v1",
        api_key: str = "not-needed",
        timeout: float = 120.0,
        ctx_size: int = 4096,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._ctx_size = ctx_size
        self._client = None
        logger.info("llama.cpp VLM 引擎初始化 model=%s base_url=%s ctx=%d", model, base_url, ctx_size)

    # ------------------------------------------------------------------
    # 核心方法
    # ------------------------------------------------------------------

    def analyze_image(
        self,
        image_path: str,
        prompt: str = "",
        **kwargs,
    ) -> VLMResult:
        t0 = time.time()
        prompt = prompt or INVOICE_EXTRACTION_PROMPT

        # 读取图片并转为 base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # 推断 MIME 类型
        ext = image_path.rsplit(".", 1)[-1].lower() if "." in image_path else "png"
        mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}
        mime_type = f"image/{mime_map.get(ext, 'png')}"

        # 构建多模态消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                    },
                ],
            }
        ]

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self._ctx_size),
                temperature=kwargs.get("temperature", 0.1),
                stream=False,
                extra_body={"n_ctx": self._ctx_size},
            )

            elapsed = (time.time() - t0) * 1000
            choice = response.choices[0]

            # 尝试从 JSON 响应中提取结构化数据
            structured_data = self._parse_json_response(
                choice.message.content or ""
            )

            result = VLMResult(
                response=choice.message.content or "",
                structured_data=structured_data,
                confidence=0.85,
                model_name=response.model or self._model,
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                elapsed_ms=elapsed,
            )

            logger.info(
                "llama.cpp VLM 完成 model=%s elapsed=%.0fms tokens(in=%d out=%d)",
                self._model, elapsed, result.prompt_tokens, result.completion_tokens,
            )
            # 打印结构化数据
            import json as _json
            try:
                sd = result.structured_data
                sd_str = _json.dumps(sd, ensure_ascii=False, indent=4)
                logger.info("VLM structured_data (%d keys, %d chars):\n%s",
                            len(sd) if sd else 0, len(sd_str), sd_str[:2000])
            except Exception:
                logger.debug("structured_data 序列化失败")
            return result

        except Exception as exc:
            logger.error("llama.cpp VLM 调用失败: %s", exc, exc_info=True)
            raise

    def is_available(self) -> bool:
        """检查 llama.cpp 服务是否可用"""
        try:
            import urllib.request
            req = urllib.request.Request(f"{self._base_url}/models")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._client

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """从 VLM 的文本响应中提取 JSON 结构化数据"""
        import json
        text = text.strip()
        # 尝试提取 markdown 代码块中的 JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            logger.debug("VLM 输出非 JSON 格式，返回原始文本")
            return {"raw_output": text}
