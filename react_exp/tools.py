"""
ReAct Agent 工具集

提供 web_search (Tavily + DuckDuckGo), wikipedia_search (REST API),
analyze_image (端侧 MiniCPM-V), analyze_image_cloud (云端 Kimi K2.6),
tavily_extract + save_content (网页提取+落盘), calculator, get_current_time
八个工具的统一注册与执行。

每个工具函数签名统一为:
    def tool_name(param: str) -> str

返回值均为 Observation 字符串（成功或错误消息）。
"""

from __future__ import annotations

import base64
import os
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import requests

# 加载 .env（兼容独立调用场景）
try:
    from dotenv import load_dotenv

    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _dotenv_path = os.path.join(_project_root, ".env")
    if os.path.exists(_dotenv_path):
        load_dotenv(_dotenv_path)
except ImportError:
    pass

from react_exp.logger_config import get_logger

logger = get_logger("react_tools")

# ==========================================================================
# 工具注册表
# ==========================================================================

# 全局工具注册表: tool_name -> callable
TOOLS: Dict[str, Callable[[str], str]] = {}

# 工具元数据: tool_name -> {description, signature, examples}
TOOL_META: Dict[str, Dict[str, Any]] = {}


def register(name: str, description: str, signature: str, examples: List[str]):
    """装饰器：将函数注册到 TOOLS 全局表中"""

    def decorator(func: Callable[[str], str]):
        TOOLS[name] = func
        TOOL_META[name] = {
            "description": description,
            "signature": signature,
            "examples": examples,
        }
        return func

    return decorator


# ==========================================================================
# 工具实现
# ==========================================================================


@register(
    name="web_search",
    description="使用 Tavily Search API 搜索互联网，返回前 5 条实时结果（含内容摘要）；失效时自动回退 DuckDuckGo",
    signature="web_search(query)",
    examples=["web_search(2025年诺贝尔奖得主)"],
)
def web_search(query: str) -> str:
    """Web 搜索 — Tavily 优先，DuckDuckGo 兜底

    Args:
        query: 搜索关键词

    Returns:
        前 5 条结果的标题、链接、摘要
    """
    query = query.strip()
    if not query:
        return "[错误] web_search: 查询内容不能为空"

    logger.info("web_search 调用 query=%s", query)

    # 优先 Tavily
    tavily_api_key = os.environ.get("TAVILY_API_KEY", "")
    if tavily_api_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_api_key)
            response = client.search(query, max_results=5)

            results = response.get("results", [])
            if results:
                lines = [f"Tavily 搜索 '{query}' 结果（共 {len(results)} 条）:\n"]
                for i, r in enumerate(results, 1):
                    title = r.get("title", "无标题")
                    url = r.get("url", "")
                    content = r.get("content", "")
                    if len(content) > 300:
                        content = content[:300] + "..."
                    lines.append(f"{i}. {title}")
                    lines.append(f"   链接: {url}")
                    lines.append(f"   摘要: {content}")
                    lines.append("")
                logger.info("web_search Tavily 完成 results=%d", len(results))
                return "\n".join(lines)
            else:
                logger.info("Tavily 返回空结果，回退 DuckDuckGo")
        except Exception as exc:
            logger.warning("Tavily 搜索失败: %s，回退 DuckDuckGo", exc)

    # 兜底：DuckDuckGo
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS  # type: ignore[import]

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(r)

        if not results:
            return f"未找到与 '{query}' 相关的结果。"

        lines = [f"DuckDuckGo 搜索 '{query}' 结果（共 {len(results)} 条）:\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "无标题")
            href = r.get("href", "")
            body = r.get("body", "")
            body = body[:300] + "..." if len(body) > 300 else body
            lines.append(f"{i}. {title}")
            lines.append(f"   链接: {href}")
            lines.append(f"   摘要: {body}")
            lines.append("")

        return "\n".join(lines)

    except ImportError:
        logger.error("web_search: duckduckgo_search 未安装，且 Tavily 不可用")
        return "[错误] web_search: 所有搜索引擎均不可用"
    except Exception as exc:
        logger.error("web_search 失败: %s", exc, exc_info=True)
        return f"[错误] web_search 执行失败: {exc}"


@register(
    name="wikipedia_search",
    description="搜索 Wikipedia，返回最相关文章的摘要（最多 800 字符）",
    signature="wikipedia_search(query)",
    examples=["wikipedia_search(transformer deep learning)"],
)
def wikipedia_search(query: str) -> str:
    """Wikipedia 百科搜索

    直接调用 Wikipedia REST API，根据 query 自动选择中文/英文端点。
    只返回条目的导言纯文本，避免内容过长。

    Args:
        query: 搜索关键词（支持中英文）

    Returns:
        条目标题 + 纯文本摘要（~800 字符）
    """
    query = query.strip()
    if not query:
        return "[错误] wikipedia_search: 查询内容不能为空"

    logger.info("wikipedia_search 调用 query=%s", query)

    # 自动检测语言
    has_chinese = bool(re.search(r"[一-鿿]", query))
    lang = "zh" if has_chinese else "en"
    api_url = f"https://{lang}.wikipedia.org/w/api.php"

    headers = {"User-Agent": "NexaAgent-ReAct/1.0 (nexa@example.com)"}

    try:
        # 第一步：搜索最相关的条目
        search_res = requests.get(
            api_url,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 1,
            },
            headers=headers,
            timeout=10,
        )
        search_res.raise_for_status()
        search_hits = search_res.json().get("query", {}).get("search", [])

        if not search_hits:
            # 中文没找到，回退英文
            if lang == "zh":
                logger.info("中文 Wikipedia 无结果，回退英文")
                api_url = "https://en.wikipedia.org/w/api.php"
                search_res = requests.get(
                    api_url,
                    params={
                        "action": "query", "list": "search",
                        "srsearch": query, "format": "json", "srlimit": 1,
                    },
                    headers=headers, timeout=10,
                )
                search_res.raise_for_status()
                search_hits = search_res.json().get("query", {}).get("search", [])
            if not search_hits:
                return f"Wikipedia 未找到与 '{query}' 相关的条目。"

        target_title = search_hits[0]["title"]

        # 第二步：获取条目导言（纯文本）
        content_res = requests.get(
            api_url,
            params={
                "action": "query",
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "titles": target_title,
                "format": "json",
            },
            headers=headers,
            timeout=10,
        )
        content_res.raise_for_status()
        pages = content_res.json().get("query", {}).get("pages", {})

        for page_id, page_info in pages.items():
            if page_id == "-1":
                return f"Wikipedia 条目 '{target_title}' 内容无法读取。"
            extract = (page_info.get("extract") or "").strip()
            if extract:
                # 截断到 800 字符
                if len(extract) > 800:
                    extract = extract[:800] + "..."
                return f"Wikipedia - {target_title}\n链接: https://{lang}.wikipedia.org/wiki/{requests.utils.quote(target_title)}\n\n{extract}"

        return f"Wikipedia 找到条目 '{target_title}'，但无正文摘要。"

    except requests.exceptions.Timeout:
        logger.error("Wikipedia API 超时")
        return "[错误] wikipedia_search: Wikipedia API 请求超时，请稍后重试"
    except requests.exceptions.ConnectionError:
        logger.error("Wikipedia API 连接失败（网络可能受限）")
        return "[错误] wikipedia_search: 无法连接 Wikipedia，网络可能受限，请尝试 web_search"
    except Exception as exc:
        logger.error("wikipedia_search 失败: %s", exc, exc_info=True)
        return f"[错误] wikipedia_search 执行失败: {exc}"


# --------------------------------------------------------------------------
# 图片工具辅助函数
# --------------------------------------------------------------------------

def _resolve_image_path(raw_path: str) -> str:
    """解析图片路径为绝对路径

    支持三种形式:
      - 绝对路径: /path/to/image.jpg
      - 相对项目根: ./data/image.jpg or data/image.jpg
      - 相对当前目录: image.jpg

    返回解析后的绝对路径，若文件不存在则抛出 FileNotFoundError。
    """
    raw_path = raw_path.strip()

    # 去掉开头的 ./
    if raw_path.startswith("./"):
        raw_path = raw_path[2:]

    # 绝对路径直接返回
    if os.path.isabs(raw_path):
        if os.path.isfile(raw_path):
            return raw_path
        raise FileNotFoundError(f"图片文件不存在: {raw_path}")

    # 尝试相对项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(project_root, raw_path),          # 相对项目根
        os.path.join(project_root, "data", raw_path),  # data/ 子目录
        os.path.join(os.getcwd(), raw_path),            # 相对当前工作目录
    ]

    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)

    raise FileNotFoundError(
        f"图片文件不存在，尝试过以下路径:\n"
        + "\n".join(f"  - {c}" for c in candidates)
    )


def _parse_image_tool_args(param: str) -> tuple:
    """解析图片工具的 | 分隔参数

    Args:
        param: 原始参数字符串，如 "path/to/img.jpg | 描述这张图"

    Returns:
        (image_path: str, prompt: str)
    """
    if "|" in param:
        parts = param.split("|", 1)
        image_path = parts[0].strip()
        prompt = parts[1].strip()
    else:
        image_path = param.strip()
        prompt = "请描述这张图片的内容"

    return image_path, prompt


# --------------------------------------------------------------------------
# 端侧图片分析
# --------------------------------------------------------------------------

@register(
    name="analyze_image",
    description="使用端侧本地 MiniCPM 视觉模型分析图片，速度快，适合简单描述或轻量感知",
    signature="analyze_image(image_path | prompt)",
    examples=["analyze_image(./data/photo.jpg | 描述图中的主要内容)"],
)
def analyze_image(param: str) -> str:
    """端侧图片分析 — llama.cpp MiniCPM-V

    通过 llama.cpp server 的 OpenAI 兼容 API 调用本地 VLM。

    Args:
        param: "图片路径 | 提示词" 格式的字符串

    Returns:
        VLM 的描述文本
    """
    param = param.strip()
    if not param:
        return "[错误] analyze_image: 参数不能为空，格式为 '图片路径 | 提示词'"

    image_path, prompt = _parse_image_tool_args(param)
    logger.info("analyze_image 调用 path=%s prompt=%s", image_path, prompt[:100])

    try:
        resolved_path = _resolve_image_path(image_path)
    except FileNotFoundError as e:
        logger.error("analyze_image: %s", e)
        return f"[错误] analyze_image: {e}"

    # 读取 .env 中的 VLM 配置
    vlm_base_url = os.environ.get("VLM_BASE_URL", "http://127.0.0.1:8080/v1")
    vlm_model = os.environ.get("VLM_MODEL_NAME", "minicpm-v")
    vlm_ctx_size = int(os.environ.get("VLM_CTX_SIZE", "4096"))

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key="not-needed",
            base_url=vlm_base_url,
            timeout=120.0,
        )

        # 读取图片并 base64 编码
        with open(resolved_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # 推断 MIME 类型
        ext = resolved_path.rsplit(".", 1)[-1].lower() if "." in resolved_path else "png"
        mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}
        mime_type = f"image/{mime_map.get(ext, 'png')}"

        t0 = time.time()
        response = client.chat.completions.create(
            model=vlm_model,
            messages=[
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
            ],
            max_tokens=vlm_ctx_size,
            temperature=0.1,
            stream=False,
            extra_body={"n_ctx": vlm_ctx_size},
        )

        elapsed_ms = (time.time() - t0) * 1000
        content = response.choices[0].message.content or "(VLM 返回空内容)"
        tokens = response.usage.total_tokens if response.usage else 0

        logger.info(
            "analyze_image 完成 elapsed=%.0fms tokens=%d model=%s",
            elapsed_ms, tokens, vlm_model,
        )

        return content

    except ImportError:
        return "[错误] analyze_image: openai 库未安装"
    except Exception as exc:
        logger.error("analyze_image VLM 调用失败: %s", exc, exc_info=True)
        return f"[错误] analyze_image 执行失败: {exc}"


# --------------------------------------------------------------------------
# 云端图片分析 — Gemini 3.5 Flash
# --------------------------------------------------------------------------

@register(
    name="analyze_image_cloud",
    description="使用云端 Kimi K2.6 视觉大模型分析图片，适合 OCR、复杂场景理解",
    signature="analyze_image_cloud(image_path | prompt)",
    examples=["analyze_image_cloud(./data/chart.png | 提取图中所有文字并解释含义)"],
)
def analyze_image_cloud(param: str) -> str:
    """云端图片分析 — Kimi K2.6

    使用 OpenAI SDK 调用 Kimi K2.6 的多模态能力进行图片理解。
    需要配置环境变量 MOONSHOT_API_KEY 和 KIMI_BASE_URL。

    Args:
        param: "图片路径 | 提示词" 格式的字符串

    Returns:
        Kimi 的分析文本
    """
    param = param.strip()
    if not param:
        return "[错误] analyze_image_cloud: 参数不能为空，格式为 '图片路径 | 提示词'"

    image_path, prompt = _parse_image_tool_args(param)
    logger.info("analyze_image_cloud 调用 path=%s prompt=%s", image_path, prompt[:100])

    try:
        resolved_path = _resolve_image_path(image_path)
    except FileNotFoundError as e:
        logger.error("analyze_image_cloud: %s", e)
        return f"[错误] analyze_image_cloud: {e}"

    # 读取 Kimi API 配置
    kimi_api_key = os.environ.get("MOONSHOT_API_KEY", "")
    kimi_base_url = os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
    if not kimi_api_key:
        return "[错误] analyze_image_cloud: 未配置 MOONSHOT_API_KEY，请在 .env 中设置"

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=kimi_api_key,
            base_url=kimi_base_url,
            timeout=120.0,
        )

        # 读取图片并 base64 编码
        with open(resolved_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # 推断 MIME 类型
        ext = resolved_path.rsplit(".", 1)[-1].lower() if "." in resolved_path else "png"
        mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "gif": "gif"}
        mime_type = mime_map.get(ext, "png")
        image_url = f"data:image/{mime_type};base64,{image_data}"

        t0 = time.time()
        response = client.chat.completions.create(
            model="kimi-k2.6",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                },
            ],
            max_tokens=4096,
            stream=False,
        )

        elapsed_ms = (time.time() - t0) * 1000
        content = response.choices[0].message.content or "(Kimi 返回空内容)"
        tokens = response.usage.total_tokens if response.usage else 0

        logger.info(
            "analyze_image_cloud 完成 elapsed=%.0fms model=kimi-k2.6 tokens=%d",
            elapsed_ms, tokens,
        )

        return content

    except ImportError:
        logger.error("analyze_image_cloud: openai SDK 未安装")
        return "[错误] analyze_image_cloud: openai SDK 未安装"
    except Exception as exc:
        logger.error("analyze_image_cloud Kimi 调用失败: %s", exc, exc_info=True)
        return f"[错误] analyze_image_cloud 执行失败: {exc}"


# --------------------------------------------------------------------------
# 网页内容提取（缓存到内存，save_content 负责落盘）
# --------------------------------------------------------------------------

# 会话级累积缓存：每次 tavily_extract 的结果追加到此列表
_session_extracts: List[dict] = []


def get_session_extracts() -> List[dict]:
    """返回本次会话所有 extract 结果（供策展步骤使用）"""
    return _session_extracts


def clear_session_extracts() -> None:
    """清空会话缓存（每次 Agent 运行前调用）"""
    global _session_extracts
    _session_extracts = []


@register(
    name="tavily_extract",
    description="使用 Tavily Extract 提取指定网页的正文内容并缓存到内存；如需持久化保存可调用 save_content 写入本地 Markdown",
    signature="tavily_extract(url)",
    examples=["tavily_extract(https://en.wikipedia.org/wiki/Artificial_intelligence)"],
)
def tavily_extract(url: str) -> str:
    """Tavily 网页内容提取（仅缓存，不自动落盘）

    提取成功后将全文缓存在内存中。LLM 可根据内容价值决定是否调用
    save_content 写入磁盘。Wikipedia 等免费可查内容无需保存，
    行业报告、研究文档、专有数据等才需要。

    Args:
        url: 要提取内容的网页 URL

    Returns:
        标题 + 预览（前 1500 字符）+ 总字数 + 保存提示
    """
    url = url.strip()
    if not url:
        return "[错误] tavily_extract: URL 不能为空"

    logger.info("tavily_extract 调用 url=%s", url)

    tavily_api_key = os.environ.get("TAVILY_API_KEY", "")
    if not tavily_api_key:
        return "[错误] tavily_extract: 未配置 TAVILY_API_KEY，请在 .env 中设置"

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=tavily_api_key)
        response = client.extract(url)

        results = response.get("results", [])
        if not results:
            return f"Tavily Extract 未能从 '{url}' 提取到内容。"

        raw_content = results[0].get("raw_content", "")
        title = results[0].get("title", "")
        if not raw_content:
            return f"Tavily Extract: '{url}' 页面无正文内容。"

        # 追加到会话级缓存，不落盘
        _session_extracts.append({
            "url": url, "title": title, "raw_content": raw_content,
        })

        preview = raw_content[:1500]
        if len(raw_content) > 1500:
            preview += "..."

        logger.info("tavily_extract 完成 url=%s len=%d cached=true", url, len(raw_content))
        return (
            f"Tavily Extract - {title or url}\n"
            f"总字符数: {len(raw_content)}\n\n"
            f"--- 内容预览（前 1500 字符）---\n{preview}\n\n"
            f"💡 全文已缓存。如果内容有长期参考价值（如行业报告、研究文档），可调用 save_content(文件名) 保存到本地。"
        )

    except ImportError:
        logger.error("tavily_extract: tavily-python 未安装")
        return "[错误] tavily_extract: tavily-python 未安装，请执行 pip install tavily-python"
    except Exception as exc:
        logger.error("tavily_extract 失败: %s", exc, exc_info=True)
        return f"[错误] tavily_extract 执行失败: {exc}"


# --------------------------------------------------------------------------
# 网页正文提取 — Jina Reader + trafilatura 双引擎
# --------------------------------------------------------------------------

@register(
    name="web_fetch",
    description="提取指定网页的正文内容。优先使用 Jina Reader（免费），失败时自动回退 trafilatura 本地解析。适合阅读文章、文档等深层内容，不依赖 Tavily API Key。",
    signature="web_fetch(url)",
    examples=["web_fetch(https://en.wikipedia.org/wiki/Artificial_intelligence)"],
)
def web_fetch(url: str) -> str:
    """网页正文提取 — Jina Reader 优先，trafilatura 兜底

    不依赖 Tavily API Key，可免费使用。
    提取正文后自动追加到会话缓存，可配合 save_content 持久化。

    Args:
        url: 要提取内容的网页 URL

    Returns:
        标题 + 正文预览（前 2000 字符）+ 总字符数
    """
    url = url.strip()
    if not url:
        return "[错误] web_fetch: URL 不能为空"

    # 确保 URL 有 scheme
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    logger.info("web_fetch 调用 url=%s", url)

    content = ""
    engine = ""

    # ── 引擎 1: Jina Reader ──
    try:
        jina_url = f"https://r.jina.ai/{url}"
        headers = {
            "Accept": "text/markdown",
            "User-Agent": "NexaAgent-ReAct/1.0",
        }
        resp = requests.get(jina_url, headers=headers, timeout=30)
        if resp.status_code == 200 and resp.text.strip():
            content = resp.text.strip()
            engine = "Jina Reader"
            logger.info("web_fetch Jina 成功 len=%d", len(content))
        else:
            logger.info("web_fetch Jina 返回空/失败 status=%d，回退 trafilatura", resp.status_code)
    except Exception as exc:
        logger.info("web_fetch Jina 异常: %s，回退 trafilatura", exc)

    # ── 引擎 2: trafilatura ──
    if not content:
        try:
            import trafilatura

            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                extracted = trafilatura.extract(
                    downloaded,
                    include_links=False,
                    include_images=False,
                    include_tables=True,
                    output_format="markdown",
                )
                if extracted:
                    content = extracted.strip()
                    engine = "trafilatura"
                    logger.info("web_fetch trafilatura 成功 len=%d", len(content))
        except ImportError:
            logger.warning("web_fetch: trafilatura 未安装")
        except Exception as exc:
            logger.error("web_fetch trafilatura 失败: %s", exc)

    if not content:
        return (
            f"[错误] web_fetch: 无法从 '{url}' 提取内容。\n"
            f"Jina Reader 和 trafilatura 均未返回有效正文。"
        )

    # 限制长度
    max_len = 8000
    if len(content) > max_len:
        content_preview = content[:max_len]
    else:
        content_preview = content

    # 追加到会话缓存
    _session_extracts.append({
        "url": url, "title": url, "raw_content": content,
    })

    return (
        f"web_fetch ({engine}) — {url}\n"
        f"总字符数: {len(content)}\n\n"
        f"--- 正文预览（前 {min(len(content), max_len)} 字符）---\n"
        f"{content_preview}"
        f"{'...(截断)' if len(content) > max_len else ''}\n\n"
        f"💡 全文已缓存。如需保存可调用 save_content(文件名)。"
    )


# --------------------------------------------------------------------------
# 内容保存（从缓存写入 .md）
# --------------------------------------------------------------------------

def write_extract_to_disk(extract: dict, filename: str) -> str:
    """将一条 extract 写入 data/extracts/{filename}.md

    供 react_agent 策展步骤直接调用，也供 save_content 工具复用。

    Args:
        extract: {"url", "title", "raw_content"}
        filename: 文件名（不含路径和 .md）

    Returns:
        保存的文件路径
    """
    safe_name = re.sub(r"[^\w\-一-鿿]", "_", filename).strip("_")[:80]
    if not safe_name:
        raise ValueError(f"文件名无效: {filename}")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extract_dir = os.path.join(project_root, "data", "extracts")
    os.makedirs(extract_dir, exist_ok=True)

    filepath = os.path.join(extract_dir, f"{safe_name}.md")

    extracted_at = datetime.now().isoformat()
    md_content = (
        f"---\n"
        f"title: {extract.get('title') or 'Extracted Content'}\n"
        f"source: {extract.get('url', '')}\n"
        f"extracted_at: {extracted_at}\n"
        f"content_length: {len(extract.get('raw_content', ''))}\n"
        f"---\n\n"
        f"{extract.get('raw_content', '')}"
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("write_extract_to_disk file=%s len=%d", filepath, len(extract.get("raw_content", "")))
    return filepath


@register(
    name="save_content",
    description="将最近一次 tavily_extract 缓存的网页全文写入本地 Markdown 文件（含 YAML frontmatter）。注意：Wikipedia 等免费可查内容无需保存，仅行业报告、研究文档、公司数据等有长期参考价值的内容才需要持久化。",
    signature="save_content(filename)",
    examples=["save_content(apple_ghg_emissions_2025)"],
)
def save_content(filename: str) -> str:
    """将最近一条 extract 写入 .md 文件

    供 ReAct 循环中的 LLM 直接调用。策展步骤（_curation_step）中则使用
    write_extract_to_disk() 逐条写入。

    Args:
        filename: 文件名（不含路径和 .md），如 "apple_ghg_2025"

    Returns:
        保存路径及元信息
    """
    filename = filename.strip()
    if not filename:
        return "[错误] save_content: 文件名不能为空"

    if not _session_extracts:
        return (
            "[错误] save_content: 没有可保存的内容。"
            "请先调用 tavily_extract(url) 提取网页内容后再保存。"
        )

    # 取最近一条 extract
    extract = _session_extracts[-1]

    try:
        filepath = write_extract_to_disk(extract, filename)
        title = extract.get("title", "")
        raw_content = extract.get("raw_content", "")
        return (
            f"✅ 已保存到: {filepath}\n"
            f"标题: {title}\n"
            f"字符数: {len(raw_content)}"
        )

    except Exception as exc:
        logger.error("save_content 失败: %s", exc, exc_info=True)
        return f"[错误] save_content 写入失败: {exc}"


# --------------------------------------------------------------------------
# 数学计算
# --------------------------------------------------------------------------

@register(
    name="calculator",
    description="安全地计算数学表达式，支持 sqrt/log/sin/cos/pow 等函数",
    signature="calculator(expression)",
    examples=["calculator(sqrt(144) + pow(2, 10))"],
)
def calculator(expression: str) -> str:
    """数学表达式计算器

    使用 numexpr 进行安全计算，支持常见数学函数。

    Args:
        expression: 数学表达式字符串

    Returns:
        计算结果
    """
    expression = expression.strip()
    if not expression:
        return "[错误] calculator: 表达式不能为空"

    logger.info("calculator 调用 expression=%s", expression)

    try:
        import numexpr

        result = numexpr.evaluate(expression)
        # numexpr 返回 numpy 标量，转为 Python 原生类型
        if hasattr(result, "item"):
            result = result.item()

        logger.info("calculator 结果=%s", result)
        return f"计算结果: {expression} = {result}"

    except ImportError:
        logger.error("calculator: numexpr 未安装")
        return "[错误] calculator: numexpr 库未安装，请执行 pip install numexpr"
    except SyntaxError as exc:
        logger.error("calculator 语法错误: %s", exc)
        return f"[错误] calculator: 表达式语法错误 - {exc}"
    except Exception as exc:
        logger.error("calculator 执行失败: %s", exc, exc_info=True)

        # 回退到 Python eval（仅用于安全的数学表达式）
        try:
            # 只允许安全的数学函数和常量
            safe_dict = {
                "sqrt": __import__("math").sqrt,
                "log": __import__("math").log,
                "log10": __import__("math").log10,
                "log2": __import__("math").log2,
                "sin": __import__("math").sin,
                "cos": __import__("math").cos,
                "tan": __import__("math").tan,
                "asin": __import__("math").asin,
                "acos": __import__("math").acos,
                "atan": __import__("math").atan,
                "pow": pow,
                "abs": abs,
                "pi": 3.141592653589793,
                "e": 2.718281828459045,
            }
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            logger.info("calculator (eval fallback) 结果=%s", result)
            return f"计算结果: {expression} = {result}"
        except Exception as e2:
            logger.error("calculator eval fallback 也失败: %s", e2)
            return f"[错误] calculator: 计算失败 - numexpr: {exc}, eval: {e2}"


# --------------------------------------------------------------------------
# 时间查询
# --------------------------------------------------------------------------

@register(
    name="get_current_time",
    description="返回当前本地日期和时间",
    signature="get_current_time()",
    examples=["get_current_time()"],
)
def get_current_time(_param: str = "") -> str:
    """获取当前本地时间

    Returns:
        格式化的日期时间字符串
    """
    logger.info("get_current_time 调用")
    now = datetime.now()
    weekday_map = {
        0: "星期一", 1: "星期二", 2: "星期三",
        3: "星期四", 4: "星期五", 5: "星期六", 6: "星期日",
    }
    weekday = weekday_map[now.weekday()]
    result = now.strftime(f"当前时间: %Y年%m月%d日 {weekday} %H:%M:%S")
    return result


# ==========================================================================
# 工具执行入口
# ==========================================================================

def execute_tool(tool_name: str, tool_args: str) -> str:
    """根据工具名和参数执行对应的工具函数

    Args:
        tool_name: 工具名称（必须已在 TOOLS 注册表中）
        tool_args: 参数字符串（原样传给工具函数）

    Returns:
        工具执行结果字符串（Observation）
    """
    if tool_name not in TOOLS:
        available = ", ".join(TOOLS.keys())
        return f"[错误] 未知工具: {tool_name}。可用工具: {available}"

    tool_func = TOOLS[tool_name]
    try:
        result = tool_func(tool_args)
        return result
    except Exception as exc:
        logger.error(
            "工具 %s 执行异常: %s", tool_name, exc, exc_info=True
        )
        return f"[错误] 工具 {tool_name} 执行失败: {exc}"


def get_tools_description() -> str:
    """生成工具列表的描述文本（用于 System Prompt）"""
    lines = []
    for name, meta in TOOL_META.items():
        lines.append(f"- **{meta['signature']}**: {meta['description']}")
        for ex in meta["examples"]:
            lines.append(f"  - 示例: `{ex}`")
    return "\n".join(lines)
