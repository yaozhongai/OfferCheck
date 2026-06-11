"""
tavily_extract / save_content — 网页内容提取与知识库沉淀

tavily_extract: 提取网页正文 → 内存缓存（会话级）
save_content: 缓存内容 → SQLite kb_documents 表
"""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime

from app.tools import register
from app.utils.logger_config import get_logger

logger = get_logger("tools.extract")

# 会话级缓存（单进程内存，后续可迁移到 STM）
_session_extracts: list[dict] = []


def get_session_extracts() -> list[dict]:
    return _session_extracts


def clear_session_extracts() -> None:
    global _session_extracts
    _session_extracts = []


@register(
    name="tavily_extract",
    description="提取指定网页的正文内容并缓存到内存。如需长期保存，后续可调用 save_content",
    signature="tavily_extract(url)",
    examples=["tavily_extract(https://example.com/report.pdf)"],
)
def tavily_extract(url: str) -> str:
    url = url.strip()
    if not url:
        return "[错误] tavily_extract: URL 不能为空"

    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if not tavily_key:
        return "[错误] tavily_extract: 未配置 TAVILY_API_KEY"

    logger.info("tavily_extract url=%s", url[:80])

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=tavily_key)
        response = client.extract(url)
        results = response.get("results", [])
        if not results:
            return f"Tavily Extract 未能从 '{url}' 提取到内容。"

        raw = results[0].get("raw_content", "")
        title = results[0].get("title", "")
        if not raw:
            return f"'{url}' 页面无正文内容。"

        _session_extracts.append({"url": url, "title": title, "raw_content": raw})

        preview = raw[:1500]
        if len(raw) > 1500:
            preview += "..."

        logger.info("tavily_extract 完成 url=%s len=%d cached=true", url, len(raw))
        return (
            f"Tavily Extract - {title or url}\n"
            f"总字符数: {len(raw)}\n\n"
            f"--- 内容预览（前 1500 字符）---\n{preview}\n\n"
            f"💡 全文已缓存。如需长期保存可调用 save_content(文件名) 存入知识库。"
        )
    except ImportError:
        return "[错误] tavily_extract: tavily-python 未安装"
    except Exception as exc:
        return f"[错误] tavily_extract 执行失败: {exc}"


@register(
    name="save_content",
    description="将最近一次 tavily_extract 缓存的内容保存到知识库（SQLite kb_documents 表）",
    signature="save_content(filename)",
    examples=["save_content(apple_environment_report_2025)"],
)
def save_content(filename: str) -> str:
    filename = filename.strip()
    if not filename:
        return "[错误] save_content: 文件名不能为空"

    if not _session_extracts:
        return "[错误] save_content: 没有可保存的内容。请先调用 tavily_extract(url) 提取网页内容。"

    extract = _session_extracts[-1]
    title = extract.get("title", filename)
    raw = extract.get("raw_content", "")
    url = extract.get("url", "")
    content_summary = raw[:2000]

    try:
        from app.storage.database import get_session, init_db
        from app.storage.models import Base
        from sqlalchemy import Column, Integer, String, Text, Float, DateTime
        init_db()

        s = get_session()
        try:
            doc_id = f"kb_{uuid.uuid4().hex[:12]}"
            content_hash = hashlib.sha256(raw.encode()).hexdigest()

            # 检查重复
            existing = s.execute(
                f"SELECT doc_id FROM kb_documents WHERE content_hash = '{content_hash}'"
            ).fetchone() if hasattr(s, 'execute') else None

            if existing:
                return f"⏭ 文档已存在: {title}（hash={content_hash[:12]}）"

            # 原生 SQL 插入（避免 ORM 模型未定义的问题）
            from sqlalchemy import text
            s.execute(text(
                "INSERT INTO kb_documents (doc_id, title, content, content_full, content_hash, "
                "source_url, doc_type, tags_json, extracted_at, status, created_at) "
                "VALUES (:doc_id, :title, :content, :content_full, :content_hash, "
                ":source_url, :doc_type, :tags_json, :extracted_at, :status, :created_at)"
            ), {
                "doc_id": doc_id,
                "title": title or filename,
                "content": content_summary,
                "content_full": raw,
                "content_hash": content_hash,
                "source_url": url,
                "doc_type": "web_article",
                "tags_json": "[]",
                "extracted_at": datetime.utcnow(),
                "status": "active",
                "created_at": datetime.utcnow(),
            })
            s.commit()
            logger.info("save_content 完成 doc_id=%s title=%s len=%d", doc_id, title, len(raw))
            return f"✅ 已保存到知识库: {title or filename}\n字符数: {len(raw)}\n来源: {url}"
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()
    except Exception as exc:
        logger.error("save_content 失败: %s", exc)
        return f"[错误] save_content: {exc}"
