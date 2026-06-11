"""
Nexa Agent V0 — Streamlit 前端

仅做 UI，所有后端逻辑通过 HTTP 调用 FastAPI (:8000)。
"""

from __future__ import annotations

import os
import base64
import concurrent.futures
import hashlib
import json
import html
import time
import uuid
import requests
from datetime import datetime

import streamlit as st

# ── 后端地址 ──
API_BASE = os.environ.get("NEXA_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Nexa Agent V0", page_icon="N", layout="wide")

# ────────────────────────────────────────────────────────────
# CSS
# ────────────────────────────────────────────────────────────

_CSS = """
<style>
.stApp {
    background: #f7f8fb;
    color: #111827;
}

.block-container,
.stMainBlockContainer {
    max-width: 1000px !important;
    padding-top: 2.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}

.nexa-header {
    text-align: center;
    padding: 0.25rem 0 1.4rem 0;
}
.nexa-header .nexa-icon {
    width: 34px;
    height: 34px;
    border: 1px solid #e8eaf0;
    border-radius: 10px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #ffffff;
    color: #111827;
    font-weight: 700;
    margin-bottom: 0.55rem;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
}
.nexa-header .logo {
    font-size: 36px;
    line-height: 1.18;
    font-weight: 700;
    color: #111827;
    letter-spacing: 0;
}
.nexa-header .tagline {
    font-size: 15px;
    color: #6b7280;
    margin-top: 0.45rem;
}

[data-testid="stSidebar"] {
    background: #ffffff;
    min-width: 280px !important;
    max-width: 280px !important;
    border-right: 1px solid #e8eaf0;
}
[data-testid="stSidebar"] .block-container,
[data-testid="stSidebar"] .stMainBlockContainer {
    padding: 1rem 0.85rem !important;
    max-width: none !important;
}
.sidebar-section-title {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #8b95a7;
    margin: 1rem 0 0.45rem 0;
}
.sidebar-section-title:first-child {
    margin-top: 0;
}
.sidebar-status-row {
    display: flex;
    align-items: center;
    gap: 7px;
    min-height: 24px;
    font-size: 13px;
    color: #374151;
}
.status-dot {
    flex: 0 0 auto;
    width: 7px;
    height: 7px;
    border-radius: 50%;
}
.status-dot.ok  { background: #22c55e; }
.status-dot.err { background: #ef4444; }
.sidebar-status-label {
    font-weight: 650;
}
.sidebar-status-value {
    color: #6b7280;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.session-tag {
    display: inline-block;
    border: 1px solid #e8eaf0;
    background: #f7f8fb;
    border-radius: 8px;
    padding: 0.25rem 0.45rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    font-size: 12px;
    color: #4b5563;
}

[data-testid="stSidebar"] .stButton button {
    min-height: 34px !important;
    font-size: 13px !important;
    border-radius: 8px !important;
    padding: 0.35rem 0.55rem !important;
    border-color: #e8eaf0 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    padding: 0.55rem !important;
    min-height: 78px !important;
    border-color: #e8eaf0 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    font-size: 12px !important;
    padding: 0.25rem 0.45rem !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #e8eaf0 !important;
    border-radius: 18px !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04) !important;
    min-height: 360px !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 1.15rem 1.2rem !important;
}
.nexa-empty-state {
    min-height: 300px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: #6b7280;
    font-size: 15px;
}
.nexa-upload-card {
    max-width: 520px;
    margin: 2.6rem auto;
    text-align: center;
}
.nexa-upload-title {
    font-size: 17px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.25rem;
}
.nexa-upload-hint {
    font-size: 13px;
    color: #6b7280;
    margin-bottom: 0.85rem;
}
.nexa-upload-card [data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section {
    border-color: #e8eaf0 !important;
    border-radius: 16px !important;
}
.nexa-file-preview {
    border: 1px solid #e8eaf0;
    border-radius: 16px;
    background: #fbfcfe;
    padding: 0.9rem;
    margin-bottom: 1rem;
}
.active-file-note {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    border: 1px solid #e8eaf0;
    border-radius: 999px;
    padding: 0.3rem 0.55rem;
    background: #fbfcfe;
    font-size: 12px;
    color: #4b5563;
    margin-bottom: 0.75rem;
}
.nexa-file-title {
    font-size: 13px;
    font-weight: 700;
    color: #374151;
    margin-bottom: 0.25rem;
}
.nexa-file-meta {
    font-size: 13px;
    color: #6b7280;
}
.file-type-pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 38px;
    height: 26px;
    padding: 0 0.45rem;
    border-radius: 8px;
    border: 1px solid #e8eaf0;
    background: #ffffff;
    font-size: 12px;
    font-weight: 700;
    color: #4b5563;
    margin-right: 0.45rem;
}
.nexa-msg-row {
    display: flex;
    width: 100%;
    margin: 0.55rem 0;
}
.nexa-msg-row.user {
    justify-content: flex-end;
}
.nexa-msg-row.assistant {
    justify-content: flex-start;
}
.nexa-msg {
    max-width: min(76%, 680px);
    border-radius: 16px;
    padding: 14px 16px;
    font-size: 15px;
    line-height: 1.58;
    border: 1px solid #e8eaf0;
    overflow-wrap: anywhere;
}
.nexa-msg.user {
    background: #f1f3f7;
    color: #111827;
}
.nexa-msg.assistant {
    background: #ffffff;
    color: #111827;
}
.nexa-msg.pending {
    color: #4b5563;
    background: #fbfcfe;
}
.nexa-msg.error {
    border-color: #fecaca;
    color: #991b1b;
    background: #fffafa;
}
.nexa-msg p {
    margin: 0 0 0.75rem 0;
}
.nexa-msg p:last-child {
    margin-bottom: 0;
}
.nexa-msg ul,
.nexa-msg ol {
    margin: 0.35rem 0 0.85rem 1.2rem;
    padding-left: 0.7rem;
}
.nexa-msg li {
    margin: 0.18rem 0;
}
.nexa-msg strong {
    font-weight: 750;
}
.nexa-msg code {
    font-size: 0.92em;
    background: #f7f8fb;
    border: 1px solid #e8eaf0;
    border-radius: 6px;
    padding: 0.05rem 0.25rem;
}
.message-files {
    margin-top: 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
}
.message-file-card {
    border: 1px solid #e8eaf0;
    border-radius: 14px;
    background: rgba(255,255,255,0.62);
    padding: 0.65rem;
}
.message-file-card img {
    display: block;
    width: min(420px, 100%);
    height: auto;
    border-radius: 12px;
    border: 1px solid #e8eaf0;
    margin-top: 0.55rem;
}
div[data-testid="stForm"] {
    margin: 0.75rem auto 0 auto;
    max-width: 1000px;
    border: 1px solid #e8eaf0;
    border-radius: 16px;
    background: #ffffff;
    padding: 0.7rem;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
}
div[data-testid="stForm"] [data-testid="stTextInput"] input {
    min-height: 42px !important;
    border-radius: 12px !important;
    border-color: #e8eaf0 !important;
    font-size: 15px !important;
}
div[data-testid="stForm"] .stButton button {
    min-height: 42px !important;
    border-radius: 12px !important;
    font-size: 14px !important;
    border-color: #e8eaf0 !important;
}

[data-testid="stExpander"] {
    border: 1px solid #eef0f5 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
    background: #fff !important;
    margin: 0.35rem 0 0.6rem 0 !important;
}
[data-testid="stExpander"] summary {
    font-size: 13px !important;
    color: #6b7280 !important;
}
.trace-summary {
    font-size: 13px;
    color: #4b5563;
    margin-bottom: 0.65rem;
}
.trace-timeline {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
}
.trace-item {
    border: 1px solid #e8eaf0;
    border-radius: 12px;
    padding: 0.55rem 0.65rem;
    background: #ffffff;
}
.trace-item.failed,
.trace-item.error {
    border-color: #fecaca;
}
.trace-main {
    display: grid;
    grid-template-columns: 18px 1fr auto auto;
    align-items: center;
    gap: 0.5rem;
    font-size: 13px;
}
.trace-node {
    color: #111827;
    font-weight: 650;
}
.trace-badge {
    border: 1px solid #e8eaf0;
    border-radius: 999px;
    padding: 0.1rem 0.45rem;
    font-size: 12px;
    color: #4b5563;
    background: #fbfcfe;
}
.trace-badge.failed,
.trace-badge.error {
    border-color: #fecaca;
    color: #991b1b;
}
.trace-badge.success {
    color: #166534;
}
.trace-duration {
    color: #6b7280;
    font-size: 12px;
    white-space: nowrap;
}
.trace-detail {
    margin: 0.45rem 0 0 1.6rem;
    color: #6b7280;
    font-size: 12px;
    line-height: 1.5;
}
.trace-detail code {
    white-space: pre-wrap;
}

@media (max-width: 768px) {
    .block-container,
    .stMainBlockContainer {
        max-width: 100% !important;
        padding-top: 2rem !important;
        padding-left: 0.85rem !important;
        padding-right: 0.85rem !important;
    }
    .nexa-header .logo {
        font-size: 32px;
    }
    .nexa-msg {
        max-width: 92%;
    }
}
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
# HTTP helpers
# ────────────────────────────────────────────────────────────

@st.cache_data(ttl=5, show_spinner=False)
def _health_check() -> dict | None:
    try:
        return requests.get(f"{API_BASE}/api/v0/health", timeout=3).json()
    except Exception:
        return None


def _post_chat(payload: dict) -> dict:
    try:
        return requests.post(f"{API_BASE}/api/v0/chat", json=payload, timeout=300).json()
    except Exception as exc:
        return {"error": str(exc), "status": "error"}


def _post_upload(uploaded, session_id: str) -> dict:
    data = uploaded.getvalue()
    try:
        files = {
            "file": (
                uploaded.name,
                data,
                uploaded.type or "application/octet-stream",
            )
        }
        resp = requests.post(
            f"{API_BASE}/api/v0/upload",
            files=files,
            data={"session_id": session_id},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        d = os.path.join("data", "uploads")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"st_{session_id}_{uuid.uuid4().hex[:8]}_{uploaded.name}")
        with open(path, "wb") as f:
            f.write(data)
        file_id = f"st_{session_id}_{uuid.uuid4().hex[:8]}"
        return {
            "session_id": session_id,
            "filename": uploaded.name,
            "file_path": path,
            "file_size": len(data),
            "content_type": uploaded.type,
            "file_id": file_id,
            "file_sha256": hashlib.sha256(data).hexdigest(),
            "fallback_local": True,
        }


def _post_image_analysis(file_info: dict, session_id: str) -> dict:
    payload = {
        "session_id": session_id,
        "file_id": file_info.get("backend_file_id") or file_info.get("file_id") or file_info.get("id"),
        "file_sha256": file_info.get("file_sha256", ""),
        "file_path": file_info.get("path"),
        "filename": file_info.get("name"),
        "content_type": file_info.get("type"),
    }
    try:
        resp = requests.post(f"{API_BASE}/api/v0/files/analyze", json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"status": "failed", "error_message": str(exc), **payload}


def _delete_session(sid: str):
    try:
        requests.delete(f"{API_BASE}/api/v0/memory/session/{sid}", timeout=3)
    except Exception:
        pass


def _get_timeline(rid: str) -> dict | None:
    try:
        return requests.get(
            f"{API_BASE}/api/v0/trace/{rid}/timeline", timeout=3
        ).json()
    except Exception:
        return None


def _get_trace_events(rid: str) -> list[dict]:
    if not rid:
        return []
    try:
        data = requests.get(f"{API_BASE}/api/v0/trace/{rid}/events", timeout=3).json()
        return data.get("events", [])
    except Exception:
        return []


@st.cache_resource(show_spinner=False)
def _executor() -> concurrent.futures.ThreadPoolExecutor:
    return concurrent.futures.ThreadPoolExecutor(max_workers=2)


# ────────────────────────────────────────────────────────────
# Session init
# ────────────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file_info" not in st.session_state:
    st.session_state.uploaded_file_info = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "current_files" not in st.session_state:
    migrated = st.session_state.uploaded_file_info
    st.session_state.current_files = [migrated] if migrated else []
if "active_file" not in st.session_state:
    st.session_state.active_file = (
        st.session_state.current_files[-1] if st.session_state.current_files else None
    )
if "upload_widget_version" not in st.session_state:
    st.session_state.upload_widget_version = 0
if "pending_request" not in st.session_state:
    st.session_state.pending_request = None
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "input_notice" not in st.session_state:
    st.session_state.input_notice = ""
if "pending_image_analysis" not in st.session_state:
    st.session_state.pending_image_analysis = None

sid = st.session_state.session_id


# ────────────────────────────────────────────────────────────
# Render: Sidebar
# ────────────────────────────────────────────────────────────

def render_sidebar():
    """左侧紧凑信息面板"""
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-section-title">状态</div>',
            unsafe_allow_html=True,
        )

        health = _health_check()
        if health:
            be_ok = health.get("status") == "ok"
            llm_ok = health.get("llm_available", False)
            vlm_ok = health.get("vlm_available", False)
            llm_model = health.get("llm_model", "")

            _metric_with_dot("后端", "在线" if be_ok else "离线", be_ok)
            _metric_with_dot("LLM", llm_model if llm_ok else "不可用", llm_ok)
            _metric_with_dot("VLM", "可用" if vlm_ok else "不可用", vlm_ok)
        else:
            _metric_with_dot("后端", "离线", False)
            _metric_with_dot("LLM", "—", False)
            _metric_with_dot("VLM", "—", False)

        st.markdown(
            '<div class="sidebar-section-title">会话</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span class="session-tag">{html.escape(sid[:12])}</span>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        if c1.button("新建", use_container_width=True, key="btn_new_session"):
            _new_session()
        if c2.button("清空", use_container_width=True, key="btn_clear_session"):
            _delete_session(sid)
            _new_session()

        st.markdown(
            '<div class="sidebar-section-title">上传票据</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "选择图片或 PDF",
            type=["jpg", "jpeg", "png", "pdf"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key=f"sidebar_uploader_{st.session_state.upload_widget_version}",
        )
        if uploaded and not _is_same_uploaded_file(uploaded, st.session_state.get("active_file")):
            _handle_upload(uploaded, source="sidebar")


def _new_session():
    st.session_state.session_id = uuid.uuid4().hex[:8]
    st.session_state.messages = []
    st.session_state.uploaded_files = []
    st.session_state.current_files = []
    st.session_state.active_file = None
    st.session_state.uploaded_file_info = None
    st.session_state.upload_widget_version += 1
    st.session_state.pending_request = None
    st.session_state.is_generating = False
    st.session_state.input_notice = ""
    st.session_state.pending_image_analysis = None
    st.rerun()


def _is_same_uploaded_file(uploaded, info: dict | None) -> bool:
    return bool(
        uploaded
        and info
        and info.get("name") == getattr(uploaded, "name", None)
        and info.get("size") == getattr(uploaded, "size", None)
    )


def _handle_upload(uploaded, source: str):
    result = _post_upload(uploaded, st.session_state.session_id)
    info = {
        "id": uuid.uuid4().hex[:12],
        "file_id": result.get("file_id", ""),
        "file_sha256": result.get("file_sha256", ""),
        "name": result.get("filename") or uploaded.name,
        "path": result.get("file_path"),
        "backend_file_id": result.get("file_id", ""),
        "size": result.get("file_size") or uploaded.size,
        "type": result.get("content_type") or uploaded.type or "application/octet-stream",
        "source": source,
        "uploaded_at": datetime.utcnow().isoformat(),
        "fallback_local": result.get("fallback_local", False),
        "analysis_status": "pending",
        "vlm_text": "",
        "structured_data": {},
    }
    st.session_state.uploaded_files.append(info)
    st.session_state.current_files = [info]
    st.session_state.active_file = info
    st.session_state.uploaded_file_info = info
    future = _executor().submit(_post_image_analysis, dict(info), st.session_state.session_id)
    st.session_state.pending_image_analysis = {
        "file_id": info["id"],
        "future": future,
        "started_at": time.time(),
    }
    st.session_state.input_notice = ""
    st.rerun()


def _remove_active_file():
    st.session_state.current_files = []
    st.session_state.active_file = None
    st.session_state.uploaded_file_info = None
    st.session_state.pending_image_analysis = None
    st.session_state.upload_widget_version += 1
    st.rerun()


def _update_file_info(file_id: str, updates: dict):
    for item in st.session_state.get("uploaded_files", []):
        if item.get("id") == file_id:
            item.update(updates)
    for item in st.session_state.get("current_files", []):
        if item.get("id") == file_id:
            item.update(updates)
    active = st.session_state.get("active_file")
    if active and active.get("id") == file_id:
        active.update(updates)
        st.session_state.uploaded_file_info = active


def _finish_pending_image_analysis():
    pending = st.session_state.get("pending_image_analysis")
    if not pending:
        return
    future = pending.get("future")
    file_id = pending.get("file_id", "")
    if not future:
        st.session_state.pending_image_analysis = None
        return
    if not future.done():
        elapsed = int((time.time() - pending.get("started_at", time.time())) * 1000)
        _update_file_info(file_id, {"analysis_status": "pending", "analysis_elapsed_ms": elapsed})
        return

    try:
        resp = future.result()
    except Exception as exc:
        resp = {"status": "failed", "error_message": str(exc)}

    status = resp.get("status", "failed")
    updates = {
        "analysis_status": status,
        "analysis_id": resp.get("file_id") or resp.get("file_sha256", ""),
        "analysis_cached": resp.get("cached", False),
        "analysis_latency_ms": resp.get("latency_ms", 0),
        "analysis_error": resp.get("error_message"),
        "vlm_text": resp.get("vlm_text", ""),
        "structured_data": resp.get("structured_data", {}) or {},
        "analysis_model_name": resp.get("model_name", ""),
    }
    _update_file_info(file_id, updates)
    st.session_state.pending_image_analysis = None
    st.rerun()


def _metric_with_dot(label: str, value: str, ok: bool):
    dot_cls = "ok" if ok else "err"
    st.markdown(
        '<div class="sidebar-status-row">'
        f'<span class="status-dot {dot_cls}"></span>'
        f'<span class="sidebar-status-label">{html.escape(label)}</span>'
        f'<span class="sidebar-status-value">{html.escape(value)}</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────────────────
# Render: Header
# ────────────────────────────────────────────────────────────

def render_header():
    st.markdown(
        '<div class="nexa-header">'
        '<div class="nexa-icon">N</div>'
        '<div class="logo">Nexa Agent V0</div>'
        '<div class="tagline">多模态票据问答 · RAG · Agent 工作流</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────────────────
# Render: File preview
# ────────────────────────────────────────────────────────────

def _format_bytes(size: int | float | None) -> str:
    size = float(size or 0)
    if size >= 1024 * 1024:
        return f"{size / 1024 / 1024:.1f} MB"
    return f"{size / 1024:.1f} KB"


def _file_suffix(info: dict) -> str:
    fname = info.get("name", "file")
    return os.path.splitext(fname)[1].replace(".", "").upper() or "FILE"


def _is_image_file(info: dict) -> bool:
    ftype = info.get("type") or ""
    suffix = _file_suffix(info).lower()
    return ftype.startswith("image/") or suffix in {"jpg", "jpeg", "png"}


def _image_mime(info: dict) -> str:
    ftype = info.get("type") or ""
    suffix = _file_suffix(info).lower()
    if ftype.startswith("image/"):
        return ftype
    if suffix in {"jpg", "jpeg"}:
        return "image/jpeg"
    return "image/png"


def _image_preview_html(info: dict, classless: bool = False) -> str:
    path = info.get("path")
    if not path or not _is_image_file(info):
        return ""
    try:
        with open(path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("ascii")
    except Exception:
        return (
            '<div class="nexa-file-meta" style="margin-top:0.6rem;">'
            f'无法预览图片：{html.escape(info.get("name", ""))}'
            '</div>'
        )
    style = ""
    if classless:
        style = (
            ' style="display:block;width:min(420px,100%);height:auto;'
            'border-radius:12px;margin-top:0.55rem;border:1px solid #e8eaf0;"'
        )
    return f'<img src="data:{html.escape(_image_mime(info))};base64,{encoded}"{style} />'


def file_card_html(info: dict, css_class: str = "nexa-file-preview") -> str:
    fname = html.escape(info.get("name", "未命名文件"))
    ftype = html.escape(info.get("type") or "未知类型")
    suffix = html.escape(_file_suffix(info)[:4])
    size = _format_bytes(info.get("size"))
    image_html = _image_preview_html(info, classless=(css_class == "message-file-card"))
    analysis_status = info.get("analysis_status")
    analysis_html = ""
    if analysis_status == "pending":
        analysis_html = '<div class="nexa-file-meta" style="margin-top:0.45rem;">图片识别中，完成后本会话将复用识别内容</div>'
    elif analysis_status == "success":
        cached = " · 缓存命中" if info.get("analysis_cached") else ""
        latency = _duration_label(info.get("analysis_latency_ms"))
        analysis_html = (
            '<div class="nexa-file-meta" style="margin-top:0.45rem;">'
            f'已识别图片内容{cached}'
            f'{(" · " + html.escape(latency)) if latency else ""}'
            '</div>'
        )
    elif analysis_status == "failed":
        err = html.escape(info.get("analysis_error") or "图片识别失败")
        analysis_html = f'<div class="nexa-file-meta" style="margin-top:0.45rem;color:#991b1b;">{err}</div>'
    return (
        f'<div class="{css_class}">'
        '<div class="nexa-file-title">'
        f'<span class="file-type-pill">{suffix}</span>{fname}</div>'
        f'<div class="nexa-file-meta">{html.escape(size)} · {ftype}</div>'
        f'{analysis_html}'
        f'{image_html}'
        '</div>'
    )


def render_file_preview(info: dict | None = None, allow_remove: bool = True):
    """展示当前 active 文件。"""
    info = info or st.session_state.get("active_file")
    if not info:
        return
    st.markdown(
        '<div class="active-file-note">当前上下文文件 · '
        f'{html.escape(info.get("name", "未命名文件"))}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(file_card_html(info), unsafe_allow_html=True)
    if allow_remove and st.button("移除当前文件", key="btn_remove_upload"):
        _remove_active_file()


def render_active_file_context(info: dict):
    st.markdown(
        '<div class="active-file-note">本轮继续使用 · '
        f'{html.escape(info.get("name", "未命名文件"))}</div>',
        unsafe_allow_html=True,
    )
    if st.button("移除当前文件", key="btn_remove_active_context"):
        _remove_active_file()


# ────────────────────────────────────────────────────────────
# Render: Trace and chat message
# ────────────────────────────────────────────────────────────

NODE_DISPLAY_NAMES = {
    "normalize_input": "输入标准化",
    "load_short_term_context": "加载上下文",
    "route_task": "任务路由",
    "react_decide": "推理决策",
    "execute_tool": "工具执行",
    "react_finish": "完成",
    "respond": "回答生成",
    "update_memory": "记忆写入",
    "fallback": "兜底处理",
    "trace_completed": "请求完成",
    "trace_failed": "请求失败",
    "model_call": "模型调用",
}


def get_node_display_name(node_name: str | None) -> str:
    if not node_name:
        return "系统事件"
    return NODE_DISPLAY_NAMES.get(node_name, node_name.replace("_", " "))


def get_event_status_badge(status: str | None) -> str:
    status = (status or "success").lower()
    labels = {
        "success": "success",
        "completed": "success",
        "failed": "failed",
        "error": "failed",
        "skipped": "skipped",
        "running": "running",
        "pending": "pending",
        "waiting": "waiting",
    }
    label = labels.get(status, status)
    return f'<span class="trace-badge {html.escape(label)}">{html.escape(label)}</span>'


def _duration_label(ms: int | float | None) -> str:
    if ms is None:
        return ""
    try:
        value = float(ms)
    except Exception:
        return ""
    if value >= 1000:
        return f"{value / 1000:.1f}s"
    return f"{value:.0f}ms"


def _payload_summary(payload: dict | None) -> str:
    payload = payload or {}
    if not payload:
        return ""
    compact = json.dumps(payload, ensure_ascii=False, default=str)
    if len(compact) > 520:
        compact = compact[:520] + "..."
    return compact


def _event_detail_html(event: dict) -> str:
    payload = event.get("payload") or {}
    details = [
        f"event_type: {event.get('event_type', '')}",
        f"node_name: {event.get('node_name') or ''}",
    ]
    if event.get("message"):
        details.append(f"message: {event.get('message')}")
    if event.get("output_summary"):
        details.append(f"output_summary: {event.get('output_summary')}")
    if event.get("error_message"):
        details.append(f"error_message: {event.get('error_message')}")

    if event.get("event_type") == "model_call_completed":
        model = payload.get("model_name") or "unknown"
        purpose = payload.get("purpose") or payload.get("node_name") or "model_call"
        latency = _duration_label(payload.get("latency_ms") or event.get("duration_ms"))
        tokens = payload.get("total_tokens")
        details.append(f"model_name: {model}")
        details.append(f"purpose: {purpose}")
        if latency:
            details.append(f"latency: {latency}")
        if tokens is not None:
            details.append(f"tokens: {tokens}")

    payload_text = _payload_summary(payload)
    if payload_text:
        details.append(f"payload: {payload_text}")
    escaped = "<br>".join(html.escape(item) for item in details if item)
    return f'<div class="trace-detail">{escaped}</div>'


def render_trace_timeline_item(event: dict):
    status = event.get("event_status") or event.get("status") or "success"
    level = event.get("event_level") or ""
    item_cls = "failed" if status == "failed" or level == "error" else status
    node_name = event.get("node_name") or event.get("event_type")
    display = get_node_display_name(node_name)
    if event.get("event_type") == "model_call_completed":
        payload = event.get("payload") or {}
        display = f"模型调用 · {payload.get('model_name') or 'unknown'}"
    duration = _duration_label(event.get("duration_ms"))
    icon = "!" if status == "failed" or level == "error" else "·"
    st.markdown(
        f'<div class="trace-item {html.escape(item_cls)}">'
        '<div class="trace-main">'
        f'<span>{html.escape(icon)}</span>'
        f'<span class="trace-node">{html.escape(display)}</span>'
        f'{get_event_status_badge(status)}'
        f'<span class="trace-duration">{html.escape(duration)}</span>'
        '</div>'
        '<details>'
        '<summary class="trace-detail">查看详情</summary>'
        f'{_event_detail_html(event)}'
        '</details>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_trace_panel(meta: dict | None):
    if not meta:
        return
    trace_events = meta.get("trace_events") or []
    rid = meta.get("request_id", "")
    if not trace_events and rid:
        trace_events = _get_trace_events(rid)
    if not trace_events and meta.get("timeline_items"):
        trace_events = [
            {
                "event_type": "node_completed",
                "event_status": item.get("status", "success"),
                "node_name": item.get("node_name"),
                "title": item.get("title"),
                "duration_ms": item.get("duration_ms"),
            }
            for item in meta.get("timeline_items", [])
        ]

    step_count = len(trace_events) or len(meta.get("路径", "").split("→"))
    latency = meta.get("耗时", "—")
    vlm_c = meta.get("VLM", 0)
    llm_c = meta.get("LLM", 0)
    failed = any(
        (e.get("event_status") == "failed" or e.get("event_level") == "error")
        for e in trace_events
    )
    status = "有失败" if failed else "已完成"
    summary = f"{status} · {step_count} 个步骤 · {latency} · LLM {llm_c} 次 · VLM {vlm_c} 次"

    with st.expander(f"调用详情 · {summary}", expanded=False):
        st.markdown(
            f'<div class="trace-summary">{html.escape(summary)}</div>',
            unsafe_allow_html=True,
        )
        if trace_events:
            st.markdown('<div class="trace-timeline">', unsafe_allow_html=True)
            for event in trace_events:
                render_trace_timeline_item(event)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.caption("暂无 trace event。")


def render_chat_message(msg: dict):
    """渲染单条聊天消息、附件和调用详情。"""
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    role_cls = "user" if role == "user" else "assistant"
    status_cls = msg.get("status", "")
    files = msg.get("files") or []

    st.markdown(
        f'<div class="nexa-msg-row {role_cls}">'
        f'<div class="nexa-msg {role_cls} {html.escape(status_cls)}">',
        unsafe_allow_html=True,
    )
    st.markdown(content or "")
    if files:
        cards = "".join(file_card_html(info, css_class="message-file-card") for info in files)
        st.markdown(f'<div class="message-files">{cards}</div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    if role_cls == "assistant" and msg.get("meta") and msg.get("status") != "pending":
        render_trace_panel(msg.get("meta"))


def render_chat_panel():
    """主对话容器：文件预览、消息和空状态在同一层级展示。"""
    with st.container(border=True):
        active_file = st.session_state.get("active_file")
        messages = st.session_state.messages

        if active_file and not messages:
            render_file_preview(active_file, allow_remove=True)
        elif active_file:
            render_active_file_context(active_file)

        if not messages and not active_file:
            render_main_upload_card()
            return

        for msg in messages:
            render_chat_message(msg)


def render_main_upload_card():
    st.markdown(
        '<div class="nexa-upload-card">'
        '<div class="nexa-upload-title">上传票据或文档</div>'
        '<div class="nexa-upload-hint">支持 JPG / PNG / PDF，最大 200MB，点击选择文件，或拖拽到此处</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "上传票据或文档",
        type=["jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=False,
        key=f"main_file_uploader_{st.session_state.upload_widget_version}",
    )
    if uploaded and not _is_same_uploaded_file(uploaded, st.session_state.get("active_file")):
        _handle_upload(uploaded, source="main")


def render_input_bar(disabled: bool = False) -> str | None:
    """与 Chat Panel 对齐的普通输入栏，避免 st.chat_input 固定在页面底部。"""
    with st.form("nexa_input_form", clear_on_submit=True, border=False):
        c1, c2 = st.columns([0.84, 0.16])
        with c1:
            prompt = st.text_input(
                "输入问题",
                placeholder="输入问题，例如：这张发票的金额是多少？",
                label_visibility="collapsed",
                disabled=disabled,
            )
        with c2:
            submitted = st.form_submit_button(
                "分析中" if disabled else "发送",
                use_container_width=True,
                disabled=disabled,
            )

    if st.session_state.get("input_notice"):
        st.warning(st.session_state.input_notice)

    if submitted and prompt.strip():
        st.session_state.input_notice = ""
        return prompt.strip()
    if submitted:
        st.session_state.input_notice = "请输入你想问的问题。"
        st.rerun()
    return None


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _copy_current_files() -> list[dict]:
    return [dict(item) for item in st.session_state.get("current_files", []) if item]


def _find_message(message_id: str) -> dict | None:
    for msg in st.session_state.messages:
        if msg.get("id") == message_id:
            return msg
    return None


def _start_chat_request(prompt: str):
    files = _copy_current_files()
    if files and files[0].get("analysis_status") == "pending":
        st.session_state.input_notice = "图片识别中，请稍候再提问。"
        return
    if files and files[0].get("analysis_status") == "failed":
        st.session_state.input_notice = "图片识别失败，请移除或重新上传后再进行图片问答。"
        return
    user_id = uuid.uuid4().hex[:12]
    assistant_id = uuid.uuid4().hex[:12]
    st.session_state.messages.append({
        "id": user_id,
        "role": "user",
        "content": prompt,
        "files": files,
        "created_at": _now_iso(),
    })
    st.session_state.messages.append({
        "id": assistant_id,
        "role": "assistant",
        "content": "正在分析票据..." if files else "正在思考...",
        "files": [],
        "created_at": _now_iso(),
        "status": "pending",
        "meta": {
            "trace_events": [
                    {
                        "event_type": "trace_started",
                        "event_status": "running",
                        "node_name": "normalize_input",
                        "title": "请求已提交",
                        "message": "等待后端返回结果",
                    }
            ]
        },
    })

    payload: dict = {"session_id": st.session_state.session_id, "message": prompt}
    if files:
        # TODO: 后端当前 chat 接口只接收 image_path；若后续支持 file_id/image_id，可在此改为传 ID。
        payload["image_path"] = files[0].get("path")
        payload["metadata"] = {
            "active_file": {
                "name": files[0].get("name"),
                "type": files[0].get("type"),
                "backend_file_id": files[0].get("backend_file_id"),
                "analysis_id": files[0].get("analysis_id"),
                "vlm_text": files[0].get("vlm_text", ""),
                "structured_data": files[0].get("structured_data", {}),
            }
        }

    future = _executor().submit(_post_chat, payload)
    st.session_state.pending_request = {
        "assistant_message_id": assistant_id,
        "user_message_id": user_id,
        "started_at": time.time(),
        "payload": payload,
        "future": future,
    }
    st.session_state.is_generating = True


def _finish_pending_request():
    pending = st.session_state.get("pending_request")
    if not pending:
        return
    future = pending.get("future")
    assistant_msg = _find_message(pending.get("assistant_message_id", ""))
    if not future:
        st.session_state.pending_request = None
        st.session_state.is_generating = False
        return

    if not future.done():
        if assistant_msg:
            elapsed = int((time.time() - pending.get("started_at", time.time())) * 1000)
            assistant_msg["meta"] = {
                "trace_events": [
                    {
                        "event_type": "node_started",
                        "event_status": "running",
                        "node_name": "route_task",
                        "title": "分析中",
                        "duration_ms": elapsed,
                        "message": "后端正在执行 Agent 工作流",
                    }
                ]
            }
        time.sleep(0.8)
        st.rerun()
        return

    try:
        resp = future.result()
    except Exception as exc:
        resp = {"error": str(exc), "status": "error"}

    if not assistant_msg:
        assistant_msg = {
            "id": pending.get("assistant_message_id", uuid.uuid4().hex[:12]),
            "role": "assistant",
        }
        st.session_state.messages.append(assistant_msg)

    if resp.get("error") or resp.get("status") == "error":
        assistant_msg.update({
            "content": f"请求失败：{resp.get('error', '未知错误')}",
            "status": "error",
            "meta": {
                "trace_events": [
                    {
                        "event_type": "trace_failed",
                        "event_status": "failed",
                        "event_level": "error",
                        "node_name": "trace_failed",
                        "title": "请求失败",
                        "error_message": resp.get("error", "未知错误"),
                    }
                ]
            },
        })
    else:
        rid = resp.get("request_id", "")
        trace_events = _get_trace_events(rid)
        timeline = _get_timeline(rid) or {}
        assistant_msg.update({
            "content": resp.get("response", ""),
            "status": "done",
            "request_id": rid,
            "meta": {
                "任务": resp.get("task_type", ""),
                "路径": "→".join(resp.get("execution_path", [])),
                "LLM": resp.get("llm_calls", 0),
                "VLM": resp.get("vlm_calls", 0),
                "耗时": f"{resp.get('latency_ms', 0):.0f}ms",
                "置信度": resp.get("confidence"),
                "request_id": rid,
                "trace_events": trace_events,
                "timeline_items": timeline.get("items", []),
            },
        })

    st.session_state.pending_request = None
    st.session_state.is_generating = False
    st.rerun()


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main():
    _finish_pending_image_analysis()
    render_sidebar()
    render_header()

    render_chat_panel()
    active_file = st.session_state.get("active_file") or {}
    image_analysis_pending = active_file.get("analysis_status") == "pending"
    prompt = render_input_bar(disabled=st.session_state.is_generating or image_analysis_pending)

    if prompt and not st.session_state.is_generating:
        _start_chat_request(prompt)
        st.rerun()

    if st.session_state.get("pending_image_analysis"):
        time.sleep(0.8)
        st.rerun()

    _finish_pending_request()


if __name__ == "__main__":
    main()
