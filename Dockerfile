# syntax=docker/dockerfile:1
# Nexa Agent — 单镜像：FastAPI 后端 + 同源托管静态前端（web/out）

# ---- Stage 1: 构建 Next.js 静态前端 → web/out ----
FROM node:20-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
ENV NEXT_OUTPUT_EXPORT=true
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ---- Stage 2: Python 后端（3.11，避开 3.9 的 PEP-604 导入问题）----
FROM python:3.11-slim AS app
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 引擎 + 场景层 + 服务层
COPY nexa_agent/ ./nexa_agent/
COPY offercheck/ ./offercheck/
COPY server/ ./server/

# 构建产物：静态前端（由 FastAPI StaticFiles 同源托管）
COPY --from=web /web/out ./web/out

EXPOSE 8000
# Railway 注入 $PORT；本地默认 8000。shell 形式以便展开变量。
CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
