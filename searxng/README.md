# 自建搜索后端 — SearXNG

Nexa Agent 的零成本、无速率限制搜索后端，用于替代额度受限的 Tavily。
SearXNG 是开源元搜索引擎，单次查询聚合 Bing / DuckDuckGo / Brave / Startpage / Wikipedia 等多源，
返回带摘要（`content`）和相关度（`score`）的 JSON —— 与 Tavily 同构，可直接喂给 Agent。

## 为什么自建

| | Tavily | 自建 SearXNG |
|---|---|---|
| 免费额度 | 1000/月，用尽即停 | **无限**（仅服务器成本） |
| 速率限制 | 有 | **无** |
| 数据源 | 单一 | 70+ 引擎聚合 |
| 隐私 | 第三方 | 全本地 |
| 运维 | 零 | ~30min/周 |

## 快速开始

```bash
cd searxng
docker compose up -d                 # 启动（首次会拉镜像）
python ../searxng/smoke_test.py      # 健康检查，应输出 "后端健康 ✅"
```

服务监听 `http://localhost:8888`，**仅绑定 127.0.0.1**（私有，不对外）。

## API 用法

```bash
# 基础查询
curl 'http://localhost:8888/search?q=DeepSeek+Harness&format=json'

# 指定语言
curl 'http://localhost:8888/search?q=深度求索&format=json&language=zh'
```

返回结果每条含字段：`title` / `url` / `content`(摘要) / `score` / `engine` / `publishedDate`。

## 配置说明

- **`docker-compose.yml`** — 端口绑定 127.0.0.1、最小 capability、日志限额
- **`config/settings.yml`** — 仅覆盖必要项，其余继承官方默认：
  - `search.formats` 开启 `json`（Agent 消费的关键）
  - `server.limiter: false`（私有实例，避免自家 Agent 被 bot 检测拦截）
  - `engines` 禁用 Google（数据中心 IP 易触发 captcha），保留 Bing/DDG/Brave/Startpage

> ⚠️ `settings.yml` 里的 `secret_key` 是本地开发密钥。任何对外部署务必重新生成：
> `openssl rand -hex 32`

## 常用运维

```bash
docker compose -f searxng/docker-compose.yml ps        # 状态
docker compose -f searxng/docker-compose.yml logs -f   # 日志
docker compose -f searxng/docker-compose.yml restart   # 重启
docker compose -f searxng/docker-compose.yml down       # 停止
docker compose -f searxng/docker-compose.yml pull && \
  docker compose -f searxng/docker-compose.yml up -d    # 升级镜像
```

## 已知特性 / 调优方向

- 不同查询命中的引擎会不同（部分引擎偶发限流），多源聚合本身即为容错。
  若某引擎长期不返回，可在 `settings.yml` 的 `engines` 里调整启用集合。
- 真要扩到高频/多 IP 场景，可加代理池 —— 当前低频基准无需。

## 后续集成路线（见飞书文档 四章 Tier 2）

本目录只完成了**搜索后端**。接入 Agent 还需：

1. **SearchProvider 抽象** — 定义统一协议，`SearXNGProvider` 包装本 API
2. **SearchRouter** — Tavily / Exa / SearXNG / DDG 有序降级 + 健康检查 + per-provider metrics
3. **增强层** — 复用现有 `web_fetch`（Jina→trafilatura）抓 top-k 正文，对齐 Tavily 输出
4. **验收** — 用自有 GAIA Eval Harness 在联网题子集上对比检索质量
