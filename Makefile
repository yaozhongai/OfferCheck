.PHONY: help agent backend vlm stop install test clean

help:
	@echo "Nexa Agent — core+app 分层架构"
	@echo ""
	@echo "用法: make <target>"
	@echo ""
	@echo "核心引擎 (nexa_agent, headless):"
	@echo "  agent Q='...'   跑一次自主调查 (可加 STAGE=stage1|stage4)"
	@echo "  test            跑核心/搜索单测"
	@echo ""
	@echo "后端 (server, FastAPI):"
	@echo "  backend         启动 server (:8000) —— 注: chat/upload 待接入核心(501)"
	@echo "  vlm             仅启动 llama.cpp VLM (:8080)"
	@echo ""
	@echo "管理: install / stop / clean"
	@echo ""
	@echo "前端: web/ (Next.js) 待建，见开发计划 7/5"

# 核心引擎：一次自主调查（场景无关；可选 STAGE 加载 OfferCheck 阶段任务定义）
# 例: make agent Q='帮我证伪这个 offer...' STAGE=stage4
agent:
	python -m nexa_agent.reflexion_agent "$(Q)" $(if $(STAGE),--stage $(STAGE),)

# 后端 + VLM
backend:
	@echo "启动 VLM + 后端 (server.main:app @ :8000)..."
	@trap 'kill 0' EXIT; \
		bash scripts/llamacpp_server_minicpm-v4_6.sh & \
		sleep 3 && \
		python -m server.main

# 仅 VLM
vlm:
	bash scripts/llamacpp_server_minicpm-v4_6.sh

# 安装依赖
install:
	pip install -r requirements.txt

# 单元测试
test:
	python -m pytest tests/ -q

# 停止
stop:
	@lsof -ti:8080 | xargs kill -9 2>/dev/null || true
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	@echo "已停止 :8080 :8000"

# 清理
clean:
	rm -rf data/ logs/ __pycache__ */__pycache__ */*/__pycache__
	@echo "已清理 data/ logs/ __pycache__"
