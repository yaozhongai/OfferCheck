.PHONY: help all backend frontend vlm stop install clean

help:
	@echo "Nexa Agent V0"
	@echo ""
	@echo "用法: make <target>"
	@echo ""
	@echo "启动:"
	@echo "  all        一键启动 VLM + 后端 + 前端"
	@echo "  backend    启动 VLM + 后端 (:8000)"
	@echo "  frontend   仅启动前端 (:8501)"
	@echo "  vlm        仅启动 llama.cpp VLM (:8080)"
	@echo ""
	@echo "管理:"
	@echo "  install    安装 Python 依赖"
	@echo "  stop       停止所有进程"
	@echo "  clean      清理 data/ logs/ __pycache__"
	@echo ""
	@echo "默认 target: make = make all"

all:
	@echo "启动 Nexa Agent..."
	@echo "  VLM  → http://localhost:8080"
	@echo "  后端 → http://localhost:8000/docs"
	@echo "  前端 → http://localhost:8501"
	@echo ""
	@trap 'kill 0' EXIT; \
		bash scripts/llamacpp_server_minicpm-v4_6.sh & \
		sleep 3 && \
		python -m app.main & \
		sleep 2 && streamlit run app/streamlit_app.py & \
		wait

# 后端 + VLM
backend:
	@echo "启动 VLM + 后端..."
	@trap 'kill 0' EXIT; \
		bash scripts/llamacpp_server_minicpm-v4_6.sh & \
		sleep 3 && \
		python -m app.main

# 仅 VLM
vlm:
	bash scripts/llamacpp_server_minicpm-v4_6.sh

# 仅前端
frontend:
	streamlit run app/streamlit_app.py

# 安装依赖
install:
	pip install -r requirements.txt

# 停止
stop:
	@lsof -ti:8080 | xargs kill -9 2>/dev/null || true
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	@lsof -ti:8501 | xargs kill -9 2>/dev/null || true
	@echo "已停止 :8080 :8000 :8501"

# 清理
clean:
	rm -rf data/ logs/ __pycache__ app/__pycache__ app/*/__pycache__
	@echo "已清理 data/ logs/ __pycache__"
