# ============================================================
# Hybrid RAG - Docker Image
# ============================================================

FROM python:3.14.7-slim-bookworm

# ============================================================
# Environment
# ============================================================

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend \
    HF_HOME=/app/.cache/huggingface

# ============================================================
# System packages
# ============================================================

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Application directory
# ============================================================

WORKDIR /app

# ============================================================
# Python dependencies
# ============================================================

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

# ============================================================
# Backend and frontend source
# ============================================================

COPY backend ./backend
COPY frontend ./frontend

# ============================================================
# Runtime directories
# ============================================================

RUN mkdir -p \
        /app/data/uploads \
        /app/data/processed \
        /app/data/vector_store \
        /app/data/evaluation \
        /app/.cache/huggingface

# ============================================================
# Application ports
# ============================================================

EXPOSE 8001
EXPOSE 8501

# ============================================================
# Default command
# ============================================================

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]