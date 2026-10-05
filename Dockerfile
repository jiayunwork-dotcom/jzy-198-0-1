FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    BLAST_DB_PATH=/data/blast.db

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# 数据放在挂载卷
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

# 只暴露 HTTP 接口
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
