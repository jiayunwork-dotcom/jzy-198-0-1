FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BLAST_DB=/data/blast.db

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# 数据库文件放在挂载卷上
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8080

# 只暴露 HTTP 接口
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
