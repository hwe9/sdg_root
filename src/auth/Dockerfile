# src/auth/Dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY auth/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt


COPY auth/ /app/src/auth
COPY core/ /app/src/core

RUN useradd -m -u 1000 authuser && chown -R authuser:authuser /app
USER authuser

EXPOSE 8005

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

CMD ["python", "-m", "uvicorn", "src.auth.main:app", "--host", "0.0.0.0", "--port", "8005"]
