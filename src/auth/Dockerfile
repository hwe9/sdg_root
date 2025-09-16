# src/auth/Dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (auth + core integration)
# - DB deps (sqlalchemy, psycopg2-binary) for dependency-managed DB init/health
# - Security deps (bcrypt, PyJWT[crypto], cryptography, passlib, keyring)
# - Web deps (fastapi, uvicorn[standard], python-multipart, slowapi, httpx)
# - Metrics used by dependency_manager (prometheus-client)
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    "uvicorn[standard]==0.24.0" \
    bcrypt>=4.0.0 \
    "PyJWT[crypto]>=2.8.0" \
    cryptography>=41.0.0 \
    passlib[bcrypt]==1.7.4 \
    python-multipart==0.0.6 \
    slowapi==0.1.9 \
    pydantic[email]>=2.0.0 \
    httpx>=0.26.0,<0.29.0 \
    prometheus-client>=0.20.0 \
    sqlalchemy==2.0.23 \
    psycopg2-binary==2.9.7 \
    keyring>=24.0.0 \
    schedule>=1.2.0

COPY auth/ /app/src/auth
COPY core/ /app/src/core

RUN useradd -m -u 1000 authuser && chown -R authuser:authuser /app
USER authuser

EXPOSE 8005

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

CMD ["python", "-m", "uvicorn", "src.auth.main:app", "--host", "0.0.0.0", "--port", "8005", "--reload"]
