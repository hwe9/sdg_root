FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    libxml2-dev \
    libxslt-dev \
    libffi-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN  pip install --no-cache-dir --upgrade pip setuptools wheel -r requirements.txt

COPY . .

RUN useradd -m -u 1000 contentuser && chown -R contentuser:contentuser /app
USER contentuser

EXPOSE 8004

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
