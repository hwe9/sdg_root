FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app


RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir bcrypt>=4.0.0 PyJWT[crypto]>=2.8.0


COPY constraints/constraints.txt /tmp/constraints.txt
COPY src/api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && (pip install --no-cache-dir --require-hashes -c /tmp/constraints.txt -r /tmp/requirements.txt \
     || pip install --no-cache-dir -r /tmp/requirements.txt)     

COPY scripts/create_app_user.sh /usr/local/bin/create_app_user.sh
RUN chmod +x /usr/local/bin/create_app_user.sh \
 && /usr/local/bin/create_app_user.sh apiuser api_group 1000 1000 /app

 COPY . /app

 EXPOSE 8000


HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost:8000/live || exit 1

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
