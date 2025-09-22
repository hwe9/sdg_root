FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    postgresql-client \
    build-essential \
    libffi-dev \
    gosu \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY constraints/constraints.txt /tmp/constraints.txt
COPY src/auth/requirements.txt      /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && (pip install --no-cache-dir --require-hashes -c /tmp/constraints.txt -r /tmp/requirements.txt \
     || pip install --no-cache-dir -r /tmp/requirements.txt)

COPY . /app

RUN set -eux; \
    if ! getent group 1000 >/dev/null; then addgroup --system --gid 1000 authgroup; fi; \
    if ! getent passwd 1000 >/dev/null; then adduser  --system --uid 1000 --gid 1000 --home /app --disabled-login authuser; fi; \
    install -d -m 0770 -o authuser -g authgroup /app/secrets


COPY src/auth/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh && chmod 0755 /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/bin/sh","/usr/local/bin/entrypoint.sh"]


EXPOSE 8005
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8005/health || exit 1

CMD ["python", "-m", "uvicorn", "src.auth.main:app", "--host", "0.0.0.0", "--port", "8005"]
