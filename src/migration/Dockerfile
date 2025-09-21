FROM python:3.11-slim

WORKDIR /app

COPY constraints/constraints.txt /tmp/constraints.txt
COPY src/migration/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && (pip install --no-cache-dir --require-hashes -c /tmp/constraints.txt -r /tmp/requirements.txt \
     || pip install --no-cache-dir -r /tmp/requirements.txt)

CMD ["python", "migration_plan.py"]