FROM python:3.11-slim

WORKDIR /app

COPY src/migration/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /tmp/requirements.txt

COPY . .

CMD ["python", "src/migration/migration_plan.py"]