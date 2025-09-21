# sdg_root/src/data_processing/Dockerfile
FROM python:3.11-slim

# Installiere ffmpeg und andere notwendige Tools
RUN apt-get update && apt-get install -y ffmpeg python3 python3-pip

WORKDIR /app

COPY constraints/constraints.txt /tmp/constraints.txt
COPY src/data_processing/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && (pip install --no-cache-dir --require-hashes -c /tmp/constraints.txt -r /tmp/requirements.txt \
     || pip install --no-cache-dir -r /tmp/requirements.txt) 

COPY . .

CMD ["python3", "main.py"]