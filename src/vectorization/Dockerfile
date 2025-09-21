FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY constraints/constraints.txt /tmp/constraints.txt
COPY src/vectorization/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && (pip install --no-cache-dir --require-hashes -c /tmp/constraints.txt -r /tmp/requirements.txt \
     || pip install --no-cache-dir -r /tmp/requirements.txt)

RUN useradd -m -u 1000 vectoruser && chown -R vectoruser:vectoruser /app
USER vectoruser

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
