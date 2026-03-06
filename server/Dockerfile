# Use the host's native platform (arm64 on Apple Silicon, amd64 on VPS).
# Override with --platform=linux/amd64 when building for x86 VPS deployment.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure the data directory exists inside the container
# (it will be overridden by the volume mount in docker-compose)
RUN mkdir -p data/examples data/models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
