FROM python:3.11-slim

LABEL org.opencontainers.image.source=https://github.com/dylanyops/solr-edismax-boost

WORKDIR /app

# Install system dependencies (Feast often needs these)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /mnt/data
RUN chmod -R 777 /mnt/data

# Copy application code
COPY src /app/src
COPY configs /app/configs
COPY data /app/data

# Copy Feast repository
COPY feature_repo /mnt/data/feature_repo

# Create runtime data directory (will be overridden by PVC in Argo)
RUN mkdir -p /mnt/data

# Environment variables
ENV APP_BASE_DIR=/app
ENV FEAST_REPO_PATH=/mnt/data/feature_repo

# Default command (not used by Argo but good for local testing)
CMD ["python", "-m", "src.pipeline"]