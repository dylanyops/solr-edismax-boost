FROM python:3.11-slim

LABEL org.opencontainers.image.source=https://github.com/dylanyops/solr-edismax-boost

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "src.pipeline"]