FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi>=0.110.0 \
    "uvicorn[standard]>=0.28.0" \
    python-multipart>=0.0.9 \
    requests>=2.31.0 \
    google-genai>=0.1.0 \
    edge-tts>=6.1.10 \
    pydantic>=2.0.0 \
    aiofiles>=23.2.1

COPY . /app/

ENV PYTHONPATH=/app
ENV PORT=5000
EXPOSE 5000

CMD ["sh", "-c", "python -m uvicorn server:app --host 0.0.0.0 --port ${PORT:-5000}"]
