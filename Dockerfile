# Multi-stage build: dev -> runtime
FROM python:3.11-slim AS builder

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY configs/ ./configs/

RUN pip install --no-cache-dir -e ".[ml,onnx]"

FROM python:3.11-slim AS runtime

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app /app

ENV PYTHONPATH=/app/src
ENTRYPOINT ["python", "-m", "ai_image_analyzer"]
CMD ["--help"]