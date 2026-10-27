# CUDA 12 + cuDNN 9 is what faster-whisper's CTranslate2 wheels expect.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends python3.11 python3.11-venv curl \
    && rm -rf /var/lib/apt/lists/*
RUN python3.11 -m venv /venv
ENV PATH=/venv/bin:$PATH PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir '.[gpu]'

EXPOSE 8765
HEALTHCHECK --interval=15s --timeout=3s CMD curl -fs localhost:8765/health/live || exit 1
CMD ["python", "-m", "app.main"]
