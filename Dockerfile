FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf \
    PIP_NO_CACHE_DIR=1

# libs needed by opencv (pulled in by the layout model) and pymupdf
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first (keeps the image ~2GB smaller than the CUDA build)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install "glmocr[selfhosted]" pymupdf requests

COPY glm_worker.py /app/glm_worker.py

CMD ["python", "/app/glm_worker.py"]
