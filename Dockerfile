FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    DEPLOYMENT_PROFILE=public-demo \
    VISION_BACKEND=demo \
    REASONING_BACKEND=deterministic \
    FIRE_CLASSIFIER_ENABLED=false

WORKDIR /app

RUN addgroup --system visionguard && adduser --system --ingroup visionguard visionguard

COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --no-cache-dir .

USER visionguard
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT','8000') + '/api/v1/health', timeout=3)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
