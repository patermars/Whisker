FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

RUN useradd --system --uid 10001 --create-home whisker \
    && mkdir -p /tmp/whisker \
    && chown -R whisker:whisker /tmp/whisker

ENV PORT=8080 \
    WHISKER_DEVICE=cpu \
    WHISKER_THREADS=1 \
    WHISKER_CHECKPOINT_PATH=/tmp/whisker/model.pt \
    WHISKER_COOKIE_SECURE=1 \
    WHISKER_REQUIRE_MODEL=1

EXPOSE 8080

USER whisker

CMD ["python", "-m", "src.container_start"]
