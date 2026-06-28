FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TFM_SERVER_DATA=/data/server_data

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY docs ./docs
COPY herramientas ./herramientas

RUN mkdir -p /data/server_data && chown -R app:app /app /data

USER app

EXPOSE 8055

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8055"]
