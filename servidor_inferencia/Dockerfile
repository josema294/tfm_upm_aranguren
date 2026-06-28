FROM python:3.12-slim

# Evitar escritura de bytecodes y buffering para logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias del runtime ligero (CPU)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código del worker y configuración
COPY app ./app
COPY configuracion ./configuracion
COPY herramientas ./herramientas

# El CMD por defecto lanza el worker en modo daemon
CMD ["python", "-m", "app.worker", "--daemon", "--worker-id", "docker-worker-01"]
