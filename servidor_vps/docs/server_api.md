# TFM Server API

Servidor ligero para actuar como puente entre el ESP32, el PC de inferencia y el dashboard.

## Arranque Local

```bash
cd vps_server
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8055
```

Tambien hay dos perfiles preparados:

```bash
# Local/preproduccion, usuario admin/admin.
TFM_ENV_FILE=.env_pre .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8055

# Produccion, exige reemplazar secretos CHANGE_ME.
TFM_ENV_FILE=.env_pro .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8055
```

Generar password hash y secretos:

```bash
.venv/bin/python -m app.auth --hash-password 'contrasena-fuerte'
.venv/bin/python -m app.auth --random-secret
```

Configurar en `.env`:

- `TFM_WEB_USERNAME`
- `TFM_WEB_PASSWORD_HASH`
- `TFM_SESSION_SECRET`
- `TFM_API_KEYS`
- `TFM_MAX_UPLOAD_BYTES`
- `TFM_MAX_CSV_ROWS`

En produccion con HTTPS usar `TFM_SECURE_COOKIE=true`.

## Interfaz

```text
http://127.0.0.1:8055/
```

Toda la UI y la API requieren autenticacion.

## Autenticacion

### Login Web

```text
GET /login
POST /login
POST /logout
```

El login crea una cookie `HttpOnly` firmada. La contrasena no se guarda en claro: se valida contra `TFM_WEB_PASSWORD_HASH`.

### Clientes Maquina

ESP32 y worker de inferencia deben enviar una API key:

```http
X-API-Key: <clave>
```

Tambien se acepta:

```http
Authorization: Bearer <clave>
```

Las claves validas se configuran en `TFM_API_KEYS`, separadas por coma.

### Limites De Subida

FastAPI aplica:

- `TFM_MAX_UPLOAD_BYTES`: tamano maximo del CSV.
- `TFM_MAX_CSV_ROWS`: numero maximo de filas.

El reverse proxy debe aplicar tambien limite de body. Ejemplo Nginx:

```nginx
client_max_body_size 20M;
```

## Endpoints

### Healthcheck

```http
GET /health
```

### Recepcion De Lotes En Vivo

```http
POST /api/v1/samples/batch
```

Ejemplo:

```json
{
  "device_id": "train_esp32_01",
  "session_id": "normal_001",
  "seq_start": 1000,
  "sample_rate_hz": 100,
  "samples": [
    {"timestamp_ms": 123450, "acc_x_g": 0.12, "acc_y_g": -0.03, "acc_z_g": 0.98},
    {"timestamp_ms": 123460, "acc_x_g": 0.13, "acc_y_g": -0.02, "acc_z_g": 0.97}
  ]
}
```

### Subida Asincrona De CSV

```http
POST /api/v1/sessions/{session_id}/csv
```

Al subir un CSV el VPS encola automaticamente un trabajo `full_session_inference`. Si el PC de inferencia esta ejecutando el worker en modo daemon, reclamara ese trabajo y publicara los resultados sin ejecutar comandos adicionales.

El CSV puede subirse en formato interno TFM:

```csv
seq,timestamp_ms,acc_x_g,acc_y_g,acc_z_g
```

O en formato sintético MLOps, que el servidor convierte automáticamente:

```csv
timestamp,accel_x,accel_y,accel_z
```

### Listar Sesiones

```http
GET /api/v1/sessions
```

### Descargar CSV De Una Sesion

```http
GET /api/v1/sessions/{session_id}/csv
```

### Consumir Muestras Incrementalmente

```http
GET /api/v1/sessions/{session_id}/samples?after_seq=1200&limit=500
```

Este endpoint esta pensado para el PC de inferencia. El PC recuerda el ultimo `seq` procesado y solicita solo datos nuevos.

### Subir Resultado De Inferencia

```http
POST /api/v1/inference/results
```

Ejemplo:

```json
{
  "session_id": "normal_001",
  "source": "pc_inference",
  "window_start_ms": 123000,
  "window_end_ms": 124000,
  "status": "normal",
  "reconstruction_error": 0.012,
  "quality": {
    "samples_expected": 100,
    "samples_received": 100,
    "packet_loss": 0
  }
}
```

### Leer Resultados De Inferencia

```http
GET /api/v1/sessions/{session_id}/results?limit=100
```

### Resumen De Inferencia

```http
GET /api/v1/sessions/{session_id}/summary
```

Devuelve conteos por estado, ratio de anomalia, score maximo, error maximo, ultimo resultado y ventana con mayor score.

### Crear Job Manual De Inferencia

```http
POST /api/v1/sessions/{session_id}/jobs
```

Sirve para relanzar inferencia de una sesion ya subida. Por defecto borra resultados anteriores de esa sesion antes de encolar el nuevo trabajo (`clear_results=true`).

### Registrar Worker De Inferencia

```http
POST /api/v1/workers/heartbeat
```

Ejemplo:

```json
{
  "worker_id": "pc-rocm-casa",
  "capabilities": ["autoencoder", "full_session_inference"],
  "current_job_id": null
}
```

### Listar Workers

```http
GET /api/v1/workers
```

### Cola De Jobs

```http
GET /api/v1/inference/jobs
GET /api/v1/inference/jobs?status=pending
POST /api/v1/inference/jobs/next
POST /api/v1/inference/jobs/{job_id}/complete
POST /api/v1/inference/jobs/{job_id}/fail
```

Estos endpoints los usa el worker del PC de inferencia. El flujo normal es: heartbeat, reclamar job pendiente, procesar toda la sesion, marcar completed o failed.

## Worker De Inferencia

Modo manual para pruebas:

```bash
cd inference_server
python -m app.worker --session-id csv_test_001 --drain
```

Modo daemon para demo:

```bash
cd inference_server
export VPS_API_KEY='<clave configurada en TFM_API_KEYS>'
python -m app.worker --daemon --worker-id pc-inferencia
```

En modo daemon el PC queda conectado al VPS desde dentro hacia fuera, sin abrir puertos locales. Cuando se sube un CSV al VPS, el job queda pendiente y el worker lo procesa automaticamente.

## Arquitectura Prevista

```text
ESP32 -> HTTP POST por lotes -> VPS FastAPI
PC inferencia -> heartbeat/claim jobs -> VPS FastAPI
PC inferencia -> GET muestras -> autoencoder -> POST resultados
Dashboard -> lee resultados del VPS
```
