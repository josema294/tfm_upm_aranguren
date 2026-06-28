# ☁️ Servidor VPS (Puente y Dashboard)

Este módulo contiene el backend del sistema, desarrollado con **FastAPI** (Python). Su función principal es servir como "puente público" entre el Hardware (ESP32) que corre en la maqueta y el Servidor de Inferencia (Deep Learning) que corre localmente.

Este servidor está diseñado para ser extremadamente ligero y seguro, de forma que pueda desplegarse en un servidor virtual privado (VPS) económico sin necesidad de recursos pesados (sin GPU, sin PyTorch).

---

## 🎯 Responsabilidades del Servidor

**Lo que SÍ hace:**

- 📡 **Recibir Telemetría:** Expone un endpoint HTTP protegido por API Key para recibir lotes de muestras del ESP32 en tiempo real, o archivos CSV completos asíncronos.
- 🗄️ **Persistencia Ligera:** Guarda las sesiones, muestras, jobs (trabajos pendientes) y resultados directamente en el sistema de archivos local (sin bases de datos pesadas).
- 🧑‍💻 **Dashboard Web:** Sirve una interfaz gráfica interactiva protegida con usuario y contraseña para monitorizar el tren y los análisis de anomalías.
- 📄 **Documentación del proyecto:** Expone desde el panel la documentación básica de la API y la descarga de la memoria del TFM incluida en `docs/memoria_tfm.pdf`.
- 🚦 **Coordinar Tareas:** Gestiona las colas de trabajos (*jobs*) y atiende a los "Workers" (el Servidor de Inferencia) que solicitan datos para procesar.

**Lo que NO hace:**

- 🧠 **No ejecuta modelos matemáticos:** No realiza inferencia ni entrena el Autoencoder. Simplemente almacena los datos y espera a que el *Worker* externo envíe los resultados analizados.

---

## 🐳 Despliegue con Docker (Recomendado - Buenas Prácticas MLOps)

La forma oficial de ejecutar este servidor (garantizando total reproducibilidad e independencia del entorno) es mediante contenedores Docker. El repositorio incluye un `Dockerfile` muy ligero basado en `python:3.12-slim` que se encarga de aislar todas las dependencias.

### 1. Variables de Entorno

Antes de construir el contenedor, el servidor necesita ciertas variables para garantizar su seguridad (usuarios, firmas y límites). El archivo `.env.example` es solo una plantilla: los valores `replace_with...` no son validos y el servidor no arrancara con ellos.

Para generar los valores necesarios:

```bash
cd servidor_vps
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
.venv/bin/python -m app.auth --hash-password "cambia-esta-clave"
.venv/bin/python -m app.auth --random-secret
.venv/bin/python -m app.auth --random-secret
```

Copia las tres salidas en `.env`:

```env
TFM_SERVER_DATA=/data/server_data
TFM_WEB_USERNAME=admin
TFM_WEB_PASSWORD_HASH=<hash_seguro>
TFM_SESSION_SECRET=<secreto_aleatorio>
TFM_API_KEYS=<tu_api_key_para_el_esp32>
TFM_SECURE_COOKIE=true
TFM_SESSION_TTL_SECONDS=14400
TFM_MAX_UPLOAD_BYTES=10485760
TFM_MAX_CSV_ROWS=120000
```

Para pruebas locales sin HTTPS usa `TFM_SECURE_COOKIE=false`. En un despliegue real con HTTPS debe mantenerse `true`.

### 2. Construcción y Ejecución (Docker)

Construye la imagen y levanta el contenedor exponiendo el puerto `8055`. 

El Dockerfile copia la carpeta `docs/` dentro de la imagen, por lo que `docs/server_api.md` y `docs/memoria_tfm.pdf` quedan disponibles desde el panel web sin montar un volumen adicional. El volumen persistente se reserva para sesiones, CSV y resultados generados en ejecución.

**⚠️ CRÍTICO: Volúmenes Persistentes**
Dado que la persistencia se realiza en archivos planos, **es obligatorio montar un volumen** para que las sesiones del tren no se borren si el contenedor se reinicia. La ruta interna `/data/server_data` debe coincidir con `TFM_SERVER_DATA`.

```bash
cd servidor_vps
docker compose up -d --build
```

*(El Dashboard estará disponible en el puerto 8055 de tu servidor).*

---

## 🛡️ Seguridad en la Nube

Al estar expuesto a Internet, este servidor incorpora varias capas defensivas desde el código:

1. **API Keys:** Todo dato entrante vía REST API requiere una cabecera de autenticación, protegiéndote de spam de telemetría falso.
2. **Login Web:** Protección estricta con limitación de intentos (`TFM_LOGIN_MAX_ATTEMPTS`).
3. **Cabeceras HTTP Defensivas:** Implementa nativamente las mitigaciones `HSTS`, `CSP`, `X-Frame-Options` y `nosniff`.

*(Recomendación: Al desplegar esto en la nube pública, es buena práctica colocarlo detrás de un proxy reverso o firewall perimetral y añadir una regla de Rate Limiting para la ruta `/login` contra fuerza bruta).*

---

## 💻 Ejecución Rápida sin Docker (Opcional / Desarrollo)

Si quieres probar el servidor rápidamente en tu ordenador sin necesidad de levantar contenedores, o si deseas editar el código y ver los cambios en directo, puedes usar el enfoque tradicional de entorno virtual.

### 1. Entorno y Configuración

```bash
cd servidor_vps
python3 -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Antes de arrancar, sustituye en `.env` los valores `TFM_WEB_PASSWORD_HASH`, `TFM_SESSION_SECRET` y `TFM_API_KEYS` por valores generados con `python -m app.auth`, como se explica en la seccion de variables de entorno.

### 2. Arrancar Servidor Local

Lanza el servidor localmente con Uvicorn (usando `--reload` para modo desarrollador):

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8055 --reload
```

---

⬅️ [Volver al README Principal](../README.md)
