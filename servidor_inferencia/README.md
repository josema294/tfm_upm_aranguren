# 🧠 Servidor de Inferencia (Deep Learning)

Este es el "cerebro" matemático del proyecto. Se ejecuta localmente en un PC y actúa en dos frentes radicalmente distintos:
1. **En Producción (Worker):** Funciona como un proceso ligero (*Daemon*) que se conecta constantemente al Servidor VPS de la nube, descarga los datos capturados, realiza la inferencia con PyTorch y devuelve los resultados.
2. **En Desarrollo (Entrenamiento y Data Science):** Ofrece un entorno contenerizado pesado para explorar los datos en *Jupyter Notebooks* y re-entrenar los modelos desde cero utilizando aceleración por hardware (GPU).

---

## 🏛️ Arquitectura del Detector (Híbrido)

El TFM utiliza un enfoque compuesto por dos detectores operando en paralelo sobre ventanas temporales (100 muestras = 1 segundo):

1. **Autoencoder Variacional Conv1D (VAE):** Detecta anomalías generales (descarrilamientos, saltos).
2. **Red Convolucional MIL:** Detecta micro-patrones muy sutiles de pérdida de adherencia (*slip*).

---

## 🚀 Despliegue en Producción (El Worker - Docker CPU)

Siguiendo las mejores prácticas de MLOps, el *Worker* de inferencia debe desplegarse mediante su contenedor Docker oficial (usando el `Dockerfile` de la raíz de este directorio).

**Decisión de Arquitectura Crítica (CPU vs GPU):**
Las pruebas experimentales del TFM demostraron que, para ventanas de datos unidimensionales tan pequeñas (100 muestras), el tiempo de mover los tensores de la RAM a la VRAM de la gráfica (*overhead*) era mayor que el tiempo de cómputo en sí. Por tanto, para obtener la mínima latencia en tiempo real y evitar cuellos de botella PCIe, **la inferencia en producción (Worker) se realiza por CPU** (con la variable `TORCH_DEVICE=cpu`), reservando la GPU exclusivamente para la fase de entrenamiento.

### 1. Variables de Entorno
Configura el archivo `.env` (puedes basarte en `.env.example`).
```env
VPS_BASE_URL=http://<IP_O_DOMINIO_DEL_VPS>:8055
VPS_API_KEY=<misma_clave_configurada_en_TFM_API_KEYS>
SESSION_ID=
TORCH_DEVICE=cpu
MODEL_PATH=models/vae_real_v6_window1s_derived.pth
SLIP_MODEL_PATH=models/slip_mil_w30_50_100_testperf_plus_validation_v2.pth
```

`VPS_API_KEY` debe coincidir con una de las claves configuradas en `TFM_API_KEYS` del servidor VPS. Si se deja el placeholder de `.env.example`, el worker no podra autenticarse.

### 2. Arrancar el Contenedor (Worker)
Como la imagen Docker del Worker no incluye los pesados modelos pre-entrenados `.pth` (para mantener la imagen ágil), deberás descargar los pesos (usando el script principal del repositorio) y mapear la carpeta como un volumen.

```bash
cd servidor_inferencia
docker compose up -d --build
```
*(Nota de desarrollo: También puedes arrancarlo localmente sin Docker para depuración rápida con Python 3.10, 3.11 o 3.12, instalando el `requirements.txt` en un `.venv` y ejecutando `python -m app.worker --daemon`).*

---

## 🔬 Modo Desarrollo y Exploración (Entornos Locales y Jupyter)

Para entrenar estos modelos desde cero, explorar el *Dataset* mediante Pandas o abrir los *Jupyter Notebooks*, **no** es indispensable utilizar contenedores pesados ni disponer de una tarjeta gráfica de alta gama.

Durante la experimentación empírica de este TFM se comprobó que, al trabajar con ventanas temporales de apenas 100 muestras unidimensionales, el coste de transferir los tensores desde la memoria RAM a la VRAM de la GPU superaba al propio cómputo. Por ello, la inmensa mayoría del entrenamiento y exploración final se llevó a cabo utilizando eficientemente la **CPU**.

Para este propósito, la vía más ágil y recomendada es ejecutar el código directamente en tu sistema operativo utilizando un entorno virtual.

### Opción A: Ejecución local rápida con .venv (Recomendado)
Para explorar los cuadernos interactivos o modificar las arquitecturas sin las capas de abstracción de Docker, basta con instalar las dependencias de *Data Science* nativamente:

```bash
cd servidor_inferencia
python3.10 -m venv .venv  # tambien valido con python3.11 o python3.12
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
jupyter lab
```
*(Esto abrirá el entorno de Jupyter en tu navegador con acceso total a los datos y scripts).*

El `requirements.txt` runtime instala `torch==2.7.1+cpu` desde el indice CPU de PyTorch para evitar descargar dependencias CUDA innecesarias. Con los requisitos actuales se recomienda usar Python 3.10-3.12; Python 3.13/3.14 puede no tener wheels compatibles para esta version de PyTorch. Si quieres entrenar con una GPU concreta, instala la variante de PyTorch correspondiente a tu entorno antes de ejecutar los scripts de entrenamiento.

### Opción B: Entorno Docker ROCm (Asunción inicial / Legado)
Bajo la premisa inicial de que el modelo exigiría aceleración masiva, y dado que el desarrollo se inició sobre una **AMD Radeon RX 6900 XT**, el repositorio conserva un archivo de orquestación preparado para el ecosistema **ROCm**:

```bash
docker compose -f docker-compose.rocm.yml up --build
```
*(Este contenedor levanta una imagen base muy pesada de AMD con Jupyter incorporado. PyTorch detectará la GPU, pero ten en cuenta que para los modelos de 1D de este proyecto, podrías no experimentar mejoras de tiempo frente a la ejecución nativa en CPU).*

---

## 📂 Modelos Pre-Entrenados (.pth)

Los pesos entrenados de PyTorch (`.pth`) no se versionan en Git. Para obtener los modelos oficiales del TFM, dirígete a la raíz del repositorio (`tfm-v2`) y ejecuta el script `python scripts/descargar_artefactos.py`. Esto colocará los archivos `.pth` necesarios en `servidor_inferencia/models/`, carpeta local ignorada por Git y montada por el contenedor del Worker como `/app/models`.

---

## 🧰 Herramientas de Inferencia (CLI)
La carpeta `herramientas/` incluye utilidades en Python diseñadas para auditar la calidad de los datos y monitorizar el sistema en tiempo real. 

### 1. Auditoría de Capturas CSV (`analyze_capture_csv.py`)
Esta herramienta audita un archivo CSV (como los generados por el modo UDP del ESP32) para evaluar la integridad física de la transmisión antes de enviarla a la Inteligencia Artificial.
*   **¿Qué hace?** Analiza saltos de secuencia, paquetes perdidos, latencia, y calcula si la tasa real de captura cumple el estándar de 100 Hz.
*   **Uso:** `python herramientas/analyze_capture_csv.py ../datos/brutos/mi_captura.csv`
*   *(Acepta banderas como `--expected-hz 100`, `--max-loss-ratio 0.01` para ser más estricto).*

### 2. Monitor de Flujo Live (`monitor_live_flow.py`)
Si estás ejecutando una sesión en directo (Live) con el ESP32 emitiendo hacia el VPS, y el Worker local absorbiendo esos datos, esta herramienta te permite ver en la consola si el cuello de botella está en la red o en la CPU.
*   **¿Qué hace?** Lee tu `.env`, se conecta a la API REST del VPS y pinta una tabla en tiempo real con los ratios: `esp_in/s` (paquetes recibidos del tren), `worker/s` (paquetes absorbidos por PyTorch) y `results/s` (ventanas analizadas).
*   **Uso:** `python herramientas/monitor_live_flow.py --interval 1`
*   *(Puedes apuntar a una sesión concreta con `--session-id tu_sesion_01`).*

---
⬅️ [Volver al README Principal](../README.md)
