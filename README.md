# Monitorización Automatizada del Estado de Sistemas Ferroviarios mediante Deep Learning

![Estado](https://img.shields.io/badge/Estado-Prototipo%20y%20Validaci%C3%B3n-success)
![Python](https://img.shields.io/badge/Python-3.10--3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-ee4c2c)

Bienvenido al repositorio oficial del trabajo de fin de master de **Jose Maria Aranguren Palma** para el master de Deep learning de la **UPM**:

### **Monitorización Automatizada del Estado de Sistemas Ferroviarios mediante Deep Learning**.

Este proyecto propone y desarrolla un sistema completo (Hardware, Backend e Inteligencia Artificial) para detectar anomalías en la infraestructura ferroviaria.

Estando en su primeras fases, se usan maquetas ferroviarias para llevar a cabo el proyecto sin necesitar afrontar los incontables problemas que supondria tratar de llevar a la practica real con rutas y trenes reales, pero permitiendono la obtencion de conocimiento y conclusiones en un entorno controlado y de bajo coste.

Para ello, utiliza sensores inerciales de bajo coste (acelerómetro ADXL345) conectado a microcontroladores (ESP32) montados a bordo de la maqueta, para ir captando las vibraciones producidas por la vía en tiempo real y gracias al modelo de deep learning entrenado poder determinar si estas son las normales que se esperaría en la circulación, o tienen algún tipo de anomalía.

---

## 🏗️ Arquitectura del Sistema

El proyecto está diseñado de forma modular y desacoplada para garantizar escalabilidad y tolerancia a fallos de red. Se divide en tres pilares fundamentales, cada uno con su propia documentación detallada:

1. **[Firmware (Microcontrolador ESP32)](firmware/README.md)**: Código C++ para el ESP32. Se encarga de muestrear el acelerómetro a alta frecuencia (100 Hz), empaquetar los datos en ventanas temporales (buffers) y transmitirlos de forma robusta vía HTTP POST al servidor. Este subproyecto es necesario si quieres capturar tus propios datos con una maqueta de tren, o quieres usar una opcion que permite, en tiempo real, hacer una evaluacion de la circulacion del tren. 

2. **[Servidor VPS (Puente y Dashboard)](servidor_vps/README.md)**: Un backend ligero escrito en FastAPI. Actúa como puente recibiendo la telemetría del microcontrolador, guardando los datos en almacenamiento persistente y coordinando las colas de trabajo. Además, provee una interfaz web (Dashboard) para visualizar el estado del tren en tiempo real. Existe una instancia de demostración desplegada en `https://deteccionanomaliasferroviarias.jmaranguren.work/`; para solicitar una cuenta de evaluación puede escribirse a `filter-clay-banked@duck.com`. El proyecto también está pensado para poder ser autohosteado y levantar tu propia versión del servidor sin necesidad de recurrir a la infraestructura de terceros; es sumamente ligero y puede correr incluso en servidores virtuales de bajos recursos, ya que la carga pesada de inferencia se realiza desde la tercera parte del proyecto, el servidor de inferencia.

3. **[Servidor de Inferencia (Deep Learning)](servidor_inferencia/README.md)**: Esta tercera parte está pensada para usarse como worker, conectándose al servidor VPS. De este modo el servidor puede realizar tareas de inferencia, siendo principalmente estas jobs asíncronos, es decir, CSVs subidos con datos capturados a posteriori para su evaluación en colas. O incluso utilizarse en tiempo real en trabajos síncronos, junto con el microcontrolador y la maqueta, de modo que según el tren circula vaya enviando los datos en tiempo real al VPS y sean inmediatamente procesados por el servidor de inferencia. Es importante destacar que en este subproyecto también se encuentran los notebooks exploratorios así como los modelos finales, por lo que es también viable modificar el modelo final, usar una versión diferente, entrenar un modelo con tus propios datos de una maqueta y circuito diferente, de forma que pueda adaptarse a tus objetivos.

---

## 📂 Estructura del Repositorio

* 📁 `firmware/` - Código fuente para la adquisición de datos (ESP32).
  * 📁 `adxl345_wifi_http_live/` - Sketch C++ principal para la transmisión en vivo al VPS.
  * 📁 `adxl345_wifi_udp/` - Sketch C++ para capturas locales por UDP hacia un PC de la misma red.
  * 📁 `adxl345_serial/` - Sketch C++ secundario para captura de telemetría local.
  * 📁 `herramientas/` - Scripts de captura local en Python (Serial y UDP).
  * 📄 `README.md` - Documentación específica del hardware y flasheo.
* 📁 `servidor_vps/` - Backend web y almacenamiento de sesiones.
  * 📁 `app/` - Código fuente de la API desarrollada con FastAPI.
  * 📄 `Dockerfile` - Receta de despliegue para producción.
  * 📄 `README.md` - Documentación específica del servidor web.
* 📁 `servidor_inferencia/` - Modelos PyTorch, Worker de inferencia y experimentación.
  * 📁 `app/` - Código del *worker* y arquitectura de las redes neuronales.
  * 📁 `herramientas/` - Scripts de entrenamiento y evaluación.
  * 📁 `notebooks/` - Cuadernos de Jupyter con análisis de datos.
  * 📄 `Dockerfile` - Receta para ejecutar la inferencia aislada.
  * 📄 `README.md` - Documentación de modelos y ejecución de experimentos.
* 📁 `datos/` - Datos del proyecto.
  * 📁 `demo/` - Archivo CSV demostrativo (`sesion_demo.csv`) para pruebas rápidas.
  * 📁 `manifiestos/` - JSON con *hashes* para descargar los datasets reales.
  * 📁 `brutos/`, `etiquetas/`, `procesados/`, `analisis/`, `figuras/` - Carpetas locales ignoradas por Git donde se descargan o generan los datasets completos y resultados auxiliares.
* 📁 `artefactos/` - Pesos de los modelos entrenados.
  * 📄 `manifiesto.json` - JSON de control de versiones de los modelos (`.pth`), descargados localmente en `servidor_inferencia/models/`.
* 📁 `scripts/` - Utilidades generales del repositorio.
  * 📄 `descargar_datos.py` / `descargar_artefactos.py` - Scripts de descarga segura.

---

## 🚀 Guía de Inicio Rápido

El objetivo de esta guía es proporcionar una visión global del funcionamiento. Para que la **reproducibilidad sea total**, es indispensable crear **entornos virtuales** de Python en cada subproyecto. Esto fijará las versiones exactas de las librerías necesarias. Para detalles finos, consulta siempre los READMEs específicos de cada subproyecto (ver sección Documentación Detallada).

### Requisitos Previos

* **Docker** y **Docker Compose** (recomendado para desplegar el servidor VPS).
* **Python 3.10, 3.11 o 3.12** y `venv` (para ejecucion local del servidor de inferencia; Python 3.13/3.14 no se recomienda con los requisitos actuales de PyTorch).

### Opción A: Inferencia usando los modelos pre-entrenados (Recomendado si solo quieres ver como funciona el trabajo ya realizado como proyecto)

Si quieres levantar el sistema completo y probar cómo detecta anomalías.

No te preocupes si solo quieres probar el resultado y no tienes maqueta de trenes ni tiempo y ganas de montar el microcontrolador, en el repositorio incluimos un mini dataset demo que puedes usar para probar. 

Por defecto se descargararan los pesos pth del mejor modelo entrenado, de este modo incluyes al repositorio el cerebro para operar:

1. **Descargar los modelos (Artefactos)**
   Los archivos `.pth` pesados no se almacenan en el repositorio estandar de Git. Descárgalos usando nuestro script o desde la seccion de releases de Github:
   
   ```bash
   python3 scripts/descargar_artefactos.py
   python3 scripts/verificar_artefactos.py
   ```

2. **Levantar el Backend (VPS)**
   El modo optimo de trabajar como se ha diseñado el proyecto, no es unicamente corriendo escripts de python para ver resultados en una terminal, sino trabajando junto con el servidor vps que aporta la interfaz visual y forma de controlar el flujo de trabajo de forma sencilla. Como hemos mencionado hay una version deplegada, pero para autohostear tu propia version deberas entrar en la carpeta del VPS, configurar las variables y arrancar Docker.

   Primero genera valores reales para el login, el secreto de sesion y la API key. Los valores `replace_with...` del `.env.example` son placeholders y el servidor los rechazara:

   ```bash
   cd servidor_vps
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   cp .env.example .env
   .venv/bin/python -m app.auth --hash-password "cambia-esta-clave"
   .venv/bin/python -m app.auth --random-secret
   .venv/bin/python -m app.auth --random-secret
   ```

   Copia las tres salidas anteriores en `.env` como `TFM_WEB_PASSWORD_HASH`, `TFM_SESSION_SECRET` y `TFM_API_KEYS`. Para una prueba local por HTTP deja `TFM_SECURE_COOKIE=false`; en produccion con HTTPS debe usarse `true`.

   Despues arranca el servidor:
   
   ```bash
   docker compose up -d --build
   ```
   
   *El panel web interactivo estará en `http://localhost:8055`.* Los detalles e instrucciones concretas del servidor vps se encuentran en su **[README](servidor_vps/README.md)**.

3. **Arrancar el Servidor de Inferencia**
   Crea un `.env` para el worker usando la misma API key configurada en `TFM_API_KEYS` del VPS. El worker usa PyTorch CPU, por lo que no necesita GPU para ejecutar los modelos pre-entrenados:
   
   ```bash
   cd servidor_inferencia
   cp .env.example .env
   # Edita VPS_API_KEY con la misma clave usada en el VPS.
   docker compose up -d --build
   ```

4. **Probar con Datos de Demostración**
   El repositorio incluye una mini demostración de datos, dicho CSV es una versión recortada de los datos obtenidos para el proyecto (`datos/demo/sesion_demo.csv`). Estos datos tienen partes normales y partes con anomalías para poder probar ambas situaciones.

   Con el VPS levantado, puedes subir el CSV demo y crear automaticamente un job de inferencia:

   ```bash
   export TFM_API_KEY="pega_aqui_la_misma_api_key"
   curl -H "X-API-Key: $TFM_API_KEY" \
     -F "file=@datos/demo/sesion_demo.csv;type=text/csv" \
     http://127.0.0.1:8055/api/v1/sessions/sesion_demo/csv
   ```

   El worker recogera el job, ejecutara el detector hibrido y publicara los resultados en el dashboard del VPS.

5. **Descargar Datasets Completos (Opcional)**
   Si quieres tener acceso a toda la telemetría capturada por la maqueta durante las sesiones de prueba intensivas, ejecuta el script de descarga de datos. Esto los bajará dentro de `datos/`, junto al CSV demo y los manifiestos. Git solo versiona `datos/demo/` y `datos/manifiestos/`; los CSV completos descargados desde la release quedan ignorados.
   
   ```bash
   python3 scripts/descargar_datos.py
   ```

---

### Opción B: Entrenar modelos con tus propios datos (Funcional)

Si has capturado tus propios datos con el hardware ( Si dispones del hardware necesario y estas interesados en como hacerlo, consulta [**firmware**](firmware/README.md) ) y deseas entrenar la red:

1. **Preparación**: Sitúa tus archivos CSV en `datos/brutos/` o indica la ruta concreta al ejecutar cada script.
2. **Entorno Virtual**: Asegúrate de tener activado el entorno virtual del `servidor_inferencia` (`source .venv/bin/activate`).
3. **Entrenamiento Directo**: Utiliza los scripts funcionales de `servidor_inferencia/herramientas/` y los módulos de apoyo en `servidor_inferencia/entrenamiento/`, pasándoles tu CSV para generar de forma rápida e iterativa los modelos óptimos.
4. **Despliegue**: Copia los archivos `.pth` resultantes y actualiza tu `.env` para que el *worker* comience a usarlos.

---

## 📖 Documentación Detallada de los Subproyectos

Este README solo rasca la superficie del flujo de trabajo. Todo el peso de la configuración de puertos, esquemas de bases de datos, pinout físico y variables de entorno está fuertemente documentado en los READMEs de cada módulo. ¡Es imprescindible leerlos!

* ➡️ **[Guía de Configuración del Backend y Dashboard (Servidor VPS)](servidor_vps/README.md)**
* ➡️ **[Guía del Worker de Inferencia y Modelos (Servidor de Inferencia)](servidor_inferencia/README.md)**
* ➡️ **[Guía de Hardware, Flasheo y Pinout (Firmware)](firmware/README.md)**
