# 🚂 Firmware del Sistema de Captura (ESP32)

Este módulo contiene todo el código C++ necesario para programar el microcontrolador (ESP32). Su objetivo es interactuar con el acelerómetro triaxial (ADXL345), recoger las vibraciones producidas por la vía a alta frecuencia (**100 Hz**) y enviar los datos de forma robusta.

Este subsistema forma la base física de mi TFM. Toda la inteligencia artificial del servidor depende de la calidad de los datos empíricos que captura este hardware a bordo de la maqueta.

---

## 🛠️ Requisitos de Hardware

- **Microcontrolador**: ESP32 (Genérico, ej. NodeMCU ESP-32S).
- **Sensor Inercial**: Acelerómetro ADXL345 (comunicación I2C).
- **Alimentación**: Batería portátil (tipo Li-Ion 18650 o PowerBank a 5V) para montaje a bordo de la maqueta.
- **Cables**: Conectores Dupont o soldadura directa (muy recomendada para evitar pérdidas de conexión por la propia vibración de la maqueta).

---

## 🔌 Esquema de Conexiones (Pinout I2C)

Las conexiones se han estandarizado mediante protocolo I2C. Para asegurar la fiabilidad, la inicialización I2C de este firmware fuerza los pines de configuración de dirección (CS y SDO).

| ADXL345 | ESP32 | Notas y Propósito |
| :--- | :--- | :--- |
| **VCC** | `3V3` | Alimentación del sensor a 3.3V. |
| **GND** | `GND` | Tierra común. |
| **SDA** | `GPIO 21` | Línea de datos I2C. |
| **SCL** | `GPIO 22` | Línea de reloj I2C. |
| **CS**  | `GPIO 4`  | El firmware lo fuerza a `HIGH` para habilitar el modo I2C. |
| **SDO** | `GPIO 17` | El firmware lo fuerza a `LOW` para fijar la dirección I2C alternativa. |

*Para la alimentación general, la batería debe conectarse al pin `VIN` (5V) y a `GND` del ESP32.*

---

## 📂 Sketches de Arduino Incluidos

Dentro de la carpeta `firmware/` encontrarás tres programas principales, evolucionados según las distintas fases del proyecto:

1. `adxl345_serial/`: Firmware de prueba por cable USB. Es útil para comprobar que el ESP32, el ADXL345 y el pinout I2C funcionan correctamente sin depender de WiFi, servidor VPS ni API keys. Imprime los valores por el puerto serie y puede usarse junto con `herramientas/capture_serial_csv.py` para generar CSV locales.
2. `adxl345_wifi_udp/`: Firmware de captura local por WiFi/UDP. Envía muestras a un PC de la misma red para guardarlas como CSV con `herramientas/receive_udp_csv.py`. Es útil para generar datasets de entrenamiento sin usar el VPS.
3. `adxl345_wifi_http_live/`: *(Producción)* El firmware definitivo. Utiliza un sistema de *doble buffer* para enviar lotes de telemetría por HTTP POST al Servidor VPS de forma segura.

---

## 🚀 Configuración y Flasheo (Modo Live)

El flujo de trabajo habitual utilizará el sketch `adxl345_wifi_http_live`. Para flashearlo en tu ESP32:

### 1. Obtener la API Key del Servidor VPS
Por razones de seguridad, el servidor VPS rechaza cualquier dato que no venga firmado. Esta API Key se configura en el subproyecto `servidor_vps/`. Concretamente, si vas al archivo `.env` del servidor VPS, verás una variable llamada `TFM_API_KEYS`. Debes copiar el valor que hayas definido ahí para insertarlo en el firmware.

### 2. Preparar el archivo de Configuración
Debes crear tu propio archivo `config.h` copiando la plantilla:

```bash
cd adxl345_wifi_http_live
cp config.example.h config.h
```

### 3. Editar `config.h`
Abre el archivo `config.h` y ajusta las constantes:

```cpp
const char *WIFI_SSID = "TU_RED_WIFI";
const char *WIFI_PASSWORD = "TU_CONTRASEÑA";
const char *VPS_BASE_URL = "http://<IP_DEL_VPS>:8055"; // o tu dominio HTTPS
const char *VPS_API_KEY = "la_api_key_del_vps"; // Obtenida en el paso 1
const char *DEVICE_ID = "esp32-maqueta";
const char *SESSION_ID = "sesion_prueba_01";
```

### 4. Compilar y Subir
Abre el archivo `adxl345_wifi_http_live.ino` con el **Arduino IDE** (asegúrate de tener instalada la placa ESP32 en el Gestor de Placas) y pulsa "Subir" estando conectado por USB.

---

## 🧰 Herramientas y Scripts (Captura Local)

Si no vas a usar el modo HTTP Live y quieres trabajar directamente con datos capturados en tu PC, el repositorio dispone de varios scripts en Python preparados en la carpeta `herramientas/`.

### Resumen de Herramientas

| Script | Propósito Principal |
| :--- | :--- |
| `receive_udp_csv.py` | Abre un puerto local para recibir telemetría UDP del ESP32 y la guarda en un CSV. Es una vía secundaria útil para capturar datos de entrenamiento en red local. |
| `capture_serial_csv.py` | Lee los datos crudos emitidos por el ESP32 a través del cable USB (Monitor Serie) y los guarda en un CSV. |
| `generate_arduino_config.py` | Convierte un archivo `.env` estándar en un header de C++ (`config.h`) listo para compilar el firmware UDP. |

> **Nota:** Todos los scripts tienen menú de ayuda accesible añadiendo la bandera `--help` (ej. `python herramientas/receive_udp_csv.py --help`). Recuerda activarte un entorno virtual (`.venv`) si te faltaran librerías.

### Recepción Inalámbrica UDP (Captura local)
La recepción UDP es otra forma de capturar datos reales desde la maqueta en una red local y guardarlos directamente como CSV. No es el flujo principal de despliegue del sistema completo, que usa `adxl345_wifi_http_live` contra el VPS, pero resulta útil para generar datos de entrenamiento sin pasar por el dashboard.
Asegúrate de que tu PC y el ESP32 están en la misma red WiFi.

Para usarlo, desde la carpeta `firmware/`, crea `adxl345_wifi_udp/config.h` a partir de su plantilla:

```bash
cd adxl345_wifi_udp
cp config.example.h config.h
```

Edita `DESTINATION_IP` con la IP local del PC que ejecutará el receptor UDP. Después abre `adxl345_wifi_udp.ino` en Arduino IDE y súbelo al ESP32.

```bash
cd ..
python herramientas/receive_udp_csv.py --port 5005 --output ../datos/brutos/mi_captura_udp.csv --seconds 60
```
*Este comando abrirá un servidor UDP en el puerto 5005 y guardará 60 segundos de vibraciones de la maqueta directamente en un CSV listos para el Servidor de Inferencia.*

### Captura por Cable USB (Serial)
Si prefieres no usar el WiFi y trabajar enchufado por cable USB, sube el sketch `adxl345_serial` y utiliza:

```bash
python herramientas/capture_serial_csv.py --port /dev/ttyUSB0 --baud 115200 --output ../datos/brutos/mi_captura_usb.csv --seconds 30
```

### Generación de Configuración (UDP)
Si prefieres no editar código C++ a mano (para evitar errores de sintaxis con las comillas o los puntos y coma) y quieres asegurarte de que tus credenciales WiFi no se suben por error a GitHub, puedes usar este asistente que convierte un archivo `.env` estándar en un archivo `config.h` de Arduino.

Simplemente crea un archivo llamado `.env` en la carpeta raíz de `firmware/` con este contenido:
```env
WIFI_SSID=TU_RED_WIFI
WIFI_PASSWORD=TU_CONTRASEÑA
DESTINATION_IP=192.168.1.x  # La IP local de tu PC donde corre receive_udp_csv.py
DESTINATION_PORT=5005       # (Opcional) Puerto UDP, por defecto 5005
WIFI_CHANNEL=0              # (Opcional) Canal WiFi, por defecto 0 (Auto)
```

Y luego ejecuta el script indicando el destino del `config.h` de tu firmware UDP compatible:

```bash
python herramientas/generate_arduino_config.py --env .env --output ruta/a/tu_firmware_udp/config.h
```

---

## 🧠 Decisiones de Arquitectura del Firmware

### Rango del Sensor (±8g)
Durante las pruebas experimentales del TFM, detecté que el rango por defecto de **±2g producía saturación** en la señal durante impactos bruscos del tren con tramos anómalos o descarrilamientos parciales, recortando los picos y ocultando la realidad dinámica. Por ello, este firmware inicializa explícitamente el ADXL345 en rango de **±8g**.

### Sampling a 100 Hz y Envío por Lotes (Batches)
El objetivo es registrar la vibración de las ruedas y la vía a una frecuencia constante de 100 muestras por segundo (100 Hz).
Si el ESP32 hiciese una petición HTTP por cada muestra individual en el modo Live, colapsaría la red WiFi y se paralizaría el procesador. 

**Solución implementada:**
1. Una tarea (en un núcleo) se encarga de rellenar buffers de memoria con 100 muestras exactas (1 segundo de datos).
2. Cuando el buffer se llena, se pasa a una tarea secundaria (en el otro núcleo).
3. Esta segunda tarea empaqueta las 100 muestras en un bloque JSON y hace un único `HTTP POST /api/v1/samples/batch` al Servidor VPS.
4. Si la red WiFi se cae temporalmente, el código prefiere **descartar lotes** a alterar el ritmo de muestreo, garantizando que los Autoencoders (AE/VAE) de Inteligencia Artificial no reciban ventanas con espacios temporales deformados.

---

⬅️ [Volver al README Principal](../README.md)
