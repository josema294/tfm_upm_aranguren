#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <Wire.h>

#include "config.h"

static const unsigned long WIFI_CONNECT_TIMEOUT_MS = 60000;
static const uint16_t MAX_BATCH_SIZE = 500;
static const uint8_t TX_QUEUE_DEPTH = 8;

static const uint8_t ADXL345_ADDR = 0x53;

static const uint8_t REG_DEVID = 0x00;
static const uint8_t REG_BW_RATE = 0x2C;
static const uint8_t REG_POWER_CTL = 0x2D;
static const uint8_t REG_DATA_FORMAT = 0x31;
static const uint8_t REG_DATAX0 = 0x32;

static const int SDA_PIN = 21;
static const int SCL_PIN = 22;
static const int ADXL345_CS_PIN = 4;
static const int ADXL345_SDO_PIN = 17;
static const unsigned long SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;

struct Sample {
  uint32_t timestamp_ms;
  float acc_x_g;
  float acc_y_g;
  float acc_z_g;
};

Sample active_batch[MAX_BATCH_SIZE];
Sample tx_batches[TX_QUEUE_DEPTH][MAX_BATCH_SIZE];

portMUX_TYPE batch_mux = portMUX_INITIALIZER_UNLOCKED;
TaskHandle_t sender_task_handle = nullptr;
WiFiClientSecure secure_client;

volatile uint16_t active_count = 0;
volatile uint16_t tx_counts[TX_QUEUE_DEPTH] = {0};
volatile uint32_t tx_seq_starts[TX_QUEUE_DEPTH] = {0};
volatile uint8_t tx_head = 0;
volatile uint8_t tx_tail = 0;
volatile uint8_t tx_queued = 0;
volatile uint32_t dropped_batches = 0;

uint32_t sequence_number = 0;
unsigned long next_sample_us = 0;
unsigned long last_metrics_ms = 0;
uint32_t batches_sent = 0;
uint32_t batches_failed = 0;
uint32_t last_post_ms = 0;
uint32_t last_build_ms = 0;
uint32_t tls_connects = 0;
uint32_t last_connect_ms = 0;
int last_status_code = 0;

uint16_t configuredBatchSize() {
  if (HTTP_BATCH_SIZE < 1) {
    return 1;
  }
  if (HTTP_BATCH_SIZE > MAX_BATCH_SIZE) {
    return MAX_BATCH_SIZE;
  }
  return HTTP_BATCH_SIZE;
}

void writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

uint8_t readRegister(uint8_t reg) {
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(ADXL345_ADDR, (uint8_t)1);
  return Wire.available() ? Wire.read() : 0;
}

bool readAccelerationRaw(int16_t &x, int16_t &y, int16_t &z) {
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(REG_DATAX0);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  if (Wire.requestFrom(ADXL345_ADDR, (uint8_t)6) != 6) {
    return false;
  }

  uint8_t x0 = Wire.read();
  uint8_t x1 = Wire.read();
  uint8_t y0 = Wire.read();
  uint8_t y1 = Wire.read();
  uint8_t z0 = Wire.read();
  uint8_t z1 = Wire.read();

  x = (int16_t)((x1 << 8) | x0);
  y = (int16_t)((y1 << 8) | y0);
  z = (int16_t)((z1 << 8) | z0);
  return true;
}

bool connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.disconnect(false);
  delay(1000);

  Serial.println("Escaneando redes WiFi visibles...");
  bool target_found = false;
  int network_count = WiFi.scanNetworks();
  for (int i = 0; i < network_count; i++) {
    Serial.print("  SSID=");
    Serial.print(WiFi.SSID(i));
    Serial.print(" RSSI=");
    Serial.print(WiFi.RSSI(i));
    Serial.print(" canal=");
    Serial.print(WiFi.channel(i));
    Serial.print(" cifrado=");
    Serial.println(WiFi.encryptionType(i));
    if (WiFi.SSID(i) == WIFI_SSID) {
      target_found = true;
    }
  }
  if (!target_found) {
    Serial.print("AVISO: no se ve el SSID configurado: ");
    Serial.println(WIFI_SSID);
  }

  Serial.println("Conectando a WiFi...");
  if (WIFI_CHANNEL > 0) {
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD, WIFI_CHANNEL);
  } else {
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  }

  unsigned long start_ms = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start_ms < WIFI_CONNECT_TIMEOUT_MS) {
    Serial.print(".");
    delay(500);
  }

  Serial.println();
  Serial.print("Status WiFi: ");
  Serial.println(WiFi.status());

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("ERROR: no se pudo conectar a WiFi.");
    return false;
  }

  Serial.print("IP ESP32: ");
  Serial.println(WiFi.localIP());
  Serial.print("RSSI: ");
  Serial.println(WiFi.RSSI());
  return true;
}

void appendJsonEscaped(String &json, const char *value) {
  json += "\"";
  for (const char *p = value; *p; ++p) {
    if (*p == '"' || *p == '\\') {
      json += "\\";
    }
    json += *p;
  }
  json += "\"";
}

String buildPayload(const Sample *samples, uint16_t count, uint32_t seq_start) {
  String json;
  json.reserve(180 + count * 82);
  json += "{\"device_id\":";
  appendJsonEscaped(json, DEVICE_ID);
  json += ",\"session_id\":";
  appendJsonEscaped(json, SESSION_ID);
  json += ",\"seq_start\":";
  json += seq_start;
  json += ",\"sample_rate_hz\":";
  json += SAMPLE_RATE_HZ;
  json += ",\"samples\":[";

  for (uint16_t i = 0; i < count; i++) {
    if (i > 0) {
      json += ",";
    }
    json += "{\"timestamp_ms\":";
    json += samples[i].timestamp_ms;
    json += ",\"acc_x_g\":";
    json += String(samples[i].acc_x_g, 5);
    json += ",\"acc_y_g\":";
    json += String(samples[i].acc_y_g, 5);
    json += ",\"acc_z_g\":";
    json += String(samples[i].acc_z_g, 5);
    json += "}";
  }

  json += "]}";
  return json;
}

bool ensureHttpsConnected() {
  if (WiFi.status() != WL_CONNECTED) {
    while (!connectWifi()) {
      delay(3000);
    }
  }

  if (secure_client.connected()) {
    return true;
  }

  secure_client.stop();
  secure_client.setTimeout(HTTPS_READ_TIMEOUT_MS);
  secure_client.setHandshakeTimeout(HTTPS_CONNECT_TIMEOUT_MS / 1000);
  if (HTTPS_INSECURE) {
    secure_client.setInsecure();
  }

  unsigned long connect_start_ms = millis();
  bool connected = secure_client.connect(VPS_HOST, VPS_PORT);
  last_connect_ms = millis() - connect_start_ms;

  if (!connected) {
    Serial.print("ERROR_TLS_CONNECT host=");
    Serial.print(VPS_HOST);
    Serial.print(" connect_ms=");
    Serial.println(last_connect_ms);
    return false;
  }

  tls_connects++;
  return true;
}

bool readLineWithTimeout(String &line, uint32_t timeout_ms) {
  line = "";
  unsigned long start_ms = millis();

  while (millis() - start_ms < timeout_ms) {
    while (secure_client.available()) {
      char c = secure_client.read();
      if (c == '\n') {
        line.trim();
        return true;
      }
      line += c;
    }

    if (!secure_client.connected() && !secure_client.available()) {
      line.trim();
      return line.length() > 0;
    }

    vTaskDelay(pdMS_TO_TICKS(1));
  }

  line.trim();
  return false;
}

bool readHttpResponse(int &status_code, bool &server_closes) {
  status_code = -1;
  server_closes = false;

  String status_line;
  if (!readLineWithTimeout(status_line, HTTPS_READ_TIMEOUT_MS)) {
    Serial.println("ERROR_HTTP_STATUS_TIMEOUT");
    return false;
  }
  if (!status_line.startsWith("HTTP/1.")) {
    Serial.print("ERROR_HTTP_STATUS_LINE=");
    Serial.println(status_line);
    return false;
  }

  int first_space = status_line.indexOf(' ');
  if (first_space < 0 || status_line.length() < first_space + 4) {
    Serial.print("ERROR_HTTP_STATUS_PARSE=");
    Serial.println(status_line);
    return false;
  }
  status_code = status_line.substring(first_space + 1, first_space + 4).toInt();

  int content_length = 0;
  while (secure_client.connected()) {
    String header;
    if (!readLineWithTimeout(header, HTTPS_READ_TIMEOUT_MS)) {
      Serial.println("ERROR_HTTP_HEADER_TIMEOUT");
      return false;
    }
    if (header.length() == 0) {
      break;
    }

    String lower = header;
    lower.toLowerCase();
    if (lower.startsWith("content-length:")) {
      content_length = lower.substring(15).toInt();
    } else if (lower.startsWith("connection:") && lower.indexOf("close") >= 0) {
      server_closes = true;
    }
  }

  unsigned long drain_start_ms = millis();
  int drained = 0;
  while (drained < content_length && millis() - drain_start_ms < HTTPS_READ_TIMEOUT_MS) {
    while (secure_client.available() && drained < content_length) {
      secure_client.read();
      drained++;
    }
    if (drained < content_length) {
      vTaskDelay(pdMS_TO_TICKS(1));
    }
  }

  if (drained < content_length) {
    Serial.println("ERROR_HTTP_BODY_TIMEOUT");
    return false;
  }

  return true;
}

bool postBatch(const Sample *samples, uint16_t count, uint32_t seq_start) {
  unsigned long build_start_ms = millis();
  String payload = buildPayload(samples, count, seq_start);
  last_build_ms = millis() - build_start_ms;

  if (!ensureHttpsConnected()) {
    batches_failed++;
    return false;
  }

  unsigned long post_start_ms = millis();
  secure_client.print("POST ");
  secure_client.print(VPS_BATCH_PATH);
  secure_client.print(" HTTP/1.1\r\n");
  secure_client.print("Host: ");
  secure_client.print(VPS_HOST);
  secure_client.print("\r\n");
  secure_client.print("Content-Type: application/json\r\n");
  secure_client.print("X-API-Key: ");
  secure_client.print(VPS_API_KEY);
  secure_client.print("\r\n");
  secure_client.print("Connection: keep-alive\r\n");
  secure_client.print("Content-Length: ");
  secure_client.print(payload.length());
  secure_client.print("\r\n\r\n");
  size_t written = secure_client.write((const uint8_t *)payload.c_str(), payload.length());
  if (written != payload.length()) {
    Serial.print("ERROR_HTTP_WRITE written=");
    Serial.print(written);
    Serial.print(" expected=");
    Serial.println(payload.length());
    secure_client.stop();
    batches_failed++;
    return false;
  }

  int status_code = -1;
  bool server_closes = false;
  bool response_ok = readHttpResponse(status_code, server_closes);
  last_post_ms = millis() - post_start_ms;
  last_status_code = status_code;

  if (server_closes || !response_ok) {
    secure_client.stop();
  }

  Serial.print("HTTP status=");
  Serial.print(status_code);
  Serial.print(" seq_start=");
  Serial.print(seq_start);
  Serial.print(" count=");
  Serial.print(count);
  Serial.print(" build_ms=");
  Serial.print(last_build_ms);
  Serial.print(" post_ms=");
  Serial.print(last_post_ms);
  Serial.print(" connect_ms=");
  Serial.print(last_connect_ms);
  Serial.print(" tls_connects=");
  Serial.print(tls_connects);
  Serial.print(" keepalive=");
  Serial.print(secure_client.connected() ? "yes" : "no");
  Serial.print(" rssi=");
  Serial.print(WiFi.RSSI());
  Serial.print(" dropped_batches=");
  Serial.println(dropped_batches);

  if (!response_ok || status_code < 200 || status_code >= 300) {
    batches_failed++;
    return false;
  }
  batches_sent++;
  return true;
}

void senderTask(void *parameter) {
  Sample local_batch[MAX_BATCH_SIZE];

  while (true) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    while (true) {
      uint16_t local_count = 0;
      uint32_t local_seq_start = 0;

      portENTER_CRITICAL(&batch_mux);
      if (tx_queued > 0) {
        uint8_t index = tx_head;
        local_count = tx_counts[index];
        local_seq_start = tx_seq_starts[index];
        for (uint16_t i = 0; i < local_count; i++) {
          local_batch[i] = tx_batches[index][i];
        }
        tx_head = (tx_head + 1) % TX_QUEUE_DEPTH;
        tx_queued--;
      }
      portEXIT_CRITICAL(&batch_mux);

      if (local_count == 0) {
        break;
      }

      postBatch(local_batch, local_count, local_seq_start);
      vTaskDelay(pdMS_TO_TICKS(1));
    }
  }
}

void queueBatchForSend(uint32_t seq_start, uint16_t count) {
  bool should_notify = false;

  portENTER_CRITICAL(&batch_mux);
  if (tx_queued < TX_QUEUE_DEPTH) {
    uint8_t index = tx_tail;
    for (uint16_t i = 0; i < count; i++) {
      tx_batches[index][i] = active_batch[i];
    }
    tx_seq_starts[index] = seq_start;
    tx_counts[index] = count;
    tx_tail = (tx_tail + 1) % TX_QUEUE_DEPTH;
    tx_queued++;
    should_notify = true;
  } else {
    dropped_batches++;
  }
  active_count = 0;
  portEXIT_CRITICAL(&batch_mux);

  if (should_notify && sender_task_handle != nullptr) {
    xTaskNotifyGive(sender_task_handle);
  }
}

void setupAdxl345() {
  pinMode(ADXL345_CS_PIN, OUTPUT);
  digitalWrite(ADXL345_CS_PIN, HIGH);
  pinMode(ADXL345_SDO_PIN, OUTPUT);
  digitalWrite(ADXL345_SDO_PIN, LOW);
  delay(10);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  uint8_t devid = readRegister(REG_DEVID);
  if (devid != 0xE5) {
    Serial.print("ERROR: ADXL345 no detectado. DEVID=0x");
    Serial.println(devid, HEX);
    Serial.println("Revisa VCC=3.3V, GND, SDA=21, SCL=22, CS=3.3V y SDO=GND.");
    while (true) {
      delay(1000);
    }
  }

  writeRegister(REG_POWER_CTL, 0x08);   // Measurement mode.
  writeRegister(REG_DATA_FORMAT, 0x0A); // Full resolution, +/-8g.
  writeRegister(REG_BW_RATE, 0x0A);     // 100 Hz output data rate.
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  setupAdxl345();

  while (!connectWifi()) {
    delay(3000);
  }
  if (HTTPS_INSECURE) {
    secure_client.setInsecure();
  }

  xTaskCreatePinnedToCore(
      senderTask,
      "http_sender",
      16384,
      nullptr,
      1,
      &sender_task_handle,
      0);

  Serial.println("ESP32 live HTTP listo.");
  Serial.print("VPS: ");
  Serial.println(VPS_BASE_URL);
  Serial.print("Session: ");
  Serial.println(SESSION_ID);
  Serial.print("Batch size: ");
  Serial.println(configuredBatchSize());
  Serial.print("TX queue depth: ");
  Serial.println(TX_QUEUE_DEPTH);
  Serial.print("Print each sample: ");
  Serial.println(PRINT_EACH_SAMPLE ? "true" : "false");
  if (PRINT_EACH_SAMPLE) {
    Serial.println("seq,timestamp_ms,acc_x_g,acc_y_g,acc_z_g");
  }

  next_sample_us = micros();
}

void printMetrics() {
  uint8_t queued = 0;
  uint16_t active = 0;
  portENTER_CRITICAL(&batch_mux);
  queued = tx_queued;
  active = active_count;
  portEXIT_CRITICAL(&batch_mux);

  Serial.print("METRICS seq=");
  Serial.print(sequence_number);
  Serial.print(" queued=");
  Serial.print(queued);
  Serial.print(" active=");
  Serial.print(active);
  Serial.print(" sent=");
  Serial.print(batches_sent);
  Serial.print(" failed=");
  Serial.print(batches_failed);
  Serial.print(" dropped=");
  Serial.print(dropped_batches);
  Serial.print(" last_build_ms=");
  Serial.print(last_build_ms);
  Serial.print(" last_post_ms=");
  Serial.print(last_post_ms);
  Serial.print(" last_connect_ms=");
  Serial.print(last_connect_ms);
  Serial.print(" tls_connects=");
  Serial.print(tls_connects);
  Serial.print(" last_status=");
  Serial.print(last_status_code);
  Serial.print(" rssi=");
  Serial.println(WiFi.RSSI());
}

void loop() {
  unsigned long now = micros();
  if ((long)(now - next_sample_us) < 0) {
    return;
  }
  next_sample_us += SAMPLE_PERIOD_US;

  int16_t raw_x, raw_y, raw_z;
  if (!readAccelerationRaw(raw_x, raw_y, raw_z)) {
    Serial.println("ERROR_READ");
    return;
  }

  Sample sample;
  sample.timestamp_ms = millis();
  sample.acc_x_g = raw_x * 0.0039f;
  sample.acc_y_g = raw_y * 0.0039f;
  sample.acc_z_g = raw_z * 0.0039f;

  uint16_t batch_size = configuredBatchSize();
  uint32_t seq = sequence_number++;
  uint32_t seq_start = seq - active_count;
  bool batch_full = false;

  portENTER_CRITICAL(&batch_mux);
  if (active_count < batch_size) {
    active_batch[active_count++] = sample;
    batch_full = active_count >= batch_size;
  }
  portEXIT_CRITICAL(&batch_mux);

  if (batch_full) {
    queueBatchForSend(seq_start, batch_size);
  }

  if (PRINT_EACH_SAMPLE) {
    Serial.print(seq);
    Serial.print(",");
    Serial.print(sample.timestamp_ms);
    Serial.print(",");
    Serial.print(sample.acc_x_g, 5);
    Serial.print(",");
    Serial.print(sample.acc_y_g, 5);
    Serial.print(",");
    Serial.println(sample.acc_z_g, 5);
  }

  unsigned long now_ms = millis();
  if (now_ms - last_metrics_ms >= METRICS_EVERY_MS) {
    last_metrics_ms = now_ms;
    printMetrics();
  }
}
