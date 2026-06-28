#pragma once

const char *WIFI_SSID = "TU_WIFI";
const char *WIFI_PASSWORD = "TU_PASSWORD";

const char *VPS_BASE_URL = "https://tu-dominio.example";
const char *VPS_HOST = "tu-dominio.example";
const uint16_t VPS_PORT = 443;
const char *VPS_BATCH_PATH = "/api/v1/samples/batch";
const char *VPS_API_KEY = "TU_API_KEY_DEL_VPS";

const char *DEVICE_ID = "esp32-train-01";
const char *SESSION_ID = "live_test_001";

// Set to 0 to let the ESP32 choose automatically.
const int WIFI_CHANNEL = 0;

// 50 samples at 100 Hz = one HTTP POST every 500 ms.
// This keeps latency low while reducing HTTPS overhead on the ESP32.
const uint16_t HTTP_BATCH_SIZE = 50;
const uint16_t SAMPLE_RATE_HZ = 100;
const bool PRINT_EACH_SAMPLE = false;
const uint32_t METRICS_EVERY_MS = 5000;
const uint32_t HTTPS_CONNECT_TIMEOUT_MS = 8000;
const uint32_t HTTPS_READ_TIMEOUT_MS = 1500;

// For the current MVP this avoids certificate pinning issues on ESP32.
// Use a pinned CA certificate before treating this as production-grade transport security.
const bool HTTPS_INSECURE = true;
