#include <Wire.h>

// ADXL345 I2C address. If SDO is tied to GND it is usually 0x53.
// If SDO is tied to 3.3V it is usually 0x1D.
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
static const unsigned long SAMPLE_PERIOD_US = 10000; // 100 Hz

unsigned long next_sample_us = 0;

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

void setup() {
  Serial.begin(115200);
  delay(1000);

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

  Serial.println("timestamp_ms,acc_x_g,acc_y_g,acc_z_g");
  next_sample_us = micros();
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

  // In full-resolution mode the scale is approximately 3.9 mg/LSB.
  float acc_x = raw_x * 0.0039f;
  float acc_y = raw_y * 0.0039f;
  float acc_z = raw_z * 0.0039f;

  Serial.print(millis());
  Serial.print(',');
  Serial.print(acc_x, 5);
  Serial.print(',');
  Serial.print(acc_y, 5);
  Serial.print(',');
  Serial.println(acc_z, 5);
}
