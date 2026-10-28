#include "mpu_fifo.h"

#include <Arduino.h>
#include <Wire.h>

#include "config.h"

namespace {

constexpr uint8_t REG_SMPLRT_DIV = 0x19;
constexpr uint8_t REG_CONFIG = 0x1A;
constexpr uint8_t REG_ACCEL_CONFIG = 0x1C;
constexpr uint8_t REG_FIFO_EN = 0x23;
constexpr uint8_t REG_INT_STATUS = 0x3A;
constexpr uint8_t REG_USER_CTRL = 0x6A;
constexpr uint8_t REG_PWR_MGMT_1 = 0x6B;
constexpr uint8_t REG_PWR_MGMT_2 = 0x6C;
constexpr uint8_t REG_FIFO_COUNT_H = 0x72;
constexpr uint8_t REG_FIFO_R_W = 0x74;

constexpr uint8_t USER_CTRL_FIFO_EN = 0x40;
constexpr uint8_t USER_CTRL_FIFO_RESET = 0x04;
constexpr uint8_t INT_STATUS_FIFO_OFLOW = 0x10;
constexpr int FIFO_BYTES = 1024;
constexpr int BYTES_PER_SAMPLE = 6;
// The ESP32 Wire buffer holds 128 bytes, so read at most 20 samples at once.
constexpr int MAX_SAMPLES_PER_READ = 20;

bool writeReg(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool readRegs(uint8_t reg, uint8_t* out, int n) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)MPU_ADDR, n) != n) return false;
  for (int i = 0; i < n; i++) out[i] = (uint8_t)Wire.read();
  return true;
}

bool restartFifo() {
  return writeReg(REG_USER_CTRL, USER_CTRL_FIFO_EN | USER_CTRL_FIFO_RESET);
}

}  // namespace

bool mpuBegin() {
  static bool wireStarted = false;
  if (!wireStarted) {
    Wire.begin(PIN_SDA, PIN_SCL, I2C_HZ);
    wireStarted = true;
  }

  if (!writeReg(REG_PWR_MGMT_1, 0x80)) return false;  // full reset
  delay(100);

  bool ok = true;
  ok &= writeReg(REG_PWR_MGMT_1, 0x00);    // awake, internal 8 MHz clock
  ok &= writeReg(REG_PWR_MGMT_2, 0x07);    // gyroscope off, it is not needed
  ok &= writeReg(REG_CONFIG, 0x01);        // low-pass 184 Hz, 1 kHz base rate
  ok &= writeReg(REG_SMPLRT_DIV, 0);       // 1 kHz / (1 + 0) = 1 kHz
  ok &= writeReg(REG_ACCEL_CONFIG, 0x08);  // +-4 g
  delay(50);                               // let the new settings take effect

  ok &= writeReg(REG_USER_CTRL, USER_CTRL_FIFO_RESET);
  delay(2);
  ok &= writeReg(REG_FIFO_EN, 0x08);  // accelerometer x, y, z into the FIFO
  ok &= writeReg(REG_USER_CTRL, USER_CTRL_FIFO_EN);

  uint8_t status;
  ok &= readRegs(REG_INT_STATUS, &status, 1);  // clears old flags
  return ok;
}

int mpuReadFifo(int16_t out[][3], int maxSamples) {
  uint8_t status;
  if (!readRegs(REG_INT_STATUS, &status, 1)) return MPU_BUS_ERROR;

  uint8_t countBytes[2];
  if (!readRegs(REG_FIFO_COUNT_H, countBytes, 2)) return MPU_BUS_ERROR;
  int count = (countBytes[0] << 8) | countBytes[1];

  if ((status & INT_STATUS_FIFO_OFLOW) || count >= FIFO_BYTES) {
    restartFifo();
    return MPU_FIFO_OVERFLOW;
  }

  int n = count / BYTES_PER_SAMPLE;
  if (n > maxSamples) n = maxSamples;
  if (n > MAX_SAMPLES_PER_READ) n = MAX_SAMPLES_PER_READ;
  if (n == 0) return 0;

  uint8_t raw[MAX_SAMPLES_PER_READ * BYTES_PER_SAMPLE];
  if (!readRegs(REG_FIFO_R_W, raw, n * BYTES_PER_SAMPLE)) return MPU_BUS_ERROR;

  for (int i = 0; i < n; i++) {
    const uint8_t* p = raw + i * BYTES_PER_SAMPLE;
    for (int axis = 0; axis < 3; axis++) {
      out[i][axis] = (int16_t)((p[2 * axis] << 8) | p[2 * axis + 1]);
    }
  }
  return n;
}
