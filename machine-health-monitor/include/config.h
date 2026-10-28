// Settings for the whole firmware.
#pragma once

#include <stdint.h>

// ---- Pins (same wiring as a standard ESP32 DevKit I2C setup) ---------------
constexpr int PIN_SDA = 21;
constexpr int PIN_SCL = 22;
constexpr int PIN_LED = 2;  // on-board LED, blinks once per spectrum

// ---- Sensor ----------------------------------------------------------------
constexpr uint8_t MPU_ADDR = 0x68;  // 0x69 if AD0 is tied to 3V3
constexpr uint32_t I2C_HZ = 400000;

// The MPU6050 samples on its own clock and stores readings in its FIFO, so
// the sample timing is exact even when the ESP32 is busy.
constexpr float SAMPLE_RATE_HZ = 1000.0f;
// Accelerometer range +-4 g: 8192 counts per g.
constexpr float ACCEL_COUNTS_PER_G = 8192.0f;

// ---- Analysis ----------------------------------------------------------------
// 1024 samples = 1.02 s of data, giving 0.98 Hz frequency resolution.
constexpr int FFT_SIZE = 1024;
// A new spectrum every 512 samples (0.5 s), so windows overlap by half.
constexpr int HOP_SIZE = 512;
// Bins sent to the PC: the 512 FFT bins merged in pairs, 1.95 Hz each.
constexpr int SPECTRUM_BINS = 256;
// Peaks below this are ignored (hand movement, slow drift).
constexpr float MIN_PEAK_HZ = 5.0f;

constexpr uint32_t SERIAL_BAUD = 115200;
