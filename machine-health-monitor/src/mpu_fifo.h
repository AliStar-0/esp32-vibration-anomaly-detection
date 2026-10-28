// MPU6050 driver for continuous accelerometer sampling through the FIFO.
#pragma once

#include <stdint.h>

constexpr int MPU_FIFO_OVERFLOW = -1;  // samples were lost, FIFO restarted
constexpr int MPU_BUS_ERROR = -2;      // the sensor did not answer

// Resets and configures the sensor: accelerometer only, 1 kHz, +-4 g, FIFO on.
// Returns false if nothing answers on the I2C bus.
bool mpuBegin();

// Moves up to maxSamples readings (x, y, z raw counts) from the sensor FIFO
// into out. Returns how many were read, or one of the negative codes above.
int mpuReadFifo(int16_t out[][3], int maxSamples);
