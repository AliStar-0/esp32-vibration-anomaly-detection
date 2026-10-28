#include "sampler.h"

#include <Arduino.h>

#include <atomic>

#include "mpu_fifo.h"
#include "sample_ring.h"

namespace {

SampleRing<4096> ring;  // about 4 s of samples
std::atomic<uint32_t> generation{0};
std::atomic<uint32_t> overflows{0};
std::atomic<uint32_t> restarts{0};
std::atomic<bool> sensorOk{false};
uint32_t lastDataMs = 0;
uint32_t lastRetryMs = 0;

// No data for this long means the sensor lost power or settings.
constexpr uint32_t STALL_MS = 200;
constexpr uint32_t RETRY_MS = 1000;

void markGap() { generation.fetch_add(1); }

void restartSensor() {
  sensorOk = mpuBegin();
  restarts.fetch_add(1);
  markGap();
  lastDataMs = millis();
}

}  // namespace

bool samplerBegin() {
  sensorOk = mpuBegin();
  lastDataMs = millis();
  return sensorOk;
}

void samplerPoll() {
  uint32_t now = millis();

  if (!sensorOk) {
    if (now - lastRetryMs >= RETRY_MS) {
      lastRetryMs = now;
      restartSensor();
    }
    return;
  }

  int16_t chunk[20][3];
  for (;;) {
    int n = mpuReadFifo(chunk, 20);
    if (n == MPU_FIFO_OVERFLOW) {
      overflows.fetch_add(1);
      markGap();
      return;
    }
    if (n == MPU_BUS_ERROR) {
      sensorOk = false;
      lastRetryMs = now;
      markGap();
      return;
    }
    if (n > 0) lastDataMs = now;
    for (int i = 0; i < n; i++) {
      if (!ring.push(chunk[i])) {  // the analysis fell behind
        overflows.fetch_add(1);
        markGap();
        return;
      }
    }
    if (n < 20) break;
  }

  if (now - lastDataMs > STALL_MS) restartSensor();
}

bool samplerPop(int16_t s[3]) { return ring.pop(s); }

SamplerStats samplerStats() {
  return {generation.load(), overflows.load(), restarts.load(), sensorOk.load()};
}
