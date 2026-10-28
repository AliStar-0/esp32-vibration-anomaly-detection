// Continuous sampling: keeps the sensor FIFO drained and recovers from faults.
#pragma once

#include <stdint.h>

struct SamplerStats {
  // Increases every time the sample stream had a gap (lost samples, sensor
  // restart). The analysis restarts its window when this changes.
  uint32_t generation;
  uint32_t overflows;  // samples lost because they were not read in time
  uint32_t restarts;   // sensor set up again after it stopped answering
  bool sensorOk;
};

bool samplerBegin();

// Reads everything waiting in the sensor FIFO. Called every 10 ms by the
// sampling task.
void samplerPoll();

// Takes the oldest unread sample (raw counts x, y, z). False if none.
bool samplerPop(int16_t s[3]);

SamplerStats samplerStats();
