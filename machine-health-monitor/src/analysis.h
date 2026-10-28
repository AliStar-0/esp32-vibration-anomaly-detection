// Turns the sample stream into vibration measurements and a spectrum.
#pragma once

#include <stdint.h>

#include "config.h"

struct Spectrum {
  uint32_t index;               // spectrum number since the start
  float rmsG[3];                // vibration strength per axis, g (gravity removed)
  float peakHz;                 // strongest frequency above MIN_PEAK_HZ
  float peakMg;                 // its amplitude, milli-g
  float binsMg[SPECTRUM_BINS];  // amplitude per frequency band, milli-g
  float dspMs;                  // time the analysis took on the ESP32
};

class Analyzer {
 public:
  void begin();
  // Forget the buffered samples, after a gap in the stream.
  void reset();
  void push(const int16_t sample[3]);
  // True when a new spectrum can be computed (every HOP_SIZE samples).
  bool ready() const;
  void compute(Spectrum& out);

  static float binHz() { return SAMPLE_RATE_HZ / FFT_SIZE; }
  static float outputBinHz() { return SAMPLE_RATE_HZ / 2 / SPECTRUM_BINS; }

 private:
  int16_t buffer_[FFT_SIZE][3];
  int head_ = 0;    // where the next sample goes; also the oldest sample
  int filled_ = 0;  // how many valid samples are in the buffer
  int sinceLast_ = 0;
  uint32_t count_ = 0;
};
