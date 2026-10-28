#include "analysis.h"

#include <Arduino.h>
#include <math.h>

#include "fft.h"

namespace {

float window[FFT_SIZE];  // Hann window, reduces leakage between bins
float windowSum = 0;
float re[FFT_SIZE];
float im[FFT_SIZE];
float power[FFT_SIZE / 2];  // amplitude squared, summed over the 3 axes (g^2)

}  // namespace

void Analyzer::begin() {
  fftInit();
  windowSum = 0;
  for (int i = 0; i < FFT_SIZE; i++) {
    window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / FFT_SIZE));
    windowSum += window[i];
  }
  reset();
}

void Analyzer::reset() {
  head_ = 0;
  filled_ = 0;
  sinceLast_ = 0;
}

void Analyzer::push(const int16_t sample[3]) {
  for (int a = 0; a < 3; a++) buffer_[head_][a] = sample[a];
  head_ = (head_ + 1) % FFT_SIZE;
  if (filled_ < FFT_SIZE) filled_++;
  sinceLast_++;
}

bool Analyzer::ready() const { return filled_ == FFT_SIZE && sinceLast_ >= HOP_SIZE; }

void Analyzer::compute(Spectrum& out) {
  uint32_t t0 = micros();
  sinceLast_ = 0;
  for (int k = 0; k < FFT_SIZE / 2; k++) power[k] = 0;

  for (int axis = 0; axis < 3; axis++) {
    // Oldest to newest, converted to g.
    double sum = 0;
    for (int i = 0; i < FFT_SIZE; i++) {
      re[i] = buffer_[(head_ + i) % FFT_SIZE][axis] / ACCEL_COUNTS_PER_G;
      sum += re[i];
    }
    // Removing the mean removes gravity, which is constant.
    float mean = (float)(sum / FFT_SIZE);
    double squares = 0;
    for (int i = 0; i < FFT_SIZE; i++) {
      float v = re[i] - mean;
      squares += (double)v * v;
      re[i] = v * window[i];
      im[i] = 0;
    }
    out.rmsG[axis] = (float)sqrt(squares / FFT_SIZE);

    fftForward(re, im);
    // Single-sided amplitude: a sine of amplitude A shows as A in its bin.
    float scale = 2.0f / windowSum;
    for (int k = 1; k < FFT_SIZE / 2; k++) {
      float amp = scale * sqrtf(re[k] * re[k] + im[k] * im[k]);
      power[k] += amp * amp;
    }
  }

  // Strongest frequency, refined between bins with a parabola fit.
  int first = (int)ceilf(MIN_PEAK_HZ / binHz());
  int best = first;
  for (int k = first; k < FFT_SIZE / 2 - 1; k++) {
    if (power[k] > power[best]) best = k;
  }
  float left = sqrtf(power[best - 1]), mid = sqrtf(power[best]), right = sqrtf(power[best + 1]);
  float denom = left - 2 * mid + right;
  float delta = denom != 0 ? 0.5f * (left - right) / denom : 0;
  if (delta > 0.5f) delta = 0.5f;
  if (delta < -0.5f) delta = -0.5f;
  out.peakHz = (best + delta) * binHz();
  out.peakMg = (mid - 0.25f * (left - right) * delta) * 1000.0f;

  // Merge neighbouring bins for the PC, keeping the larger value so peaks
  // keep their height.
  const int merge = (FFT_SIZE / 2) / SPECTRUM_BINS;
  for (int b = 0; b < SPECTRUM_BINS; b++) {
    float biggest = 0;
    for (int j = 0; j < merge; j++) {
      float amp = sqrtf(power[b * merge + j]);
      if (amp > biggest) biggest = amp;
    }
    out.binsMg[b] = biggest * 1000.0f;
  }

  out.index = count_++;
  out.dspMs = (micros() - t0) / 1000.0f;
}
