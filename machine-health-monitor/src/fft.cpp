#include "fft.h"

#include <math.h>
#include <stdint.h>

namespace {

float cosTable[FFT_SIZE / 2];
float sinTable[FFT_SIZE / 2];
uint16_t bitReversed[FFT_SIZE];

}  // namespace

void fftInit() {
  int bits = 0;
  while ((1 << bits) < FFT_SIZE) bits++;

  for (int i = 0; i < FFT_SIZE; i++) {
    int r = 0;
    for (int b = 0; b < bits; b++) {
      if (i & (1 << b)) r |= 1 << (bits - 1 - b);
    }
    bitReversed[i] = (uint16_t)r;
  }
  for (int k = 0; k < FFT_SIZE / 2; k++) {
    double angle = 2.0 * M_PI * k / FFT_SIZE;
    cosTable[k] = (float)cos(angle);
    sinTable[k] = (float)-sin(angle);
  }
}

void fftForward(float* re, float* im) {
  // Reorder so the butterflies can work in place.
  for (int i = 0; i < FFT_SIZE; i++) {
    int j = bitReversed[i];
    if (j > i) {
      float t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }

  for (int len = 2; len <= FFT_SIZE; len <<= 1) {
    int half = len / 2;
    int step = FFT_SIZE / len;
    for (int start = 0; start < FFT_SIZE; start += len) {
      for (int k = 0; k < half; k++) {
        float wr = cosTable[k * step];
        float wi = sinTable[k * step];
        int a = start + k;
        int b = a + half;
        float tr = re[b] * wr - im[b] * wi;
        float ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
      }
    }
  }
}
