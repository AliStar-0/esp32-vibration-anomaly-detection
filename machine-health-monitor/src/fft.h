// In-place radix-2 FFT of FFT_SIZE complex points.
#pragma once

#include "config.h"

static_assert((FFT_SIZE & (FFT_SIZE - 1)) == 0, "FFT_SIZE must be a power of two");

// Builds the lookup tables. Call once before fftForward.
void fftInit();

// re and im hold FFT_SIZE values each and are replaced by the transform.
void fftForward(float* re, float* im);
