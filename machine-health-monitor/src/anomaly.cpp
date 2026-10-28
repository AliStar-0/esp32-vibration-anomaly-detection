#include "anomaly.h"

#include <math.h>

#include "model.h"

static_assert(MODEL_INPUT_BINS == SPECTRUM_BINS,
              "model.h was made for a different spectrum size. Run train.bat again.");

namespace {

// Smoothing and alarm rules; tools/train.py uses the same values.
constexpr float SMOOTHING = 0.3f;    // weight of the newest score
constexpr int ALARM_RUN = 3;         // spectra in a row above a level (1.5 s)
constexpr int CLEAR_RUN = 6;         // spectra in a row below 0.8 to clear (3 s)
constexpr int SETTLE_SPECTRA = 10;   // ignore alarms for 5 s after starting
constexpr float WARNING_LEVEL = 1.0f;
constexpr float FAULT_LEVEL = 2.0f;
constexpr float CLEAR_LEVEL = 0.8f;
constexpr float BIN_HZ = SAMPLE_RATE_HZ / 2 / SPECTRUM_BINS;

// Band features: log10 of the amplitude in each band (same as train.py).
void bandFeatures(const float* binsMg, float* out) {
  for (int k = 0; k < MODEL_N_BANDS; k++) {
    float sum = 0;
    for (int i = MODEL_BAND_EDGES[k]; i < MODEL_BAND_EDGES[k + 1]; i++) sum += binsMg[i] * binsMg[i];
    out[k] = log10f(sqrtf(sum) + MODEL_FLOOR_MG);
  }
}

// Autoencoder: dense layers, ReLU on hidden layers, linear output.
void reconstruct(const float* in, float* out) {
  static float a[MODEL_MAX_WIDTH], b[MODEL_MAX_WIDTH];
  const float* x = in;
  const float* w = MODEL_WEIGHTS;
  const float* bias = MODEL_BIASES;
  for (int layer = 0; layer < MODEL_N_LAYERS; layer++) {
    int nIn = MODEL_LAYER_SIZES[layer], nOut = MODEL_LAYER_SIZES[layer + 1];
    bool last = layer == MODEL_N_LAYERS - 1;
    float* y = last ? out : (layer % 2 == 0 ? a : b);
    for (int o = 0; o < nOut; o++) {
      float acc = bias[o];
      for (int i = 0; i < nIn; i++) acc += w[o * nIn + i] * x[i];
      y[o] = (!last && acc < 0) ? 0 : acc;
    }
    w += nIn * nOut;
    bias += nOut;
    x = y;
  }
}

}  // namespace

const char* stateName(MachineState s) {
  switch (s) {
    case MachineState::NoModel: return "NO MODEL";
    case MachineState::Stopped: return "STOPPED";
    case MachineState::Starting: return "STARTING";
    case MachineState::Ok: return "OK";
    case MachineState::Warning: return "WARNING";
    case MachineState::Fault: return "FAULT";
  }
  return "?";
}

Assessment Detector::update(const Spectrum& s) {
  Assessment r{};
  r.totalRmsMg = 1000.0f * sqrtf(s.rmsG[0] * s.rmsG[0] + s.rmsG[1] * s.rmsG[1] + s.rmsG[2] * s.rmsG[2]);
  r.worstBand = -1;

#if MODEL_IS_PLACEHOLDER
  r.state = MachineState::NoModel;
  r.healthPct = NAN;
  return r;
#endif

  if (r.totalRmsMg < MODEL_STOP_RMS_MG) {
    state_ = MachineState::Stopped;
    smoothed_ = -1;
    settle_ = SETTLE_SPECTRA;
    above1_ = above2_ = below_ = 0;
    r.state = state_;
    r.healthPct = NAN;
    return r;
  }

  float z[MODEL_N_BANDS], rec[MODEL_N_BANDS];
  bandFeatures(s.binsMg, z);
  for (int k = 0; k < MODEL_N_BANDS; k++) z[k] = (z[k] - MODEL_FEATURE_MEAN[k]) / MODEL_FEATURE_STD[k];
  reconstruct(z, rec);

  float worst = 0;
  for (int k = 0; k < MODEL_N_BANDS; k++) {
    float e = (rec[k] - z[k]) * (rec[k] - z[k]);
    if (e > worst) {
      worst = e;
      r.worstBand = k;
    }
  }
  r.score = worst / MODEL_THRESHOLD;
  smoothed_ = smoothed_ < 0 ? r.score : SMOOTHING * r.score + (1 - SMOOTHING) * smoothed_;
  r.smoothed = smoothed_;
  r.healthPct = 100.0f * expf(-0.693f * smoothed_ * smoothed_);
  r.worstLowHz = MODEL_BAND_EDGES[r.worstBand] * BIN_HZ;
  r.worstHighHz = MODEL_BAND_EDGES[r.worstBand + 1] * BIN_HZ;

  above1_ = smoothed_ > WARNING_LEVEL ? above1_ + 1 : 0;
  above2_ = smoothed_ > FAULT_LEVEL ? above2_ + 1 : 0;
  below_ = smoothed_ < CLEAR_LEVEL ? below_ + 1 : 0;

  if (state_ == MachineState::Stopped) state_ = MachineState::Starting;
  if (settle_ > 0) {
    settle_--;
    if (settle_ == 0) state_ = MachineState::Ok;
  } else if (above2_ >= ALARM_RUN) {
    state_ = MachineState::Fault;
  } else if (above1_ >= ALARM_RUN && state_ != MachineState::Fault) {
    state_ = MachineState::Warning;
  } else if (below_ >= CLEAR_RUN || state_ == MachineState::Starting) {
    state_ = MachineState::Ok;
  }
  r.state = state_;
  return r;
}
