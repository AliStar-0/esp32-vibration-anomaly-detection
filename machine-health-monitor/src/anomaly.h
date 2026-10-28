// On-device anomaly detection: compares each spectrum with what the model
// learned from healthy running (include/model.h, made by tools/train.py).
#pragma once

#include <stdint.h>

#include "analysis.h"

enum class MachineState : uint8_t {
  NoModel,   // include/model.h is still the placeholder
  Stopped,   // no running vibration
  Starting,  // just started; alarms held back while it settles
  Ok,
  Warning,
  Fault,
};

const char* stateName(MachineState s);

struct Assessment {
  MachineState state;
  float score;        // this spectrum: worst-band error / alarm level
  float smoothed;     // score averaged over the last few spectra
  float healthPct;    // 100 = like healthy running, 50 at the warning level
  int worstBand;      // band that differs most from healthy
  float worstLowHz;   // its frequency range
  float worstHighHz;
  float totalRmsMg;   // overall vibration, all axes
};

class Detector {
 public:
  Assessment update(const Spectrum& s);

 private:
  MachineState state_ = MachineState::Starting;
  float smoothed_ = -1;
  int settle_ = 10;  // SETTLE_SPECTRA: also wait after power-on
  int above1_ = 0;
  int above2_ = 0;
  int below_ = 0;
};
