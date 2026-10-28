// Machine health monitor, stage 2: vibration spectrum and anomaly detection.
//
// Core 0 keeps the sensor FIFO drained (1000 samples per second).
// Core 1 (this loop) computes a spectrum every 0.5 s, compares it with the
// healthy-running model and reports the machine state.
//
// Serial commands:
//   s  start streaming full spectra (used by tools/live_view.py)
//   q  stop streaming

#include <Arduino.h>

#include "analysis.h"
#include "anomaly.h"
#include "model.h"
#include "config.h"
#include "sampler.h"

static Analyzer analyzer;
static Spectrum spectrum;
static Detector detector;
static bool streaming = false;
static uint32_t lastGeneration = 0;
static uint32_t lastOverflows = 0;
static uint32_t lastRestarts = 0;
static bool ledOn = false;
static MachineState lastState = MachineState::Starting;

static void samplerTask(void*) {
  for (;;) {
    samplerPoll();
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

static void handleCommands() {
  while (Serial.available() > 0) {
    int c = Serial.read();
    if (c == 's') {
      streaming = true;
      Serial.println("# Streaming spectra.");
    } else if (c == 'q') {
      streaming = false;
      Serial.println("# Streaming stopped.");
    }
  }
}

static void report(const Spectrum& s, const Assessment& a) {
  float seconds = millis() / 1000.0f;
  Serial.printf("t=%7.1fs  %-8s", seconds, stateName(a.state));
  if (a.state == MachineState::Ok || a.state == MachineState::Warning || a.state == MachineState::Fault ||
      a.state == MachineState::Starting) {
    Serial.printf(" health %3.0f%%  score %5.2f", a.healthPct, a.smoothed);
  }
  Serial.printf(" | vib %5.1f mg  peak %6.1f Hz (%5.0f rpm) | dsp %.1f ms", a.totalRmsMg, s.peakHz,
                s.peakHz * 60.0f, s.dspMs);
  if (a.state == MachineState::Warning || a.state == MachineState::Fault) {
    Serial.printf(" | biggest change %.0f-%.0f Hz", a.worstLowHz, a.worstHighHz);
  }
  Serial.println();

  if (a.state != lastState) {
    if (a.state == MachineState::Warning || a.state == MachineState::Fault) {
      Serial.printf("# %s: vibration differs from healthy running, mostly at %.0f-%.0f Hz.\n",
                    stateName(a.state), a.worstLowHz, a.worstHighHz);
    } else if (a.state == MachineState::Ok && (lastState == MachineState::Warning || lastState == MachineState::Fault)) {
      Serial.println("# Back to normal.");
    }
    lastState = a.state;
  }

  if (!streaming) return;
  // Machine-readable lines for tools/live_view.py.
  Serial.printf("H,%s,%.3f,%.3f,%.1f,%.1f,%.1f\n", stateName(a.state), a.score, a.smoothed,
                isnan(a.healthPct) ? -1.0f : a.healthPct, a.worstLowHz, a.worstHighHz);
  Serial.printf("S,%.1f,%.5f,%.5f,%.5f,%.2f,%.2f", seconds, s.rmsG[0], s.rmsG[1], s.rmsG[2], s.peakHz, s.peakMg);
  for (int b = 0; b < SPECTRUM_BINS; b++) Serial.printf(",%.2f", s.binsMg[b]);
  Serial.println();
}

// LED: blinks while OK, fast while WARNING, solid on FAULT, off when stopped.
static void updateLed(MachineState st) {
  switch (st) {
    case MachineState::Fault: ledOn = true; break;
    case MachineState::Stopped: ledOn = false; break;
    default: ledOn = !ledOn; break;
  }
  digitalWrite(PIN_LED, ledOn ? HIGH : LOW);
}

static void checkStream() {
  SamplerStats st = samplerStats();
  if (st.generation == lastGeneration) return;
  lastGeneration = st.generation;
  analyzer.reset();  // the window would contain a jump, start it again

  if (st.overflows != lastOverflows) {
    Serial.println("# Warning: samples were lost, analysis restarted.");
  }
  if (st.restarts != lastRestarts) {
    Serial.println(st.sensorOk ? "# Sensor stopped answering and was set up again (loose wire?)."
                               : "# Sensor not answering. Check the wiring.");
  }
  lastOverflows = st.overflows;
  lastRestarts = st.restarts;
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  pinMode(PIN_LED, OUTPUT);
  delay(300);
  Serial.println("# Machine health monitor, stage 2: anomaly detection");

  while (!samplerBegin()) {
    Serial.println("# MPU6050 not found. Check wiring: SDA->21, SCL->22, VCC->3V3, GND->GND");
    delay(1000);
  }
  analyzer.begin();
  lastGeneration = samplerStats().generation;

  xTaskCreatePinnedToCore(samplerTask, "sampler", 4096, nullptr, 5, nullptr, 0);

  Serial.printf("# Ready. %d-point FFT at %.0f Hz: %.2f Hz resolution, new spectrum every %.1f s\n",
                FFT_SIZE, SAMPLE_RATE_HZ, Analyzer::binHz(), HOP_SIZE / SAMPLE_RATE_HZ);
#if MODEL_IS_PLACEHOLDER
  Serial.println("# No trained model yet: record healthy data, run train.bat, upload again.");
#elif defined(MODEL_SUMMARY)
  Serial.println("# Model: " MODEL_SUMMARY);
#endif
  Serial.println("# Send 's' to stream full spectra (tools/live_view.py does this), 'q' to stop.");
}

void loop() {
  handleCommands();
  checkStream();

  int16_t sample[3];
  while (samplerPop(sample)) {
    analyzer.push(sample);
    if (analyzer.ready()) {
      analyzer.compute(spectrum);
      Assessment a = detector.update(spectrum);
      report(spectrum, a);
      updateLed(a.state);
    }
  }
  delay(2);
}
