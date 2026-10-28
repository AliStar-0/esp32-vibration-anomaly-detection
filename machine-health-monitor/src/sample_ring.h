// Lock-free ring buffer passing samples from the sampling task (core 0) to the
// analysis code (core 1). Safe with exactly one writer and one reader.
#pragma once

#include <atomic>
#include <stdint.h>

template <uint32_t N>
class SampleRing {
  static_assert((N & (N - 1)) == 0, "N must be a power of two");

 public:
  bool push(const int16_t s[3]) {
    uint32_t head = head_.load(std::memory_order_relaxed);
    if (head - tail_.load(std::memory_order_acquire) >= N) return false;  // full
    for (int i = 0; i < 3; i++) data_[head & (N - 1)][i] = s[i];
    head_.store(head + 1, std::memory_order_release);
    return true;
  }

  bool pop(int16_t s[3]) {
    uint32_t tail = tail_.load(std::memory_order_relaxed);
    if (tail == head_.load(std::memory_order_acquire)) return false;  // empty
    for (int i = 0; i < 3; i++) s[i] = data_[tail & (N - 1)][i];
    tail_.store(tail + 1, std::memory_order_release);
    return true;
  }

 private:
  int16_t data_[N][3];
  std::atomic<uint32_t> head_{0};
  std::atomic<uint32_t> tail_{0};
};
