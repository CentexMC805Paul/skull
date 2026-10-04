#pragma once
#include <cstdint>
#include <cstddef>
#include <random>

// ============================================================
//  Plattformunabhaengige Zufallszahlen
//
//  std::mt19937 ist im C++-Standard bitgenau festgelegt. Die Verteilungen
//  (std::uniform_real_distribution, std::uniform_int_distribution, ...)
//  dagegen nicht: libstdc++ (Linux), libc++ (macOS) und MSVC liefern bei
//  gleichem Seed verschiedene Zahlen. Damit haetten Startgewichte und
//  Trainingsfenster auf jedem System andere Werte, und ein Training waere
//  zwischen Systemen nicht reproduzierbar.
//
//  Die Funktionen hier rechnen die rohen 32-Bit-Werte des mt19937 selbst um.
//  Gleicher Seed = gleiche Zahlenfolge auf jedem System.
// ============================================================

// Gleichverteilt in [0, 1) mit 53 Bit Aufloesung.
inline double rng_uniform01(std::mt19937& g) {
    const uint32_t a = g() >> 5;    // 27 Bit
    const uint32_t b = g() >> 6;    // 26 Bit
    return ((double)a * 67108864.0 + (double)b) / 9007199254740992.0;
}

// Gleichverteilt in [lo, hi).
inline double rng_uniform(std::mt19937& g, double lo, double hi) {
    return lo + (hi - lo) * rng_uniform01(g);
}

// Gleichverteilt in {0, 1, ..., n-1} (ohne Modulo-Verzerrung, per Verwerfen). n == 0 liefert 0.
inline size_t rng_below(std::mt19937& g, size_t n) {
    if (n <= 1) return 0;
    const uint64_t range     = (uint64_t)n;
    const uint64_t threshold = (0 - range) % range;   // (2^64 - range) mod range
    for (;;) {
        const uint64_t hi = g();
        const uint64_t lo = g();
        const uint64_t x  = (hi << 32) | lo;
        if (x >= threshold) return (size_t)(x % range);
    }
}
