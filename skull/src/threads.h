#pragma once
#include <thread>
#include <vector>
#include <exception>
#include <algorithm>
#include <system_error>
#include <cstddef>

// ============================================================
//  Einfache, deterministische Parallelisierung
//
//  parallel_slots(n_slots, n_threads, fn) ruft fn(slot) fuer jeden Slot 0..n_slots-1 auf.
//  Slot s wird immer von Thread (s % n_threads) bearbeitet, in aufsteigender Reihenfolge.
//  Der Aufrufer ordnet Arbeit fest den SLOTS zu (nicht den Threads) und fasst die
//  Slot-Ergebnisse in fester Reihenfolge zusammen. Dann haengt das Ergebnis (bis auf das
//  letzte Bit) nicht davon ab, wie viele Threads tatsaechlich rechnen: die Thread-Anzahl
//  aendert nur, WER einen Slot rechnet, nicht WAS gerechnet wird.
//
//  Threads werden pro Aufruf gestartet (kein Pool): einfach und robust; der Aufwand
//  (einige zehn Mikrosekunden) ist gegen die Rechenzeit eines Trainingsschritts
//  vernachlaessigbar. Eine Ausnahme in einem Thread wird nach dem Join erneut geworfen.
//  Laesst sich ein Thread nicht starten (Ressourcenlimit), rechnet der Aufrufer seine
//  Slots selbst.
// ============================================================

inline size_t skull_hardware_threads() {
    unsigned n = std::thread::hardware_concurrency();
    return n == 0 ? 1 : (size_t)n;
}

template <typename F>
void parallel_slots(size_t n_slots, size_t n_threads, F&& fn) {
    if (n_slots == 0) return;
    n_threads = std::max<size_t>(1, std::min(n_threads, n_slots));
    if (n_threads == 1) {
        for (size_t s = 0; s < n_slots; ++s) fn(s);
        return;
    }

    std::vector<std::exception_ptr> errors(n_threads);
    auto run = [&](size_t w) {
        try {
            for (size_t s = w; s < n_slots; s += n_threads) fn(s);
        } catch (...) {
            errors[w] = std::current_exception();
        }
    };

    std::vector<std::thread> workers;
    std::vector<char> started(n_threads, 0);
    workers.reserve(n_threads - 1);
    for (size_t w = 1; w < n_threads; ++w) {
        try {
            workers.emplace_back(run, w);
            started[w] = 1;
        } catch (const std::system_error&) {
            // Thread nicht startbar: seine Slots rechnet der Aufrufer unten selbst
        }
    }
    run(0);
    for (size_t w = 1; w < n_threads; ++w)
        if (!started[w]) run(w);
    for (auto& th : workers) th.join();
    for (auto& e : errors)
        if (e) std::rethrow_exception(e);
}
