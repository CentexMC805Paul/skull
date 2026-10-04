#pragma once
#include <cstddef>
#include <functional>

#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#else
    #include <pthread.h>
#endif

// ============================================================
//  Grosser Stack fuer den Interpreter
//
//  Der Interpreter ist ein Tree-Walker: jede Skull-Funktionsebene
//  belegt mehrere C++-Frames. Wie gross die sind, haengt stark vom
//  Compiler und von der Optimierungsstufe ab (gemessen: ca. 1.5 KB
//  bei -O2, aber 9-12 KB bei -O3 pro Skull-Aufruf). Mit dem
//  Standardstack (Linux/macOS 8 MB, Windows 1 MB) waere ein festes
//  Rekursionslimit deshalb nie sicher.
//
//  run_with_stack() fuehrt die Funktion in einem Thread mit
//  garantiert grossem Stack aus (nur reservierter Adressraum, der
//  Speicher wird erst bei Bedarf belegt). Gelingt das Anlegen des
//  Threads nicht, laeuft die Funktion im aktuellen Thread weiter.
// ============================================================

namespace skull_stack_detail {

struct Ctx {
    std::function<int()>* fn;
    int                   result;
};

#if defined(_WIN32)
inline DWORD WINAPI thread_main(LPVOID p) {
    Ctx* c = static_cast<Ctx*>(p);
    c->result = (*c->fn)();
    return 0;
}
#else
inline void* thread_main(void* p) {
    Ctx* c = static_cast<Ctx*>(p);
    c->result = (*c->fn)();
    return nullptr;
}
#endif

}  // namespace skull_stack_detail

inline int run_with_stack(std::size_t stack_bytes, std::function<int()> fn) {
    using namespace skull_stack_detail;
    Ctx ctx{&fn, 1};

#if defined(_WIN32)
    HANDLE h = CreateThread(nullptr, stack_bytes, thread_main, &ctx,
                            STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
    if (!h) return fn();
    WaitForSingleObject(h, INFINITE);
    CloseHandle(h);
#else
    pthread_attr_t attr;
    if (pthread_attr_init(&attr) != 0) return fn();
    pthread_t th;
    bool started = pthread_attr_setstacksize(&attr, stack_bytes) == 0 &&
                   pthread_create(&th, &attr, thread_main, &ctx) == 0;
    pthread_attr_destroy(&attr);
    if (!started) return fn();
    pthread_join(th, nullptr);
#endif
    return ctx.result;
}
