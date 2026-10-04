#pragma once

// ============================================================
//  CPU-Pruefung beim Start
//
//  Eine Skull-Version, die mit AVX2 gebaut wurde (Standard auf x86-64), stuerzt auf einer CPU ohne AVX2
//  sonst mit "Illegal instruction" ab. Das Programm prueft deshalb als Erstes, ob die CPU (und das
//  Betriebssystem) AVX2 und FMA koennen, und sagt sonst, was zu tun ist.
// ============================================================
#if defined(__AVX2__) && (defined(__x86_64__) || defined(_M_X64))
    #define SKULL_NEEDS_AVX2 1
    #if defined(_MSC_VER)
        #include <intrin.h>
        #include <immintrin.h>
    #endif
#endif

// true, wenn die CPU alle Befehle kann, mit denen dieses Programm gebaut wurde
inline bool skull_cpu_supported() {
#if defined(SKULL_NEEDS_AVX2)
    #if defined(_MSC_VER)
        int r[4];
        __cpuid(r, 0);
        if (r[0] < 7) return false;
        __cpuid(r, 1);
        const bool fma = (r[2] & (1 << 12)) != 0, osxsave = (r[2] & (1 << 27)) != 0, avx = (r[2] & (1 << 28)) != 0;
        if (!fma || !osxsave || !avx) return false;
        if ((_xgetbv(0) & 6) != 6) return false;   // das Betriebssystem sichert die YMM-Register
        __cpuidex(r, 7, 0);
        return (r[1] & (1 << 5)) != 0;             // AVX2
    #else
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    #endif
#else
    return true;
#endif
}

inline const char* skull_cpu_help() {
    return "FEHLER: Diese Skull-Version nutzt AVX2-Befehle, die diese CPU (oder dieses Betriebssystem) nicht bietet.\n"
           "Loesung: das Paket mit '-compat' im Namen herunterladen, oder selbst bauen mit\n"
           "  cmake -S . -B build -DSKULL_ENABLE_AVX2=OFF   (oder: ./build.sh, das erkennt es automatisch)\n";
}
