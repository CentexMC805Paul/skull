// ============================================================
//  SIMD STATUS AUSGEBEN
// ============================================================
inline void print_skull_info() {
    std::cout << "[Skull] Tensor-Engine geladen\n";
#if SKULL_AVX2
    std::cout << "[Skull] AVX2 aktiv (4x doubles pro Takt)\n";
#else
    std::cout << "[Skull] Scalar-Modus (kein AVX2)\n";
#endif
}
