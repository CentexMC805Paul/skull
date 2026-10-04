// ============================================================
//  Test fuer src/rng.h: Zufallszahlen muessen auf jedem System dieselben sein.
//
//  Hintergrund: std::mt19937 ist im C++-Standard bitgenau festgelegt, die
//  std::*_distribution-Klassen nicht (libstdc++, libc++ und MSVC liefern bei
//  gleichem Seed andere Werte). Skull nutzt deshalb eigene Umrechnungen.
//  Dieser Test schlaegt auf einer Plattform an, auf der sich das aendert.
//
//  Exit-Code 0 = alles in Ordnung.
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include "rng.h"

static int failures = 0;

static void expect(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "ok" : "FEHLER", what.c_str());
    if (!ok) ++failures;
}

int main() {
    // 1) Der Motor selbst: [rand.predef] verlangt, dass der 10000. Wert eines
    //    default-konstruierten mt19937 gleich 4123659995 ist.
    {
        std::mt19937 g;
        uint32_t v = 0;
        for (int i = 0; i < 10000; ++i) v = g();
        expect(v == 4123659995u, "mt19937: 10000. Wert = 4123659995 (Vorgabe des C++-Standards)");
    }

    // 2) Umrechnung in [0,1): bitgenau dieselben Werte wie NumPys
    //    np.random.seed(42); np.random.rand() (gleiche 27+26-Bit-Konstruktion).
    {
        std::mt19937 g(42);
        const double ref[4] = {0.37454011884736249, 0.95071430640991617,
                               0.73199394181140509, 0.5986584841970366};
        bool same = true;
        for (double r : ref) same = same && (rng_uniform01(g) == r);
        expect(same, "rng_uniform01(seed 42): erste 4 Werte stimmen bitgenau (wie NumPy)");
    }

    // 3) Intervall und Ganzzahlen
    {
        std::mt19937 g(7);
        expect(rng_uniform(g, -2.0, 3.0) == -1.618458553130214, "rng_uniform(-2, 3) mit seed 7: fester Wert");

        std::mt19937 h(42);
        const size_t ref[6] = {499, 710, 31, 444, 313, 902};
        bool same = true;
        for (size_t r : ref) same = same && (rng_below(h, 1000) == r);
        expect(same, "rng_below(1000) mit seed 42: feste Folge 499 710 31 444 313 902");
    }

    // 4) Eigenschaften
    {
        std::mt19937 g(1);
        bool in_range = true;
        for (int i = 0; i < 100000; ++i) {
            double u = rng_uniform01(g);
            if (!(u >= 0.0 && u < 1.0)) in_range = false;
        }
        expect(in_range, "rng_uniform01 liegt immer in [0, 1)");

        bool below_ok = true;
        for (size_t n : {1u, 2u, 3u, 7u, 1000u}) {
            for (int i = 0; i < 2000; ++i) if (rng_below(g, n) >= n) below_ok = false;
        }
        expect(below_ok, "rng_below(n) liegt immer in [0, n)");
        expect(rng_below(g, 0) == 0 && rng_below(g, 1) == 0, "rng_below(0) und rng_below(1) liefern 0");

        // Grobe Gleichverteilung: 6 Fachwerte, je ~10000 von 60000
        std::vector<int> count(6, 0);
        for (int i = 0; i < 60000; ++i) ++count[rng_below(g, 6)];
        bool balanced = true;
        for (int c : count) if (std::abs(c - 10000) > 500) balanced = false;   // ~5 Sigma
        expect(balanced, "rng_below(6): Haeufigkeiten gleichmaessig (je ca. 10000 +- 500)");
    }

    if (failures) { std::printf("\n%d Pruefung(en) fehlgeschlagen\n", failures); return 1; }
    std::printf("\nAlle Pruefungen bestanden\n");
    return 0;
}
