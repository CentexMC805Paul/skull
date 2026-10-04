// ============================================================
//  Test fuer das Sampling in src/generator.h: seed, top_k, top_p
//
//  - filter_probs: exakte Erwartungswerte fuer top_k / top_p / beides, Gleichstaende
//  - sample(): liefert nie ein herausgefiltertes Token (auch nicht bei Rundungsfehlern)
//  - skull_generate: gleicher seed = gleiche Ausgabe, anderer seed = andere Ausgabe,
//    top_k = 1 und winziges top_p entsprechen greedy
//
//  Exit-Code 0 = alles in Ordnung.
// ============================================================
#include <cstdio>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>
#include "generator.h"
#include "trainer.h"

static int failures = 0;

static void expect(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "ok" : "FEHLER", what.c_str());
    if (!ok) ++failures;
}

static bool near_vec(const FlatVec& a, const FlatVec& b, double tol = 1e-12) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (std::fabs(a[i] - b[i]) > tol) return false;
    return true;
}

static void check_filter() {
    std::printf("filter_probs\n");
    const FlatVec base = {0.5, 0.3, 0.1, 0.05, 0.05};
    { FlatVec p = base; filter_probs(p, 0, 1.0);
      expect(near_vec(p, base), "top_k = 0 und top_p = 1: unveraendert"); }
    { FlatVec p = base; filter_probs(p, 2, 1.0);
      expect(near_vec(p, {0.5 / 0.8, 0.3 / 0.8, 0, 0, 0}), "top_k = 2: die zwei wahrscheinlichsten, neu normiert"); }
    { FlatVec p = base; filter_probs(p, 1, 1.0);
      expect(near_vec(p, {1, 0, 0, 0, 0}), "top_k = 1: Einheitsvektor auf das wahrscheinlichste Token"); }
    { FlatVec p = base; filter_probs(p, 99, 1.0);
      expect(near_vec(p, base), "top_k groesser als das Vokabular: unveraendert"); }
    { FlatVec p = base; filter_probs(p, 0, 0.7);
      expect(near_vec(p, {0.5 / 0.8, 0.3 / 0.8, 0, 0, 0}), "top_p = 0.7: kleinste Menge mit Summe >= 0.7 (0.5 + 0.3)"); }
    { FlatVec p = base; filter_probs(p, 0, 0.5);
      expect(near_vec(p, {1, 0, 0, 0, 0}), "top_p = 0.5: 0.5 reicht schon (>=), nur ein Token"); }
    { FlatVec p = base; filter_probs(p, 0, 1e-9);
      expect(near_vec(p, {1, 0, 0, 0, 0}), "winziges top_p: mindestens ein Token bleibt (das wahrscheinlichste)"); }
    { FlatVec p = base; filter_probs(p, 4, 0.85);
      expect(near_vec(p, {0.5 / 0.9, 0.3 / 0.9, 0.1 / 0.9, 0, 0}), "top_k = 4 und top_p = 0.85: das strengere (top_p) gewinnt"); }
    { FlatVec p = base; filter_probs(p, 2, 0.99);
      expect(near_vec(p, {0.5 / 0.8, 0.3 / 0.8, 0, 0, 0}), "top_k = 2 und top_p = 0.99: das strengere (top_k) gewinnt"); }
    { FlatVec p = {0.25, 0.25, 0.25, 0.25}; filter_probs(p, 2, 1.0);
      expect(near_vec(p, {0.5, 0.5, 0, 0}), "Gleichstand: bei gleicher Wahrscheinlichkeit zaehlt die kleinere ID"); }
    { FlatVec p = {0.1, 0.4, 0.4, 0.1}; filter_probs(p, 1, 1.0);
      expect(near_vec(p, {0, 1, 0, 0}), "Gleichstand an der Spitze: kleinere ID (1) vor 2"); }
    { FlatVec p = base; filter_probs(p, 3, 0.9);
      double sum = 0.0; for (double x : p) sum += x;
      expect(std::fabs(sum - 1.0) < 1e-12, "Summe nach dem Filtern ist 1"); }
}

static void check_sample_never_filtered() {
    std::printf("sample(): nie ein herausgefiltertes Token\n");
    // Summe absichtlich weit unter 1 (wie ein extremer Rundungsfehler): in 40 % der Ziehungen ist der
    // Zufallswert groesser als die Summe, dann greift der Fallback. Er darf nie ein Token mit
    // Wahrscheinlichkeit 0 liefern (frueher: das letzte Token des Vokabulars, egal welche Wahrscheinlichkeit).
    FlatVec p = {0.3, 0.3, 0.0, 0.0};
    bool ok = true;
    std::mt19937 rng(123);
    for (int i = 0; i < 20000; ++i) {
        int t = sample(p, rng);
        if (t < 0 || t > 1) ok = false;
    }
    expect(ok, "20000 Ziehungen mit Fallback-Fall: nur Token mit Wahrscheinlichkeit > 0");
    FlatVec p2 = {0.5, 0.5 - 1e-12, 0.0, 0.0};
    ok = true;
    for (int i = 0; i < 200000; ++i) { int t = sample(p2, rng); if (t < 0 || t > 1) ok = false; }
    expect(ok, "200000 Ziehungen mit Summe knapp unter 1: nur Token mit Wahrscheinlichkeit > 0");
    FlatVec onehot = {0, 0, 1.0 - 1e-13, 0};
    ok = true;
    for (int i = 0; i < 50000; ++i) if (sample(onehot, rng) != 2) ok = false;
    expect(ok, "Einheitsvektor mit Rundungsrest: immer dasselbe Token");
}

// Zufaellige (untrainierte) Gewichte genuegen, um Reproduzierbarkeit zu pruefen.
static SkullWeights random_transformer(size_t vocab) {
    TransformerConfig c; c.vocab = vocab; c.dim = 16; c.context = 6; c.heads = 2; c.layers = 2;
    Transformer m(c);
    m.init(3);
    SkullWeights w;
    w.is_transformer = true; w.tcfg = c; w.dim = c.dim; w.vocab = vocab; w.tparams = m.params;
    return w;
}
static SkullWeights random_bigram(size_t vocab) {
    SkullWeights w;
    w.dim = 12; w.vocab = vocab;
    w.W_embed = xavier_init(vocab, w.dim, 1);
    w.W_hidden = xavier_init(w.dim, w.dim, 2);
    w.W_out = xavier_init(w.dim, vocab, 3);
    return w;
}

static std::string generated_part(const std::string& full) {
    const std::string marker = "--- Ausgabe ---\n";
    const size_t a = full.find(marker);
    const size_t b = full.find("\n--- Ende ---");
    if (a == std::string::npos || b == std::string::npos) return "";
    return full.substr(a + marker.size(), b - a - marker.size());
}

static std::string run_generate(const std::string& path, GenerateConfig cfg) {
    cfg.weights_path = path;
    std::ostringstream os;
    skull_generate(cfg, os);
    return os.str();
}

static void check_generate(const char* label, const SkullWeights& w) {
    std::printf("skull_generate (%s)\n", label);
    const std::string path = "gencheck_weights.tmp";
    save_weights(path, w);

    GenerateConfig base;
    base.prompt = "ab"; base.tokens = 80; base.temperature = 1.5;

    GenerateConfig s1 = base; s1.has_seed = true; s1.seed = 11;
    GenerateConfig s2 = base; s2.has_seed = true; s2.seed = 11;
    GenerateConfig s3 = base; s3.has_seed = true; s3.seed = 12;
    const std::string o1 = generated_part(run_generate(path, s1));
    const std::string o2 = generated_part(run_generate(path, s2));
    const std::string o3 = generated_part(run_generate(path, s3));
    expect(!o1.empty() && o1 == o2, "gleicher seed -> identische Ausgabe");
    expect(o1 != o3, "anderer seed -> andere Ausgabe");

    GenerateConfig greedy = base; greedy.temperature = 0.0;
    GenerateConfig k1 = base; k1.top_k = 1; k1.has_seed = true; k1.seed = 5;
    GenerateConfig p0 = base; p0.top_p = 1e-9; p0.has_seed = true; p0.seed = 6;
    const std::string og = generated_part(run_generate(path, greedy));
    expect(!og.empty() && generated_part(run_generate(path, k1)) == og, "top_k = 1 entspricht greedy");
    expect(generated_part(run_generate(path, p0)) == og, "winziges top_p entspricht greedy");

    GenerateConfig k5 = base; k5.top_k = 5; k5.has_seed = true; k5.seed = 11;
    expect(generated_part(run_generate(path, k5)) != o1, "top_k = 5 veraendert die Ausgabe gegenueber unbeschraenkt");

    bool threw = false;
    GenerateConfig bad = base; bad.top_p = 1.5;
    try { run_generate(path, bad); } catch (const std::runtime_error& e) { threw = std::string(e.what()).find("top_p") != std::string::npos; }
    expect(threw, "top_p ausserhalb (0, 1] -> Fehler");

    const std::string warn = run_generate(path, [&] { GenerateConfig c = greedy; c.top_k = 3; return c; }());
    expect(warn.find("top_k/top_p wirken nur bei temperature > 0") != std::string::npos,
           "top_k bei temperature = 0: Warnung");
    std::remove(path.c_str());
}

int main() {
    check_filter();
    check_sample_never_filtered();
    check_generate("Transformer", random_transformer(100));
    check_generate("Bigram", random_bigram(100));

    if (failures) { std::printf("\n%d Pruefung(en) fehlgeschlagen\n", failures); return 1; }
    std::printf("\nAlle Pruefungen bestanden\n");
    return 0;
}
