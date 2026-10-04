// ============================================================
//  Gradiententest fuer src/transformer.h
//
//  Vergleicht den von Hand geschriebenen Backward-Pass mit zentralen
//  endlichen Differenzen  (L(w+e) - L(w-e)) / 2e  fuer JEDEN Parameter.
//  Dazu: Kausalitaet, Verlust bei der Initialisierung, Adam kann eine
//  feste Sequenz auswendig lernen, Konfigurationspruefung.
//
//  Exit-Code 0 = alles in Ordnung.
// ============================================================
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <string>
#include <cstring>
#include "transformer.h"
#include "threads.h"

static int failures = 0;

static void expect(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "ok" : "FEHLER", what.c_str());
    if (!ok) ++failures;
}

static double loss_only(Transformer& m, const std::vector<int>& ids, const std::vector<int>& tgt) {
    m.forward(ids.data(), ids.size());
    return m.loss(tgt.data(), ids.size());
}

static void make_data(const TransformerConfig& cfg, unsigned seed, size_t t,
                      std::vector<int>& ids, std::vector<int>& tgt) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> tok(0, (int)cfg.vocab - 1);
    ids.resize(t); tgt.resize(t);
    for (size_t i = 0; i < t; ++i) { ids[i] = tok(rng); tgt[i] = tok(rng); }
}

// Parameter leicht stoeren, damit auch LayerNorm-beta und Biases ungleich 0 sind
static Transformer make_model(const TransformerConfig& cfg, unsigned seed) {
    Transformer m(cfg);
    m.init(seed);
    std::mt19937 rng(seed + 1000);
    std::uniform_real_distribution<double> noise(-0.2, 0.2);
    for (auto& p : m.params) p += noise(rng);
    return m;
}

static void check_gradients(const TransformerConfig& cfg, size_t t, const char* name) {
    std::printf("Gradiententest: %s (vocab=%zu dim=%zu context=%zu heads=%zu layers=%zu, %zu Parameter, t=%zu)\n",
                name, cfg.vocab, cfg.dim, cfg.context, cfg.heads, cfg.layers, cfg.param_count(), t);
    Transformer m = make_model(cfg, 7);
    std::vector<int> ids, tgt;
    make_data(cfg, 11, t, ids, tgt);

    m.zero_grads();
    m.step_loss_and_grad(ids.data(), tgt.data(), t);
    const std::vector<double> ana = m.grads;

    const double eps = 1e-5;
    double worst = 0.0;
    size_t worst_i = 0, bad = 0;
    for (size_t i = 0; i < m.params.size(); ++i) {
        const double orig = m.params[i];
        m.params[i] = orig + eps; const double lp = loss_only(m, ids, tgt);
        m.params[i] = orig - eps; const double lm = loss_only(m, ids, tgt);
        m.params[i] = orig;
        const double num = (lp - lm) / (2.0 * eps);
        const double diff = std::fabs(ana[i] - num);
        const double tol  = 1e-7 + 1e-5 * std::max(std::fabs(ana[i]), std::fabs(num));
        if (diff > tol) ++bad;
        const double rel = diff / (1e-7 + std::max(std::fabs(ana[i]), std::fabs(num)));
        if (rel > worst) { worst = rel; worst_i = i; }
    }
    std::printf("  groesster relativer Fehler %.3g bei Parameter %zu (analytisch %.6g)\n",
                worst, worst_i, ana[worst_i]);
    expect(bad == 0, "alle " + std::to_string(m.params.size()) + " Gradienten stimmen mit endlichen Differenzen ueberein (" +
                     std::to_string(bad) + " Abweichungen)");
}

static void check_causality(const TransformerConfig& cfg) {
    std::printf("Kausalitaet\n");
    Transformer m = make_model(cfg, 3);
    const size_t t = cfg.context, V = cfg.vocab;
    std::vector<int> a, tgt;
    make_data(cfg, 5, t, a, tgt);
    std::vector<double> la = m.forward(a.data(), t);          // Kopie

    // Token an Position k aendern: Logits der Positionen < k duerfen sich nicht aendern
    const size_t k = t / 2;
    std::vector<int> b = a;
    b[k] = (b[k] + 1) % (int)V;
    const std::vector<double>& lb = m.forward(b.data(), t);
    double before = 0.0, after = 0.0;
    for (size_t i = 0; i < t; ++i)
        for (size_t j = 0; j < V; ++j) {
            double dlt = std::fabs(la[i * V + j] - lb[i * V + j]);
            if (i < k) before = std::max(before, dlt); else after = std::max(after, dlt);
        }
    expect(before < 1e-12, "Aenderung an Position " + std::to_string(k) + " veraendert fruehere Positionen nicht");
    expect(after > 1e-6,   "... veraendert aber die Position selbst und spaetere");

    // Kuerzere Sequenz = Anfang der laengeren
    const size_t tp = k + 1;
    const std::vector<double>& lc = m.forward(a.data(), tp);
    double dmax = 0.0;
    for (size_t i = 0; i < tp; ++i)
        for (size_t j = 0; j < V; ++j) dmax = std::max(dmax, std::fabs(la[i * V + j] - lc[i * V + j]));
    expect(dmax < 1e-12, "Logits einer Teilsequenz gleichen dem Anfang der vollen Sequenz");
}

static void check_init_loss_and_overfit(const TransformerConfig& cfg) {
    std::printf("Verlust bei Initialisierung und Auswendiglernen\n");
    Transformer m(cfg);
    m.init(42);
    std::vector<int> ids, tgt;
    make_data(cfg, 9, cfg.context, ids, tgt);
    const double l0 = loss_only(m, ids, tgt);
    const double lnV = std::log((double)cfg.vocab);
    std::printf("  Anfangsverlust %.4f, ln(vocab) = %.4f\n", l0, lnV);
    expect(std::fabs(l0 - lnV) < 0.7, "Anfangsverlust liegt nahe ln(vocab) (Gleichverteilung)");

    Adam opt(m.params.size(), 0.01);
    for (int s = 0; s < 400; ++s) {
        m.zero_grads();
        m.step_loss_and_grad(ids.data(), tgt.data(), ids.size());
        opt.step(m.params, m.grads);
    }
    const double l1 = loss_only(m, ids, tgt);
    std::printf("  Verlust nach 400 Adam-Schritten: %.5f\n", l1);
    expect(l1 < 0.05 * l0, "Adam lernt eine feste Sequenz auswendig (Verlust < 5 % des Anfangswerts)");
}

static void check_validation() {
    std::printf("Konfigurationspruefung\n");
    auto throws = [](TransformerConfig c) {
        try { Transformer m(c); } catch (const std::runtime_error&) { return true; }
        return false;
    };
    TransformerConfig c; c.vocab = 8; c.dim = 6; c.context = 4; c.heads = 4; c.layers = 1;
    expect(throws(c), "dim=6, heads=4 (nicht teilbar) wird abgelehnt");
    c.heads = 3; expect(!throws(c), "dim=6, heads=3 ist gueltig");
    c.layers = 0;  expect(throws(c), "layers=0 wird abgelehnt");
    c.layers = 1; c.context = 0; expect(throws(c), "context=0 wird abgelehnt");

    TransformerConfig ok; ok.vocab = 5; ok.dim = 4; ok.context = 3; ok.heads = 2; ok.layers = 1;
    Transformer m(ok); m.init(1);
    std::vector<int> ids = {0, 1, 2, 3};   // 4 > context
    bool thrown = false;
    try { m.forward(ids.data(), ids.size()); } catch (const std::runtime_error&) { thrown = true; }
    expect(thrown, "Sequenz laenger als context wird abgelehnt");
    std::vector<int> bad = {0, 9};
    thrown = false;
    try { m.forward(bad.data(), bad.size()); } catch (const std::runtime_error&) { thrown = true; }
    expect(thrown, "Token-ID ausserhalb des Vokabulars wird abgelehnt");
}

// Mehrere Threads rechnen gleichzeitig mit je eigenem Workspace und Gradientenpuffer auf demselben
// (nur gelesenen) Modell. Ergebnis muss bitgleich zur sequentiellen Rechnung sein.
static void check_thread_safety() {
    std::printf("Thread-Sicherheit (gleichzeitige Durchlaeufe auf einem Modell)\n");
    TransformerConfig c; c.vocab = 13; c.dim = 16; c.context = 9; c.heads = 2; c.layers = 2;
    Transformer m = make_model(c, 21);
    const size_t n_seq = 24, t = 9, P = m.params.size();
    std::vector<std::vector<int>> ids(n_seq), tgt(n_seq);
    for (size_t i = 0; i < n_seq; ++i) make_data(c, 100 + (unsigned)i, t, ids[i], tgt[i]);

    std::vector<std::vector<double>> seq_g(n_seq, std::vector<double>(P, 0.0));
    std::vector<double> seq_l(n_seq);
    { Transformer::Workspace ws;
      for (size_t i = 0; i < n_seq; ++i) seq_l[i] = m.step_loss_and_grad(ids[i].data(), tgt[i].data(), t, ws, seq_g[i].data()); }

    std::vector<std::vector<double>> par_g(n_seq, std::vector<double>(P, 0.0));
    std::vector<double> par_l(n_seq);
    std::vector<Transformer::Workspace> ws(8);
    parallel_slots(n_seq, 8, [&](size_t i) {
        par_l[i] = m.step_loss_and_grad(ids[i].data(), tgt[i].data(), t, ws[i % 8], par_g[i].data());
    });

    bool same = true;
    for (size_t i = 0; i < n_seq; ++i) {
        if (std::memcmp(&seq_l[i], &par_l[i], sizeof(double)) != 0) same = false;
        if (std::memcmp(seq_g[i].data(), par_g[i].data(), P * sizeof(double)) != 0) same = false;
    }
    expect(same, "24 Sequenzen auf 8 Threads: Loss und Gradienten bitgleich zur sequentiellen Rechnung");
}

int main() {
    TransformerConfig a; a.vocab = 7; a.dim = 8;  a.context = 5; a.heads = 2; a.layers = 2;
    check_gradients(a, 5, "2 Schichten, 2 Koepfe, volle Laenge");
    check_gradients(a, 3, "2 Schichten, 2 Koepfe, kuerzere Sequenz");

    TransformerConfig b; b.vocab = 6; b.dim = 6; b.context = 4; b.heads = 3; b.layers = 1;
    check_gradients(b, 4, "1 Schicht, 3 Koepfe");

    TransformerConfig c; c.vocab = 5; c.dim = 4; c.context = 3; c.heads = 1; c.layers = 1;
    check_gradients(c, 3, "1 Schicht, 1 Kopf");

    TransformerConfig d; d.vocab = 11; d.dim = 8; d.context = 6; d.heads = 2; d.layers = 2;
    check_causality(d);
    check_init_loss_and_overfit(d);
    check_validation();
    check_thread_safety();

    if (failures) { std::printf("\n%d Pruefung(en) fehlgeschlagen\n", failures); return 1; }
    std::printf("\nAlle Pruefungen bestanden\n");
    return 0;
}
