// ============================================================
//  Test fuer den Trainingsrahmen in src/trainer.h
//
//  Prueft mit einem Mock-Modell, dessen Validierungs-Loss vorgegeben ist,
//  genau und deterministisch:
//    - welches Modell gespeichert wird (bestes Validierungs-Modell, nicht das letzte)
//    - Early Stopping (patience)
//    - Aufteilung in Training und Validierung (split_data)
//  Dazu: Parallelisierung (threads.h, Slots) muss exakt und thread-unabhaengig sein.
//  Das Echtzeit-Verhalten auf echten Daten (Ueberanpassung usw.) zeigen die
//  Skript-Tests; hier geht es um die Logik, die nicht vom Zufall abhaengen darf.
//
//  Exit-Code 0 = alles in Ordnung.
// ============================================================
#include <cstdio>
#include <cmath>
#include <cstring>
#include <atomic>
#include <vector>
#include <string>
#include "trainer.h"

static int failures = 0;

static void expect(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "ok" : "FEHLER", what.c_str());
    if (!ok) ++failures;
}

// Mock: Trainings-Loss 1/Epoche, Validierungs-Loss aus einer vorgegebenen Folge.
// Die "Gewichte" tragen die Epochennummer als dim, daran erkennt man, welche gespeichert wurden.
class MockModel : public TrainableModel {
public:
    explicit MockModel(std::vector<double> val_seq) : val_(std::move(val_seq)) {}
    void print_info(size_t) const override {}
    double train_epoch() override { ++epoch_; return 1.0 / (double)epoch_; }
    double evaluate(const std::vector<int>&) override { return val_.at(epoch_ - 1); }
    SkullWeights export_weights() const override {
        SkullWeights w;
        w.dim   = epoch_;
        w.vocab = 1;
        w.W_embed.assign(w.vocab * w.dim, 0.0);
        w.W_hidden.assign(w.dim * w.dim, 0.0);
        w.W_out.assign(w.dim * w.vocab, 0.0);
        return w;
    }
private:
    std::vector<double> val_;
    size_t epoch_ = 0;
};

static std::string tmp_path(const char* name) { return std::string("traincheck_") + name + ".weights"; }

static TrainResult run(int epochs, int patience, std::vector<double> val_seq, bool with_val,
                       const char* name, size_t& saved_dim) {
    TrainConfig cfg;
    cfg.epochs   = epochs;
    cfg.patience = patience;
    MockModel m(std::move(val_seq));
    std::vector<int> val_ids;
    if (with_val) val_ids = {1, 2, 3, 4};
    const std::string path = tmp_path(name);
    TrainResult r = run_training(cfg, m, val_ids, path);
    saved_dim = load_weights(path).dim;
    std::remove(path.c_str());
    return r;
}


static bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

// Synthetische Token-Folge mit Struktur (damit der Loss sinkt)
static std::vector<int> synthetic_ids(size_t n, int vocab) {
    std::vector<int> ids(n);
    for (size_t i = 0; i < n; ++i) ids[i] = (int)((i * 7 + (i / 5) * 3) % (size_t)vocab);
    return ids;
}

static void check_parallel_slots() {
    std::printf("parallel_slots\n");
    {
        bool once = true;
        for (size_t threads : {1u, 2u, 3u, 7u, 64u}) {
            std::vector<std::atomic<int>> hits(23);
            for (auto& h : hits) h = 0;
            parallel_slots(hits.size(), threads, [&](size_t sl) { ++hits[sl]; });
            for (auto& h : hits) if (h != 1) once = false;
        }
        expect(once, "jeder Slot wird genau einmal ausgefuehrt (1, 2, 3, 7, 64 Threads)");
    }
    {
        bool threw = false;
        try {
            parallel_slots(16, 4, [&](size_t sl) { if (sl == 9) throw std::runtime_error("boom"); });
        } catch (const std::runtime_error& e) { threw = std::string(e.what()) == "boom"; }
        expect(threw, "Ausnahme aus einem Thread wird nach dem Join weitergereicht");
    }
    {
        bool ok = true;
        parallel_slots(0, 4, [&](size_t) { ok = false; });
        expect(ok, "0 Slots: nichts wird ausgefuehrt");
    }
}

static void check_adam_chunks() {
    std::printf("Adam: parallele Abschnitte == sequentiell (bitgleich)\n");
    const size_t n = 100003;   // bewusst kein Vielfaches der Abschnittsgroesse
    std::vector<double> p0(n), g(n);
    for (size_t i = 0; i < n; ++i) { p0[i] = std::sin(0.001 * i); g[i] = std::cos(0.0007 * i) * 3.0; }
    auto run = [&](size_t threads) {
        std::vector<double> p = p0;
        Adam opt(n, 0.01, 1.0);
        for (int step = 0; step < 5; ++step)
            opt.step(p, g, 0.5, [&](size_t nc, const std::function<void(size_t)>& fn) { parallel_slots(nc, threads, fn); });
        return p;
    };
    const std::vector<double> a = run(1), b = run(4), c = run(7);
    expect(same_bits(a, b) && same_bits(a, c), "1, 4 und 7 Threads liefern dieselben Parameter");
    std::vector<double> d = p0;
    Adam seq(n, 0.01, 1.0);
    for (int step = 0; step < 5; ++step) seq.step(d, g, 0.5);
    expect(same_bits(a, d), "identisch zur Variante ohne run_chunks");
    expect(!same_bits(a, p0), "Parameter haben sich veraendert");
}

static void check_trainer_threads() {
    std::printf("Transformer-Training: Ergebnis unabhaengig von der Thread-Anzahl\n");
    const std::vector<int> ids = synthetic_ids(1200, 40);
    std::vector<int> val(ids.begin() + 1000, ids.end());
    std::vector<int> train(ids.begin(), ids.begin() + 1000);

    auto run = [&](int threads, int batch) {
        TrainConfig cfg;
        cfg.epochs = 3; cfg.rate = 0.01; cfg.batch = batch; cfg.dim = 16; cfg.context = 12;
        cfg.heads = 2; cfg.layers = 2; cfg.threads = threads; cfg.vocab = 40;
        TransformerConfig tc = make_transformer_config(cfg, 40, train.size() - 1);
        TransformerTrainer t(cfg, tc, train, nullptr);
        std::vector<double> losses;
        for (int e = 0; e < cfg.epochs; ++e) losses.push_back(t.train_epoch());
        losses.push_back(t.evaluate(val));
        return std::make_pair(t.export_weights().tparams, losses);
    };
    for (int batch : {8, 5, 3}) {
        auto r1 = run(1, batch), r2 = run(2, batch), r3 = run(3, batch), r8 = run(8, batch);
        const bool w = same_bits(r1.first, r2.first) && same_bits(r1.first, r3.first) && same_bits(r1.first, r8.first);
        const bool l = same_bits(r1.second, r2.second) && same_bits(r1.second, r3.second) && same_bits(r1.second, r8.second);
        expect(w, "batch " + std::to_string(batch) + ": Gewichte bitgleich bei 1, 2, 3 und 8 Threads");
        expect(l, "batch " + std::to_string(batch) + ": Trainings- und Validierungs-Loss bitgleich");
    }
    auto r = run(2, 8);
    expect(r.second.front() > r.second[2], "der Loss sinkt waehrend des Trainings");
}

int main() {
    // Validierungs-Loss: faellt bis Epoche 3 (1.0), steigt danach wieder (Ueberanpassung)
    const std::vector<double> curve = {3.0, 2.0, 1.0, 1.5, 1.2, 1.3, 1.4, 1.45};

    std::printf("Bestes Modell wird gespeichert (ohne Early Stopping)\n");
    {
        size_t saved = 0;
        TrainResult r = run(8, 0, curve, true, "best", saved);
        expect(r.has_val, "Validierung war aktiv");
        expect(r.best_epoch == 3, "beste Epoche = 3");
        expect(r.val_loss == 1.0, "val_loss = Wert des besten Modells (1.0), nicht der letzten Epoche");
        expect(r.epochs_run == 8 && !r.stopped_early, "alle 8 Epochen gelaufen");
        expect(saved == 3, "gespeichert wurde das Modell aus Epoche 3, nicht das letzte");
        expect(std::fabs(r.train_loss - 1.0 / 8.0) < 1e-15, "train_loss = Trainings-Loss der LETZTEN Epoche");
    }

    std::printf("Early Stopping (patience)\n");
    {
        size_t saved = 0;
        TrainResult r = run(8, 3, curve, true, "patience", saved);
        expect(r.stopped_early, "Abbruch durch Early Stopping");
        expect(r.epochs_run == 6, "Abbruch nach Epoche 6 (Epoche 4, 5, 6 ohne Verbesserung)");
        expect(r.best_epoch == 3 && saved == 3, "bestes Modell (Epoche 3) wird trotzdem gespeichert");
    }
    {
        // Verbesserung setzt den Zaehler zurueck: kein Abbruch trotz patience = 2
        const std::vector<double> zigzag = {3.0, 2.5, 2.6, 2.4, 2.5, 2.3, 2.4, 2.2};
        size_t saved = 0;
        TrainResult r = run(8, 2, zigzag, true, "zigzag", saved);
        expect(!r.stopped_early && r.epochs_run == 8, "Verbesserung setzt die Geduld zurueck (kein Abbruch)");
        expect(r.best_epoch == 8 && saved == 8, "bestes Modell ist das der letzten Epoche");
    }
    {
        // Letzte Epoche ohne Verbesserung: es gibt nichts mehr abzubrechen
        const std::vector<double> late_bad = {3.0, 2.0, 1.0, 1.1};
        size_t saved = 0;
        TrainResult r = run(4, 1, late_bad, true, "lastbad", saved);
        expect(!r.stopped_early && r.epochs_run == 4, "Abbruch erst vor der letzten Epoche moeglich");
        expect(saved == 3, "bestes Modell trotzdem gespeichert");
    }

    std::printf("Ohne Validierung\n");
    {
        size_t saved = 0;
        TrainResult r = run(5, 3, {}, false, "noval", saved);
        expect(!r.has_val && std::isnan(r.val_loss), "has_val = false, val_loss ist NaN");
        expect(r.epochs_run == 5 && !r.stopped_early, "patience wirkt ohne Validierung nicht");
        expect(saved == 5, "gespeichert wird das Modell der letzten Epoche");
    }

    std::printf("Divergenz wird erkannt\n");
    {
        bool threw = false;
        try { size_t s = 0; run(3, 0, {2.0, std::nan(""), 1.0}, true, "nan", s); }
        catch (const std::runtime_error& e) { threw = std::string(e.what()).find("nicht endlich") != std::string::npos; }
        std::remove(tmp_path("nan").c_str());
        expect(threw, "NaN im Validierungs-Loss -> Fehler 'nicht endlich'");
    }

    std::printf("Aufteilung in Training und Validierung\n");
    {
        std::vector<int> ids(1000);
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = (int)i;
        DataSplit s = split_data(ids, 0.1);
        expect(s.val.size() == 100 && s.train.size() == 900 && s.note.empty(), "1000 Token, val 0.1 -> 900 / 100, kein Hinweis");
        expect(s.val.front() == 900 && s.val.back() == 999, "Validierung = die LETZTEN Token der Daten");
        expect(s.train.front() == 0 && s.train.back() == 899, "Training = der Anfang, ohne Ueberlappung");
    }
    {
        std::vector<int> ids(999, 1);
        DataSplit s = split_data(ids, 0.1);
        expect(s.val.empty() && s.train.size() == 999 && !s.note.empty(), "99 Validierungs-Token sind zu wenig -> uebersprungen, mit Hinweis");
    }
    {
        std::vector<int> ids(5000, 1);
        DataSplit s = split_data(ids, 0.0);
        expect(s.val.empty() && s.train.size() == 5000 && s.note.empty(), "val = 0 -> keine Validierung, kein Hinweis");
    }
    {
        std::vector<int> ids(1000000, 1);
        DataSplit s = split_data(ids, 0.1);
        expect(s.val.size() == 50000 && s.train.size() == 950000, "grosse Daten: Validierung auf 50000 Token begrenzt, Rest wird trainiert");
    }
    {
        std::vector<int> ids(120, 1);
        DataSplit s = split_data(ids, 0.5);
        expect(s.val.empty() && s.train.size() == 120 && !s.note.empty(),
               "val 0.5 bei 120 Token waeren nur 60 Validierungs-Token -> uebersprungen, alles wird trainiert");
    }

    check_parallel_slots();
    check_adam_chunks();
    check_trainer_threads();

    if (failures) { std::printf("\n%d Pruefung(en) fehlgeschlagen\n", failures); return 1; }
    std::printf("\nAlle Pruefungen bestanden\n");
    return 0;
}
