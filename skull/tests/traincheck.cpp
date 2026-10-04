// ============================================================
//  Test fuer den Trainingsrahmen in src/trainer.h
//
//  Prueft mit einem Mock-Modell, dessen Validierungs-Loss vorgegeben ist,
//  genau und deterministisch:
//    - welches Modell gespeichert wird (bestes Validierungs-Modell, nicht das letzte)
//    - Early Stopping (patience)
//    - Aufteilung in Training und Validierung (split_data)
//    - Parallelisierung (threads.h, Slots) muss exakt und thread-unabhaengig sein
//    - Checkpoints: pausieren + fortsetzen == durchgehender Lauf (bitgleich), kaputte oder
//      unpassende Checkpoints werden abgelehnt
//  Das Echtzeit-Verhalten auf echten Daten (Ueberanpassung usw.) zeigen die
//  Skript-Tests; hier geht es um die Logik, die nicht vom Zufall abhaengen darf.
//
//  Exit-Code 0 = alles in Ordnung.
// ============================================================
#include <cstdio>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
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
    size_t param_count() const override { return 1234; }
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
    uint32_t kind() const override { return ckpt::KIND_BIGRAM; }
    void save_state(std::ostream& o) const override { weights_detail::put_u64(o, (uint64_t)epoch_); }
    void load_state(std::istream& in) override {
        uint64_t e = 0;
        if (!weights_detail::get_u64(in, e)) throw std::runtime_error("Mock: Zustand unlesbar");
        epoch_ = (size_t)e;
    }
private:
    std::vector<double> val_;
    size_t epoch_ = 0;
};

static std::string tmp_path(const char* name) { return std::string("traincheck_") + name + ".weights"; }
static bool file_exists(const std::string& p) { std::ifstream f(p, std::ios::binary); return (bool)f; }
static std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::ostringstream os;
    os << f.rdbuf();
    return os.str();
}
static void spit(const std::string& p, const std::string& bytes) {
    std::ofstream f(p, std::ios::binary);
    f.write(bytes.data(), (std::streamsize)bytes.size());
}

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

// ---- Checkpoints mit dem Mock-Modell: Zustand des Rahmens ----
static void check_checkpoints_mock() {
    std::printf("Checkpoints: Zustand des Trainingsrahmens\n");
    const std::vector<double> curve = {3.0, 2.0, 1.0, 1.5, 1.2, 1.3, 1.4, 1.45};   // Bestwert in Epoche 3
    const std::vector<int> val_ids = {1, 2, 3, 4};
    const uint64_t H = 777;

    // Durchgehender Lauf als Referenz
    TrainConfig full; full.epochs = 8; full.patience = 0; full.out_path = tmp_path("ck_full");
    MockModel m0(curve);
    TrainResult r0 = run_training(full, m0, val_ids, full.out_path, H);

    // Pause nach Epoche 5, danach fortsetzen
    TrainConfig part = full; part.out_path = tmp_path("ck_part"); part.stop_after = 5;
    MockModel m1(curve);
    TrainResult r1 = run_training(part, m1, val_ids, part.out_path, H);
    expect(r1.paused && r1.epochs_run == 5, "stop_after = 5: nach Epoche 5 pausiert");
    expect(file_exists(part.out_path + ".ckpt"), "Checkpoint wurde geschrieben");
    expect(load_weights(part.out_path).dim == 3, "Zwischengewichte = bestes Modell bis dahin (Epoche 3)");

    TrainConfig cont = part; cont.stop_after = 0; cont.resume_path = part.out_path + ".ckpt";
    MockModel m2(curve);
    TrainResult r2 = run_training(cont, m2, val_ids, cont.out_path, H);
    expect(!r2.paused && r2.epochs_run == 8, "fortgesetzt bis Epoche 8");
    expect(r2.best_epoch == r0.best_epoch && r2.val_loss == r0.val_loss && r2.train_loss == r0.train_loss,
           "Ergebnis (beste Epoche, Val-Loss, Train-Loss) identisch zum durchgehenden Lauf");
    expect(r2.best_epoch == 3, "das beste Modell vor der Pause (Epoche 3) bleibt erhalten");
    expect(slurp(full.out_path) == slurp(cont.out_path), "gespeicherte Gewichte byte-identisch zum durchgehenden Lauf");

    // Geduld ueber die Pause hinweg: Epoche 4 und 5 schlechter (vor der Pause), 6 wieder (nach der Pause)
    TrainConfig pp = full; pp.patience = 3; pp.stop_after = 5; pp.out_path = tmp_path("ck_pat");
    MockModel m3(curve);
    TrainResult rp1 = run_training(pp, m3, val_ids, pp.out_path, H);
    TrainConfig pc = pp; pc.stop_after = 0; pc.resume_path = pp.out_path + ".ckpt";
    MockModel m4(curve);
    TrainResult rp2 = run_training(pc, m4, val_ids, pc.out_path, H);
    TrainConfig pref = full; pref.patience = 3; pref.out_path = tmp_path("ck_patref");
    MockModel m5(curve);
    TrainResult rref = run_training(pref, m5, val_ids, pref.out_path, H);
    expect(rp1.paused && rp2.stopped_early && rp2.epochs_run == rref.epochs_run && rref.epochs_run == 6,
           "Early Stopping greift nach der Pause genauso wie im durchgehenden Lauf (Epoche 6)");

    // periodische Checkpoints
    TrainConfig per = full; per.checkpoint = 3; per.out_path = tmp_path("ck_per");
    MockModel m6(curve);
    run_training(per, m6, val_ids, per.out_path, H);
    expect(file_exists(per.out_path + ".ckpt"), "checkpoint = 3: Checkpoint vorhanden (auch am Ende)");

    std::printf("Checkpoints: Ablehnung unpassender oder kaputter Dateien\n");
    auto fails_with = [&](TrainConfig c, uint64_t hash, const char* needle) {
        MockModel m(curve);
        try { run_training(c, m, val_ids, tmp_path("ck_never"), hash); }
        catch (const std::runtime_error& e) { return std::string(e.what()).find(needle) != std::string::npos; }
        return false;
    };
    TrainConfig bad = full; bad.resume_path = part.out_path + ".ckpt";
    expect(fails_with(bad, H + 1, "passt nicht zu den aktuellen Daten"), "andere Daten (Hash) -> abgelehnt");
    bad.resume_path = tmp_path("ck_gibtsnicht") + ".ckpt";
    expect(fails_with(bad, H, "nicht gefunden"), "fehlende Datei -> abgelehnt");
    const std::string good = slurp(part.out_path + ".ckpt");
    spit(tmp_path("ck_trunc") + ".ckpt", good.substr(0, good.size() / 2));
    bad.resume_path = tmp_path("ck_trunc") + ".ckpt";
    expect(fails_with(bad, H, "abgeschnitten"), "abgeschnittene Datei -> abgelehnt");
    spit(tmp_path("ck_junk") + ".ckpt", "das ist kein checkpoint");
    bad.resume_path = tmp_path("ck_junk") + ".ckpt";
    expect(fails_with(bad, H, "Magic"), "Datei ohne Magic -> abgelehnt");

    for (const char* n : {"ck_full", "ck_part", "ck_pat", "ck_patref", "ck_per"}) {
        std::remove(tmp_path(n).c_str());
        std::remove((tmp_path(n) + ".ckpt").c_str());
    }
    std::remove((tmp_path("ck_trunc") + ".ckpt").c_str());
    std::remove((tmp_path("ck_junk") + ".ckpt").c_str());
}

// ---- Pausieren + Fortsetzen mit ECHTEN Modellen == durchgehender Lauf (bitgleich) ----
static std::string real_data_file() {
    const std::string path = "traincheck_data.txt";
    std::ofstream f(path);
    std::string text;
    for (int i = 0; i < 60; ++i)
        text += "zeile " + std::to_string(i % 7) + ": das ist ein kleiner test fuer skull, nummer " + std::to_string(i * 13 % 10) + ".\n";
    f << text;
    return path;
}

static void check_resume_real_models() {
    std::printf("Pausieren + Fortsetzen == durchgehender Lauf (echte Modelle, bitgleich)\n");
    const std::string data = real_data_file();

    struct Variant { const char* name; size_t context; int threads; int batch; unsigned seed; size_t steps; };
    const Variant variants[] = {
        {"Bigram",                              1, 1, 1, 42, 0},
        {"Bigram, batch 4, seed 7",             1, 1, 4, 7,  0},
        {"Bigram mit steps (zufaellige Fenster)", 1, 1, 2, 11, 150},
        {"Transformer, 1 Thread",               8, 1, 4, 42, 0},
        {"Transformer, 3 Threads, seed 9",      8, 3, 6, 9,  0},
        {"Transformer mit steps",               8, 2, 3, 5,  200},
    };
    for (const auto& v : variants) {
        TrainConfig base;
        base.data_path = data; base.epochs = 6; base.rate = 0.01; base.dim = 16; base.vocab = 256;
        base.context = v.context; base.heads = v.context > 1 ? 2 : 0; base.layers = v.context > 1 ? 2 : 0;
        base.threads = v.threads; base.batch = v.batch; base.seed = v.seed; base.val = 0.1; base.steps = v.steps;

        TrainConfig a = base; a.out_path = "traincheck_a.weights";
        skull_train(a);

        TrainConfig b = base; b.out_path = "traincheck_b.weights"; b.stop_after = 3;
        TrainResult rb = skull_train(b);
        TrainConfig c = base; c.out_path = "traincheck_b.weights"; c.resume_path = "traincheck_b.weights.ckpt";
        skull_train(c);

        expect(rb.paused, std::string(v.name) + ": nach Epoche 3 pausiert");
        expect(slurp(a.out_path) == slurp(c.out_path),
               std::string(v.name) + ": Gewichte nach Pause + Fortsetzen byte-identisch zum durchgehenden Lauf");
    }

    // Seed: gleicher Seed = gleiche Gewichte, anderer Seed = andere
    {
        TrainConfig base;
        base.data_path = data; base.epochs = 2; base.rate = 0.01; base.dim = 16; base.context = 8; base.heads = 2;
        base.layers = 1; base.batch = 2; base.val = 0.0;
        TrainConfig s1 = base; s1.seed = 5;  s1.out_path = "traincheck_s1.weights";
        TrainConfig s2 = base; s2.seed = 5;  s2.out_path = "traincheck_s2.weights";
        TrainConfig s3 = base; s3.seed = 6;  s3.out_path = "traincheck_s3.weights";
        skull_train(s1); skull_train(s2); skull_train(s3);
        expect(slurp(s1.out_path) == slurp(s2.out_path), "gleicher seed -> byte-identische Gewichte");
        expect(slurp(s1.out_path) != slurp(s3.out_path), "anderer seed -> andere Gewichte");
    }

    // Checkpoint eines anderen Modelltyps / anderer Daten wird abgelehnt
    {
        TrainConfig base;
        base.data_path = data; base.epochs = 4; base.rate = 0.01; base.dim = 16; base.context = 8; base.heads = 2;
        base.layers = 1; base.val = 0.1; base.out_path = "traincheck_t.weights"; base.stop_after = 2;
        skull_train(base);
        auto rejected = [&](TrainConfig c, const char* needle) {
            try { skull_train(c); } catch (const std::runtime_error& e) { return std::string(e.what()).find(needle) != std::string::npos; }
            return false;
        };
        TrainConfig bigram = base; bigram.context = 1; bigram.stop_after = 0; bigram.out_path = "traincheck_t2.weights";
        bigram.resume_path = "traincheck_t.weights.ckpt";
        expect(rejected(bigram, "gehoert zu einem Transformer"), "Transformer-Checkpoint fuer ein Bigram-Training -> abgelehnt");
        TrainConfig other = base; other.stop_after = 0; other.val = 0.0; other.out_path = "traincheck_t3.weights";
        other.resume_path = "traincheck_t.weights.ckpt";
        expect(rejected(other, "passt nicht zu den aktuellen Daten"), "anderer val-Anteil (andere Aufteilung) -> abgelehnt");
        TrainConfig dims = base; dims.stop_after = 0; dims.dim = 24; dims.out_path = "traincheck_t4.weights";
        dims.resume_path = "traincheck_t.weights.ckpt";
        expect(rejected(dims, "Parameterzahl"), "andere Modellgroesse -> abgelehnt");
    }

    for (const char* n : {"a", "b", "s1", "s2", "s3", "t", "t2", "t3", "t4"}) {
        std::remove((std::string("traincheck_") + n + ".weights").c_str());
        std::remove((std::string("traincheck_") + n + ".weights.ckpt").c_str());
    }
    std::remove(data.c_str());
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
        expect(r.params == 1234, "params = param_count() des Modells");
        expect(r.seconds >= 0.0 && r.seconds < 60.0, "seconds ist die Rechenzeit (hier winzig, nie negativ)");
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
    check_checkpoints_mock();
    check_resume_real_models();

    if (failures) { std::printf("\n%d Pruefung(en) fehlgeschlagen\n", failures); return 1; }
    std::printf("\nAlle Pruefungen bestanden\n");
    return 0;
}
