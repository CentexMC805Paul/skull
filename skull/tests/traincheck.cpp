// ============================================================
//  Test fuer den Trainingsrahmen in src/trainer.h
//
//  Prueft mit einem Mock-Modell, dessen Validierungs-Loss vorgegeben ist,
//  genau und deterministisch:
//    - welches Modell gespeichert wird (bestes Validierungs-Modell, nicht das letzte)
//    - Early Stopping (patience)
//    - Aufteilung in Training und Validierung (split_data)
//  Das Echtzeit-Verhalten auf echten Daten (Ueberanpassung usw.) zeigen die
//  Skript-Tests; hier geht es um die Logik, die nicht vom Zufall abhaengen darf.
//
//  Exit-Code 0 = alles in Ordnung.
// ============================================================
#include <cstdio>
#include <cmath>
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

    if (failures) { std::printf("\n%d Pruefung(en) fehlgeschlagen\n", failures); return 1; }
    std::printf("\nAlle Pruefungen bestanden\n");
    return 0;
}
