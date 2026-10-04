#pragma once
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include "tensor.h"
#include "tokenizer.h"
#include "gpu.h"
#include "weights.h"
#include "transformer.h"
#include "rng.h"
#include "threads.h"
#include "version.h"

// ============================================================
//  SKULL TRAINER
//
//  Zwei Modelle, gewaehlt ueber das Feld `context`:
//
//  context = 1 (Standard): Bigram-Modell. Embedding -> Hidden (ReLU) ->
//    Softmax. Es sieht pro Schritt genau EIN Token (das vorige) und sagt
//    das naechste voraus. Training mit SGD (`rate` = SGD-Lernrate).
//
//  context > 1: Transformer mit kausaler Attention (src/transformer.h).
//    Sieht bis zu `context` vorige Token. Felder heads, layers, dim.
//    Training mit Adam (`rate` = Spitzen-Lernrate, typisch 0.001 - 0.01; kurzer
//    Warmup, danach Cosine-Abfall auf 10 %).
//
//  Forward und Backward sind in beiden Faellen von Hand ausgeschrieben und
//  arbeiten direkt auf Vektoren (kein Autograd-Graph pro Schritt).
//  Das haelt den Speicher konstant und die Schritte schnell.
//
//  Aufbau dieser Datei:
//    TrainableModel   gemeinsame Schnittstelle (Epoche trainieren, bewerten, Gewichte holen)
//    BigramModel, TransformerTrainer   die beiden Modelle
//    run_training()   gemeinsamer Rahmen: Epochen, Validierung, bestes Modell, Early Stopping
//    skull_train()    Eingaben pruefen, Daten laden/aufteilen, Modell waehlen
//
//  Optionen (train { ... }):
//    data, out, epochs, rate, batch, dim, vocab, steps, context, heads, layers,
//    val, patience, threads, bpe, bpe_vocab, gpu, prefer_amd
//  out = Pfad der Gewichte-Datei (Standard: data + ".weights").
//  steps = 0 (Standard): jede Epoche geht durch ALLE Tokens.
//  steps > 0: pro Epoche nur so viele Token-Schritte (zufaellige Fenster),
//  damit bei grossen Dateien nach und nach alles gesehen wird, ohne dass
//  eine Epoche ewig dauert.
//  val = Anteil der Daten (vom ENDE der Datei), der nicht trainiert, sondern zum
//  Bewerten benutzt wird (Standard 0.1; 0 = aus; hoechstens 50000 Token, der Rest
//  wird trainiert). Mit Validierung werden die Gewichte mit dem besten
//  Validierungs-Loss gespeichert.
//  patience = Abbruch, wenn sich der Validierungs-Loss N Epochen lang nicht
//  verbessert (0 = aus).
//  threads = Anzahl Rechen-Threads des Transformers (0 = alle Kerne). Parallel laufen die
//  Sequenzen eines Batches (batch > 1) und die Validierung. Das Ergebnis ist unabhaengig von
//  der Thread-Anzahl bitgleich (feste Slot-Zuordnung, siehe threads.h).
// ============================================================

inline void softmax_inplace(FlatVec& v) {
    double maxv = *std::max_element(v.begin(), v.end());
    double sum = 0.0;
    for (auto& x : v) { x = std::exp(x - maxv); sum += x; }
    for (auto& x : v) x /= sum;
}

inline double cross_entropy(const FlatVec& probs, int target_id) {
    if (target_id < 0 || (size_t)target_id >= probs.size()) return 0.0;
    return -std::log(std::max(probs[target_id], 1e-10));
}

// Xavier-Initialisierung, identische Zufallsfolge wie tensor_rand(r, c, seed)
// (und dank rng.h auf jedem Betriebssystem dieselbe).
inline FlatVec xavier_init(size_t rows, size_t cols, unsigned seed) {
    FlatVec v(rows * cols);
    double limit = std::sqrt(6.0 / (double)(rows + cols));
    std::mt19937 rng(seed);
    for (auto& x : v) x = rng_uniform(rng, -limit, limit);
    return v;
}

// M[i*cols + j] += s * u[i] * v[j]   (Zeilen mit u[i] == 0 werden uebersprungen)
inline void outer_add(FlatVec& M, size_t cols, const FlatVec& u, const FlatVec& v, double s) {
    for (size_t i = 0; i < u.size(); ++i) {
        double a = s * u[i];
        if (a == 0.0) continue;
        double* row = &M[i * cols];
        for (size_t j = 0; j < cols; ++j) row[j] += a * v[j];
    }
}

struct TrainConfig {
    std::string data_path  = "";
    std::string out_path   = "";      // Ziel der Gewichte; leer = data + ".weights"
    int         epochs     = 10;
    double      rate       = 0.001;
    int         batch      = 1;       // Tokens (Bigram) bzw. Sequenzen (Transformer) pro Update
    size_t      dim        = 64;
    size_t      vocab      = 256;
    size_t      steps      = 0;       // 0 = alle Tokens pro Epoche
    size_t      context    = 1;       // 1 = Bigram-Modell, > 1 = Transformer mit diesem Kontext
    size_t      heads      = 0;       // nur Transformer (0 = Standard: 2)
    size_t      layers     = 0;       // nur Transformer (0 = Standard: 1)
    double      val        = 0.1;     // Anteil Validierungsdaten (vom Dateiende), 0 = aus
    int         patience   = 0;       // Early Stopping nach N Epochen ohne Verbesserung, 0 = aus
    int         threads    = 0;       // nur Transformer: Anzahl Threads (0 = alle Kerne); aendert das Ergebnis nicht
    bool        use_bpe    = false;   // BPE-Tokenisierung aktivieren
    int         bpe_vocab  = 1000;    // BPE Ziel-Vokabular
    bool        use_gpu    = false;   // GPU via OpenCL (derzeit nur initialisiert)
    bool        prefer_amd = false;   // AMD GPU bevorzugen
};

// Ergebnis eines Trainings (fuer last_loss() / last_val_loss() in der Sprache)
struct TrainResult {
    double train_loss  = std::numeric_limits<double>::quiet_NaN();  // Trainings-Loss der letzten Epoche
    double val_loss    = std::numeric_limits<double>::quiet_NaN();  // Validierungs-Loss des gespeicherten Modells
    bool   has_val     = false;
    int    best_epoch  = 0;
    int    epochs_run  = 0;
    bool   stopped_early = false;
};

inline std::string weights_path_for(const TrainConfig& cfg) {
    return cfg.out_path.empty() ? cfg.data_path + ".weights" : cfg.out_path;
}

// ---- gemeinsame Schnittstelle der beiden Modelle ----
class TrainableModel {
public:
    virtual ~TrainableModel() = default;
    virtual void print_info(size_t total_tokens) const = 0;
    // Eine Epoche trainieren; liefert den mittleren Trainings-Loss (je Token).
    virtual double train_epoch() = 0;
    // Mittlerer Loss je Token auf ids (nur Vorwaertsrechnung).
    virtual double evaluate(const std::vector<int>& ids) = 0;
    // Aktuelle Gewichte inkl. Tokenizer als speicherbares Paket.
    virtual SkullWeights export_weights() const = 0;
};

// ---- Aufteilen in Training und Validierung ----
struct DataSplit {
    std::vector<int> train;
    std::vector<int> val;      // leer = keine Validierung
    std::string      note;     // Hinweis, falls gewuenschte Validierung nicht moeglich war
};

inline DataSplit split_data(const std::vector<int>& ids, double val_fraction) {
    DataSplit s;
    const size_t MIN_VAL_TOKENS = 100;     // darunter ist ein Validierungs-Loss kaum aussagekraeftig
    const size_t MAX_VAL_TOKENS = 50000;   // darueber lohnt die Genauigkeit den Rechenaufwand pro Epoche nicht
    const size_t n = ids.size();
    size_t n_val = (val_fraction > 0.0) ? (size_t)((double)n * val_fraction) : 0;
    if (n_val > MAX_VAL_TOKENS) n_val = MAX_VAL_TOKENS;
    if (val_fraction > 0.0 && (n_val < MIN_VAL_TOKENS || n - n_val < 2)) {
        s.note = "Validierung uebersprungen: " + std::to_string(n_val) + " von " + std::to_string(n) +
                 " Token waeren zu wenig (mindestens " + std::to_string(MIN_VAL_TOKENS) +
                 " noetig) - es wird auf allen Daten trainiert";
    }
    if (n_val >= MIN_VAL_TOKENS && n - n_val >= 2) {
        s.train.assign(ids.begin(), ids.end() - (std::ptrdiff_t)n_val);
        s.val.assign(ids.end() - (std::ptrdiff_t)n_val, ids.end());
    } else {
        s.train = ids;
    }
    return s;
}

// ============================================================
//  Bigram-Modell
// ============================================================
class BigramModel : public TrainableModel {
public:
    BigramModel(const TrainConfig& cfg, size_t vocab, const std::vector<int>& ids, const BPETokenizer* bpe)
        : cfg_(cfg), ids_(ids), dim_(cfg.dim), vocab_(vocab), window_rng_(12345) {
        n_pairs_      = ids_.size() - 1;
        steps_per_ep_ = (cfg_.steps > 0 && cfg_.steps < n_pairs_) ? cfg_.steps : n_pairs_;

        w_.dim      = dim_;
        w_.vocab    = vocab_;
        w_.W_embed  = xavier_init(vocab_, dim_,  42);
        w_.W_hidden = xavier_init(dim_,   dim_,  43);
        w_.W_out    = xavier_init(dim_,   vocab_, 44);
        if (bpe) { w_.has_bpe = true; w_.bpe = *bpe; }

        embed_.assign(dim_, 0.0); hidden_.assign(dim_, 0.0); d_pre_.assign(dim_, 0.0); d_embed_.assign(dim_, 0.0);
        probs_.assign(vocab_, 0.0); d_logits_.assign(vocab_, 0.0);
        use_batch_ = cfg_.batch > 1;
        if (use_batch_) { gW_out_.assign(dim_ * vocab_, 0.0); gW_hidden_.assign(dim_ * dim_, 0.0); }
    }

    size_t param_count() const { return vocab_ * dim_ + dim_ * dim_ + dim_ * vocab_; }

    void print_info(size_t total_tokens) const override {
        std::cout << "[Skull] Modell:     Bigram (1 Token Kontext)\n";
        std::cout << "[Skull] Tokens:     " << total_tokens << "\n";
        std::cout << "[Skull] Vokabular:  " << vocab_ << "\n";
        std::cout << "[Skull] Schritte/Epoche: " << steps_per_ep_ << "\n";
        std::cout << "[Skull] Parameter:  " << param_count() << "\n\n";
    }

    double train_epoch() override {
        double epoch_loss = 0.0;
        size_t start = 0;
        if (steps_per_ep_ < n_pairs_)
            start = rng_below(window_rng_, n_pairs_ - steps_per_ep_ + 1);

        for (size_t s = 0; s < steps_per_ep_; ++s) {
            const int input_id  = ids_[start + s];
            const int target_id = ids_[start + s + 1];

            forward(input_id);
            epoch_loss += cross_entropy(probs_, target_id);

            // --- Backward (von Hand) ---
            d_logits_ = probs_;
            d_logits_[(size_t)target_id] -= 1.0;

            for (size_t i = 0; i < dim_; ++i) {
                if (hidden_[i] <= 0.0) { d_pre_[i] = 0.0; continue; }
                const double* orow = &w_.W_out[i * vocab_];
                double acc = 0.0;
                for (size_t j = 0; j < vocab_; ++j) acc += orow[j] * d_logits_[j];
                d_pre_[i] = acc;
            }

            std::fill(d_embed_.begin(), d_embed_.end(), 0.0);
            for (size_t i = 0; i < dim_; ++i) {
                const double dp = d_pre_[i];
                if (dp == 0.0) continue;
                const double* hrow = &w_.W_hidden[i * dim_];
                for (size_t j = 0; j < dim_; ++j) d_embed_[j] += hrow[j] * dp;
            }

            // --- Update (SGD) ---
            if (!use_batch_) {
                outer_add(w_.W_out,    vocab_, hidden_, d_logits_, -cfg_.rate);
                outer_add(w_.W_hidden, dim_,   d_pre_,  embed_,    -cfg_.rate);
                double* row = &w_.W_embed[(size_t)input_id * dim_];
                for (size_t d = 0; d < dim_; ++d) row[d] -= cfg_.rate * d_embed_[d];
            } else {
                outer_add(gW_out_,    vocab_, hidden_, d_logits_, 1.0);
                outer_add(gW_hidden_, dim_,   d_pre_,  embed_,    1.0);
                FlatVec& g = gEmbed_[input_id];
                if (g.empty()) g.assign(dim_, 0.0);
                for (size_t d = 0; d < dim_; ++d) g[d] += d_embed_[d];
                if (++in_batch_ == cfg_.batch) flush_batch();
            }
        }
        flush_batch();   // Rest-Batch am Epochenende
        return epoch_loss / (double)steps_per_ep_;
    }

    double evaluate(const std::vector<int>& v) override {
        if (v.size() < 2) return std::numeric_limits<double>::quiet_NaN();
        double total = 0.0;
        for (size_t i = 0; i + 1 < v.size(); ++i) {
            forward(v[i]);
            total += cross_entropy(probs_, v[i + 1]);
        }
        return total / (double)(v.size() - 1);
    }

    SkullWeights export_weights() const override { return w_; }

private:
    // Vorwaerts fuer ein Eingabe-Token; Ergebnis: probs_ (Softmax), hidden_, embed_
    void forward(int input_id) {
        const double* erow = &w_.W_embed[(size_t)input_id * dim_];
        for (size_t d = 0; d < dim_; ++d) embed_[d] = erow[d];

        for (size_t i = 0; i < dim_; ++i) {
            const double* hrow = &w_.W_hidden[i * dim_];
            double pre = 0.0;
            for (size_t j = 0; j < dim_; ++j) pre += hrow[j] * embed_[j];
            hidden_[i] = pre > 0.0 ? pre : 0.0;
        }

        std::fill(probs_.begin(), probs_.end(), 0.0);   // probs_ dient zuerst als Logits
        for (size_t i = 0; i < dim_; ++i) {
            const double h = hidden_[i];
            if (h == 0.0) continue;
            const double* orow = &w_.W_out[i * vocab_];
            for (size_t j = 0; j < vocab_; ++j) probs_[j] += h * orow[j];
        }
        softmax_inplace(probs_);
    }

    void flush_batch() {
        if (in_batch_ == 0) return;
        const double s = cfg_.rate / (double)in_batch_;
        for (size_t k = 0; k < gW_out_.size(); ++k)    w_.W_out[k]    -= s * gW_out_[k];
        for (size_t k = 0; k < gW_hidden_.size(); ++k) w_.W_hidden[k] -= s * gW_hidden_[k];
        for (auto& kv : gEmbed_) {
            double* row = &w_.W_embed[(size_t)kv.first * dim_];
            for (size_t d = 0; d < dim_; ++d) row[d] -= s * kv.second[d];
        }
        std::fill(gW_out_.begin(), gW_out_.end(), 0.0);
        std::fill(gW_hidden_.begin(), gW_hidden_.end(), 0.0);
        gEmbed_.clear();
        in_batch_ = 0;
    }

    TrainConfig cfg_;
    const std::vector<int>& ids_;
    size_t dim_, vocab_;
    size_t n_pairs_ = 0, steps_per_ep_ = 0;
    SkullWeights w_;
    FlatVec embed_, hidden_, d_pre_, d_embed_, probs_, d_logits_;
    bool use_batch_ = false;
    FlatVec gW_out_, gW_hidden_;
    std::unordered_map<int, FlatVec> gEmbed_;
    int in_batch_ = 0;
    std::mt19937 window_rng_;
};

// ============================================================
//  Transformer
// ============================================================

// Prueft die Eingaben und baut die Modellkonfiguration. n_pairs = Trainingspaare.
inline TransformerConfig make_transformer_config(const TrainConfig& cfg, size_t vocab, size_t n_pairs) {
    using namespace weights_detail;
    if (cfg.context > MAX_CONTEXT)
        throw std::runtime_error("train: 'context' darf hoechstens " + std::to_string(MAX_CONTEXT) + " sein");
    if (cfg.layers > MAX_LAYERS)
        throw std::runtime_error("train: 'layers' darf hoechstens " + std::to_string(MAX_LAYERS) + " sein");

    TransformerConfig tc;
    tc.vocab   = vocab;
    tc.dim     = cfg.dim;
    tc.context = std::min(cfg.context, n_pairs);
    tc.heads   = cfg.heads  ? cfg.heads  : 2;
    tc.layers  = cfg.layers ? cfg.layers : 1;
    try {
        tc.validate();
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(std::string("train: ") + e.what());
    }
    if (tc.context < cfg.context)
        std::cout << "[Skull] Hinweis: context wurde auf " << tc.context
                  << " reduziert (die Trainingsdaten haben nur " << n_pairs << " Trainingspaare)\n";

    const size_t n_params = tc.param_count();
    const size_t MAX_PARAMS = 200u * 1000u * 1000u;
    if (n_params > MAX_PARAMS)
        throw std::runtime_error("train: Modell zu gross (" + std::to_string(n_params) +
                                 " Parameter, maximal " + std::to_string(MAX_PARAMS) + ")");
    return tc;
}

// Zufaellige Fenster der Laenge `context`, Adam, Mini-Batch aus mehreren Fenstern.
//
// Parallelisierung: Die Sequenzen eines Batches werden fest auf SLOTS verteilt (Sequenz b gehoert
// zu Slot b % kSlots); jeder Slot hat eigenen Arbeitsspeicher und einen eigenen Gradientenpuffer.
// Die Slots laufen auf bis zu `threads` Threads. Danach werden die Slot-Gradienten in fester
// Reihenfolge addiert. Weil die Slot-Anzahl nur von batch und Modellgroesse abhaengt (nicht von
// der Maschine), ist das Ergebnis unabhaengig von der Thread-Anzahl bitgleich.
class TransformerTrainer : public TrainableModel {
public:
    static constexpr size_t kMaxSlots = 8;

    TransformerTrainer(const TrainConfig& cfg, const TransformerConfig& tc,
                       const std::vector<int>& ids, const BPETokenizer* bpe)
        : cfg_(cfg), tc_(tc), ids_(ids), model_(tc),
          opt_(tc.param_count(), cfg.rate, 1.0), rng_(12345) {
        model_.init(42);
        n_pairs_ = ids_.size() - 1;
        T_       = tc_.context;
        const size_t pairs_per_epoch = (cfg_.steps > 0 && cfg_.steps < n_pairs_) ? cfg_.steps : n_pairs_;
        seqs_per_epoch_    = (pairs_per_epoch + T_ - 1) / T_;
        batch_             = (size_t)cfg_.batch;
        updates_per_epoch_ = (seqs_per_epoch_ + batch_ - 1) / batch_;
        total_updates_     = (size_t)cfg_.epochs * updates_per_epoch_;
        if (bpe) { has_bpe_ = true; bpe_ = *bpe; }

        // Slots: hoechstens 8, nicht mehr als Sequenzen pro Batch, und der Speicher fuer die
        // Gradientenpuffer (Slots x Parameter) bleibt begrenzt. Nur von batch und Modell abhaengig.
        const size_t max_by_memory = std::max<size_t>(1, ((size_t)1 << 28) / tc_.param_count());
        slots_ = std::max<size_t>(1, std::min({kMaxSlots, batch_, max_by_memory}));
        threads_ = cfg_.threads > 0 ? (size_t)cfg_.threads : skull_hardware_threads();
        threads_ = std::min<size_t>(threads_, 64);
        work_.resize(std::max(slots_, kMaxSlots));
        if (slots_ > 1) slot_grads_.assign(slots_, std::vector<double>(tc_.param_count(), 0.0));
    }

    size_t threads() const { return threads_; }

    void print_info(size_t total_tokens) const override {
        std::cout << "[Skull] Modell:     Transformer (" << tc_.layers << " Schicht(en), " << tc_.heads
                  << " Kopf/Koepfe, Kontext " << T_ << ")\n";
        std::cout << "[Skull] Tokens:     " << total_tokens << "\n";
        std::cout << "[Skull] Vokabular:  " << tc_.vocab << "\n";
        std::cout << "[Skull] Sequenzen/Epoche: " << seqs_per_epoch_ << " (je " << T_ << " Token), "
                  << updates_per_epoch_ << " Update(s)\n";
        std::cout << "[Skull] Threads:    " << std::min(threads_, std::max<size_t>(slots_, 1))
                  << " (von " << skull_hardware_threads() << " Kernen; Ergebnis haengt nicht davon ab)\n";
        std::cout << "[Skull] Parameter:  " << tc_.param_count() << "\n\n";
    }

    double train_epoch() override {
        double epoch_loss = 0.0;
        size_t seqs_done = 0;
        std::vector<size_t> starts;
        std::vector<double> slot_loss;
        for (size_t u = 0; u < updates_per_epoch_; ++u) {
            const size_t n_in = std::min(batch_, seqs_per_epoch_ - seqs_done);
            // Fenster in fester Reihenfolge ziehen (unabhaengig von Threads)
            starts.resize(n_in);
            for (size_t b = 0; b < n_in; ++b) starts[b] = rng_below(rng_, n_pairs_ - T_ + 1);

            if (slots_ == 1) {
                model_.zero_grads();
                for (size_t b = 0; b < n_in; ++b)
                    epoch_loss += model_.step_loss_and_grad(&ids_[starts[b]], &ids_[starts[b] + 1], T_,
                                                            work_[0], model_.grads.data());
            } else {
                const size_t used = std::min(slots_, n_in);
                slot_loss.assign(used, 0.0);
                parallel_slots(used, threads_, [&](size_t sl) {
                    std::vector<double>& g = slot_grads_[sl];
                    std::fill(g.begin(), g.end(), 0.0);
                    double l = 0.0;
                    for (size_t b = sl; b < n_in; b += slots_)       // feste Zuordnung Sequenz -> Slot
                        l += model_.step_loss_and_grad(&ids_[starts[b]], &ids_[starts[b] + 1], T_,
                                                       work_[sl], g.data());
                    slot_loss[sl] = l;
                });
                // Slot-Gradienten in fester Reihenfolge addieren (parallel ueber Abschnitte des Vektors;
                // jedes Element wird immer in der Reihenfolge Slot 0, 1, 2, ... summiert)
                const size_t P = model_.grads.size();
                const size_t chunk = 16384, n_chunks = (P + chunk - 1) / chunk;
                const double inv = 1.0 / (double)n_in;
                parallel_slots(n_chunks, threads_, [&](size_t c) {
                    const size_t lo = c * chunk, hi = std::min(P, lo + chunk);
                    for (size_t i = lo; i < hi; ++i) {
                        double acc = slot_grads_[0][i];
                        for (size_t sl = 1; sl < used; ++sl) acc += slot_grads_[sl][i];
                        model_.grads[i] = acc * inv;     // Mittel ueber die Sequenzen des Batches
                    }
                });
                for (size_t sl = 0; sl < used; ++sl) epoch_loss += slot_loss[sl];
            }
            opt_.step(model_.params, model_.grads, lr_schedule(update_no_++, total_updates_),
                      [&](size_t n_chunks, const std::function<void(size_t)>& fn) {
                          parallel_slots(n_chunks, threads_, fn);
                      });
            seqs_done += n_in;
        }
        return epoch_loss / (double)seqs_done;
    }

    // Aufeinanderfolgende Fenster der Laenge `context` (ohne Ueberlappung); jedes Token
    // wird mit hoechstens `context` vorigen Token vorhergesagt. Die Fenster werden fest auf
    // Slots verteilt und die Teilsummen in fester Reihenfolge addiert (threadunabhaengig).
    double evaluate(const std::vector<int>& v) override {
        if (v.size() < 2) return std::numeric_limits<double>::quiet_NaN();
        const size_t n_windows = (v.size() - 1 + T_ - 1) / T_;
        const size_t used = std::min(kMaxSlots, n_windows);
        std::vector<double> sum(used, 0.0);
        std::vector<size_t> cnt(used, 0);
        parallel_slots(used, threads_, [&](size_t sl) {
            for (size_t wdw = sl; wdw < n_windows; wdw += used) {
                const size_t start = wdw * T_;
                const size_t len = std::min(T_, v.size() - 1 - start);
                model_.forward(&v[start], len, work_[sl]);
                sum[sl] += model_.loss(&v[start + 1], len, work_[sl]) * (double)len;
                cnt[sl] += len;
            }
        });
        double total = 0.0;
        size_t count = 0;
        for (size_t sl = 0; sl < used; ++sl) { total += sum[sl]; count += cnt[sl]; }
        return total / (double)count;
    }

    SkullWeights export_weights() const override {
        SkullWeights w;
        w.is_transformer = true;
        w.tcfg    = tc_;
        w.dim     = tc_.dim;
        w.vocab   = tc_.vocab;
        w.tparams = model_.params;
        if (has_bpe_) { w.has_bpe = true; w.bpe = bpe_; }
        return w;
    }

private:
    TrainConfig cfg_;
    TransformerConfig tc_;
    const std::vector<int>& ids_;
    Transformer model_;
    Adam opt_;
    std::mt19937 rng_;
    size_t n_pairs_ = 0, T_ = 0, seqs_per_epoch_ = 0, batch_ = 1, updates_per_epoch_ = 0;
    size_t total_updates_ = 0, update_no_ = 0;
    size_t slots_ = 1, threads_ = 1;
    std::vector<Transformer::Workspace> work_;
    std::vector<std::vector<double>> slot_grads_;
    bool has_bpe_ = false;
    BPETokenizer bpe_;
};

// ============================================================
//  Gemeinsamer Trainingsrahmen
// ============================================================
inline double perplexity(double loss) { return std::exp(loss); }

inline TrainResult run_training(const TrainConfig& cfg, TrainableModel& model,
                                const std::vector<int>& val_ids, const std::string& weights_path) {
    using clock = std::chrono::high_resolution_clock;
    const bool has_val = val_ids.size() >= 2;
    TrainResult res;
    res.has_val = has_val;

    double best_val = std::numeric_limits<double>::infinity();
    int    best_epoch = 0, bad_epochs = 0;
    SkullWeights best;
    auto t_start = clock::now();
    auto seconds = [&]() { return std::chrono::duration<double>(clock::now() - t_start).count(); };

    for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
        const double loss = model.train_epoch();
        if (!std::isfinite(loss))
            throw std::runtime_error(
                "train: Loss ist nicht endlich (Training divergiert) - 'rate' verkleinern");
        res.train_loss = loss;
        res.epochs_run = epoch;

        bool improved = false;
        double v = std::numeric_limits<double>::quiet_NaN();
        if (has_val) {
            v = model.evaluate(val_ids);
            if (!std::isfinite(v))
                throw std::runtime_error(
                    "train: Validierungs-Loss ist nicht endlich (Training divergiert) - 'rate' verkleinern");
            if (v < best_val) {
                best_val = v; best_epoch = epoch; bad_epochs = 0; improved = true;
                best = model.export_weights();
            } else {
                ++bad_epochs;
            }
        }

        if (epoch == 1 || epoch % 10 == 0 || epoch == cfg.epochs) {
            std::cout << "Epoche " << epoch << "/" << cfg.epochs << "  |  Loss: " << loss;
            if (has_val)
                std::cout << "  |  Val: " << v << " (Perplexitaet " << perplexity(v) << ")"
                          << (improved ? " *" : "");
            std::cout << "  |  Zeit: " << seconds() << "s\n";
        }

        if (has_val && cfg.patience > 0 && bad_epochs >= cfg.patience && epoch < cfg.epochs) {
            std::cout << "[Skull] Early Stopping nach Epoche " << epoch << ": seit " << bad_epochs
                      << " Epochen keine Verbesserung auf den Validierungsdaten\n";
            res.stopped_early = true;
            break;
        }
    }

    std::cout << "\n[Skull] Training abgeschlossen! " << seconds() << "s\n";

    SkullWeights final_w;
    if (has_val) {
        res.val_loss   = best_val;
        res.best_epoch = best_epoch;
        final_w        = std::move(best);
        std::cout << "[Skull] Beste Validierung: Epoche " << best_epoch << ", Val-Loss " << best_val
                  << " (Perplexitaet " << perplexity(best_val) << ") - diese Gewichte werden gespeichert\n";
        if (best_epoch < res.epochs_run)
            std::cout << "[Skull] Hinweis: Nach Epoche " << best_epoch
                      << " wurde das Modell auf den Validierungsdaten schlechter (Ueberanpassung). "
                         "Weniger Epochen, mehr Daten oder ein kleineres Modell helfen.\n";
    } else {
        res.best_epoch = res.epochs_run;
        final_w        = model.export_weights();
    }

    save_weights(weights_path, final_w);
    std::cout << "[Skull] Gewichte gespeichert: " << weights_path << "\n\n";
    return res;
}

inline TrainResult skull_train(const TrainConfig& cfg) {
    // ---- Eingaben pruefen ----
    if (cfg.data_path.empty())
        throw std::runtime_error("train: 'data' fehlt (Pfad zur Trainingsdatei als String)");
    if (cfg.epochs < 1)
        throw std::runtime_error("train: 'epochs' muss >= 1 sein");
    if (!(cfg.rate > 0.0) || !std::isfinite(cfg.rate))
        throw std::runtime_error("train: 'rate' muss eine positive Zahl sein");
    if (cfg.batch < 1)
        throw std::runtime_error("train: 'batch' muss >= 1 sein");
    if (cfg.dim < 1 || cfg.dim > weights_detail::MAX_DIM)
        throw std::runtime_error("train: 'dim' muss zwischen 1 und 65536 liegen");
    if (cfg.vocab < 1 || cfg.vocab > weights_detail::MAX_VOCAB)
        throw std::runtime_error("train: 'vocab' muss zwischen 1 und 16777216 liegen");
    if (cfg.context < 1)
        throw std::runtime_error("train: 'context' muss >= 1 sein (1 = Bigram, > 1 = Transformer)");
    if (!(cfg.val >= 0.0 && cfg.val <= 0.5))
        throw std::runtime_error("train: 'val' muss zwischen 0 und 0.5 liegen (Anteil der Daten, 0 = keine Validierung)");
    if (cfg.patience < 0)
        throw std::runtime_error("train: 'patience' muss >= 0 sein (0 = aus)");
    if (cfg.context <= 1 && (cfg.heads != 0 || cfg.layers != 0))
        std::cout << "[WARNUNG] 'heads' und 'layers' wirken nur mit context > 1 (Transformer); "
                     "ohne context trainiert Skull das Bigram-Modell\n";
    if (cfg.patience > 0 && cfg.val <= 0.0)
        std::cout << "[WARNUNG] 'patience' wirkt nur mit Validierung (val > 0)\n";

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Skull Trainer v" SKULL_VERSION "\n";
    std::cout << "========================================\n";
    std::cout << "  Datei:    " << cfg.data_path << "\n";
    std::cout << "  Dim:      " << cfg.dim       << "\n";
    std::cout << "  Epochen:  " << cfg.epochs    << "\n";
    std::cout << "  Rate:     " << cfg.rate      << (cfg.context > 1 ? " (Adam)" : " (SGD)") << "\n";
    std::cout << "  Batch:    " << cfg.batch     << "\n";
    std::cout << "  Modell:   " << (cfg.context > 1 ? "Transformer" : "Bigram") << "\n";
    std::cout << "  BPE:      " << (cfg.use_bpe ? "ja" : "nein") << "\n";
    std::cout << "  Rechnen:  CPU";
#if SKULL_AVX2
    std::cout << " (AVX2 aktiv)";
#endif
    std::cout << "\n";
    std::cout << "========================================\n\n";

    // GPU: Das Geraet wird initialisiert und angezeigt, die Berechnung
    // laeuft aber noch auf der CPU. Das sagen wir offen.
    if (cfg.use_gpu) {
        bool ok = skull_init_gpu(cfg.prefer_amd);
        std::cout << "[Skull] Hinweis: GPU-Training ist noch nicht implementiert"
                  << (ok ? " (Geraet wurde erkannt)" : " (kein Geraet verfuegbar)")
                  << " - das Training laeuft auf der CPU.\n";
    }

    // ---- Daten laden und tokenisieren ----
    auto tok_result = skull_tokenize_file(cfg.data_path, cfg.use_bpe, cfg.bpe_vocab);
    const size_t vocab = cfg.use_bpe ? (size_t)tok_result.vocab_size : cfg.vocab;

    if (tok_result.tokens.size() < 2)
        throw std::runtime_error("train: mindestens 2 Tokens noetig, gefunden: " +
                                 std::to_string(tok_result.tokens.size()));

    // Token-IDs ausserhalb des Vokabulars werden auf 0 abgebildet
    // (der Generator macht dasselbe mit Prompt-Zeichen).
    std::vector<int> ids = tok_result.tokens;
    size_t clipped = 0;
    for (auto& id : ids)
        if (id < 0 || (size_t)id >= vocab) { id = 0; ++clipped; }
    if (clipped > 0)
        std::cout << "[Skull] Hinweis: " << clipped << " Tokens lagen ausserhalb von vocab="
                  << vocab << " und wurden auf ID 0 abgebildet\n";

    // ---- Training / Validierung trennen ----
    DataSplit split = split_data(ids, cfg.val);
    if (!split.note.empty())
        std::cout << "[Skull] Hinweis: " << split.note << "\n";
    if (!split.val.empty())
        std::cout << "[Skull] Validierung: " << split.val.size() << " Token (" << (int)(cfg.val * 100.0 + 0.5)
                  << " % vom Dateiende), Training: " << split.train.size() << " Token\n";

    const BPETokenizer* bpe = cfg.use_bpe ? &tok_result.bpe : nullptr;
    std::unique_ptr<TrainableModel> model;
    if (cfg.context > 1) {
        TransformerConfig tc = make_transformer_config(cfg, vocab, split.train.size() - 1);
        model = std::make_unique<TransformerTrainer>(cfg, tc, split.train, bpe);
    } else {
        model = std::make_unique<BigramModel>(cfg, vocab, split.train, bpe);
    }
    model->print_info(ids.size());

    return run_training(cfg, *model, split.val, weights_path_for(cfg));
}
