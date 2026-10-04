#pragma once
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include "tensor.h"
#include "tokenizer.h"
#include "gpu.h"
#include "weights.h"
#include "transformer.h"
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
//    Training mit Adam (`rate` = Adam-Lernrate, typisch 0.001 - 0.01).
//
//  Forward und Backward sind in beiden Faellen von Hand ausgeschrieben und
//  arbeiten direkt auf Vektoren (kein Autograd-Graph pro Schritt).
//  Das haelt den Speicher konstant und die Schritte schnell.
//
//  Optionen (train { ... }):
//    data, out, epochs, rate, batch, dim, vocab, steps, context, heads, layers,
//    bpe, bpe_vocab, gpu, prefer_amd
//  out = Pfad der Gewichte-Datei (Standard: data + ".weights").
//  steps = 0 (Standard): jede Epoche geht durch ALLE Tokens.
//  steps > 0: pro Epoche nur so viele Token-Schritte (zufaellige Fenster),
//  damit bei grossen Dateien nach und nach alles gesehen wird, ohne dass
//  eine Epoche ewig dauert.
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

// Xavier-Initialisierung, identische Zufallsfolge wie tensor_rand(r, c, seed).
inline FlatVec xavier_init(size_t rows, size_t cols, unsigned seed) {
    FlatVec v(rows * cols);
    double limit = std::sqrt(6.0 / (double)(rows + cols));
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-limit, limit);
    for (auto& x : v) x = dist(rng);
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
    int         batch      = 1;       // Tokens pro Gewichts-Update (Gradient wird gemittelt)
    size_t      dim        = 64;
    size_t      vocab      = 256;
    size_t      steps      = 0;       // 0 = alle Tokens pro Epoche
    size_t      context    = 1;       // 1 = Bigram-Modell, > 1 = Transformer mit diesem Kontext
    size_t      heads      = 0;       // nur Transformer (0 = Standard: 2)
    size_t      layers     = 0;       // nur Transformer (0 = Standard: 1)
    bool        use_bpe    = false;   // BPE-Tokenisierung aktivieren
    int         bpe_vocab  = 1000;    // BPE Ziel-Vokabular
    bool        use_gpu    = false;   // GPU via OpenCL (derzeit nur initialisiert)
    bool        prefer_amd = false;   // AMD GPU bevorzugen
};

inline std::string weights_path_for(const TrainConfig& cfg) {
    return cfg.out_path.empty() ? cfg.data_path + ".weights" : cfg.out_path;
}

// Transformer-Training: zufaellige Fenster der Laenge `context`, Adam, Mini-Batch aus mehreren Fenstern.
inline void train_transformer(const TrainConfig& cfg, const std::vector<int>& ids,
                              size_t vocab, const BPETokenizer* bpe) {
    using namespace weights_detail;
    if (cfg.context > MAX_CONTEXT)
        throw std::runtime_error("train: 'context' darf hoechstens " + std::to_string(MAX_CONTEXT) + " sein");
    if (cfg.layers > MAX_LAYERS)
        throw std::runtime_error("train: 'layers' darf hoechstens " + std::to_string(MAX_LAYERS) + " sein");

    const size_t n_pairs = ids.size() - 1;
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
                  << " reduziert (die Daten haben nur " << n_pairs << " Trainingspaare)\n";

    const size_t n_params = tc.param_count();
    const size_t MAX_PARAMS = 200u * 1000u * 1000u;
    if (n_params > MAX_PARAMS)
        throw std::runtime_error("train: Modell zu gross (" + std::to_string(n_params) +
                                 " Parameter, maximal " + std::to_string(MAX_PARAMS) + ")");

    const size_t T                = tc.context;
    const size_t pairs_per_epoch  = (cfg.steps > 0 && cfg.steps < n_pairs) ? cfg.steps : n_pairs;
    const size_t seqs_per_epoch   = (pairs_per_epoch + T - 1) / T;
    const size_t batch            = (size_t)cfg.batch;
    const size_t updates_per_epoch = (seqs_per_epoch + batch - 1) / batch;

    std::cout << "[Skull] Modell:     Transformer (" << tc.layers << " Schicht(en), " << tc.heads
              << " Kopf/Koepfe, Kontext " << T << ")\n";
    std::cout << "[Skull] Tokens:     " << ids.size() << "\n";
    std::cout << "[Skull] Vokabular:  " << vocab << "\n";
    std::cout << "[Skull] Sequenzen/Epoche: " << seqs_per_epoch << " (je " << T << " Token), "
              << updates_per_epoch << " Update(s)\n";
    std::cout << "[Skull] Parameter:  " << n_params << "\n\n";

    Transformer model(tc);
    model.init(42);
    Adam opt(model.params.size(), cfg.rate, 1.0);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<size_t> start_dist(0, n_pairs - T);
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
        double epoch_loss = 0.0;
        size_t seqs_done = 0;
        for (size_t u = 0; u < updates_per_epoch; ++u) {
            const size_t n_in = std::min(batch, seqs_per_epoch - seqs_done);
            model.zero_grads();
            for (size_t b = 0; b < n_in; ++b) {
                const size_t s = start_dist(rng);
                epoch_loss += model.step_loss_and_grad(&ids[s], &ids[s + 1], T);
            }
            if (n_in > 1) {
                const double inv = 1.0 / (double)n_in;
                for (auto& g : model.grads) g *= inv;
            }
            opt.step(model.params, model.grads);
            seqs_done += n_in;
        }
        const double avg_loss = epoch_loss / (double)seqs_done;
        if (!std::isfinite(avg_loss))
            throw std::runtime_error(
                "train: Loss ist nicht endlich (Training divergiert) - 'rate' verkleinern");

        if (epoch == 1 || epoch % 10 == 0 || epoch == cfg.epochs) {
            double secs = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t_start).count();
            std::cout << "Epoche " << epoch << "/" << cfg.epochs
                      << "  |  Loss: " << avg_loss
                      << "  |  Zeit: " << secs << "s\n";
        }
    }

    double total = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_start).count();
    std::cout << "\n[Skull] Training abgeschlossen! " << total << "s\n";
    std::cout << "[Skull] Parameter: " << n_params << "\n";

    SkullWeights w;
    w.is_transformer = true;
    w.tcfg   = tc;
    w.dim    = tc.dim;
    w.vocab  = tc.vocab;
    w.tparams = model.params;
    if (bpe) { w.has_bpe = true; w.bpe = *bpe; }
    std::string wp = weights_path_for(cfg);
    save_weights(wp, w);
    std::cout << "[Skull] Gewichte gespeichert: " << wp << "\n\n";
}

inline void skull_train(const TrainConfig& cfg) {
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
    if (cfg.context <= 1 && (cfg.heads != 0 || cfg.layers != 0))
        std::cout << "[WARNUNG] 'heads' und 'layers' wirken nur mit context > 1 (Transformer); "
                     "ohne context trainiert Skull das Bigram-Modell\n";

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
    const size_t dim   = cfg.dim;

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

    if (cfg.context > 1) {
        train_transformer(cfg, ids, vocab, cfg.use_bpe ? &tok_result.bpe : nullptr);
        return;
    }

    const size_t n_pairs        = ids.size() - 1;
    const size_t steps_per_ep   = (cfg.steps > 0 && cfg.steps < n_pairs) ? cfg.steps : n_pairs;
    const size_t params         = vocab * dim + dim * dim + dim * vocab;

    std::cout << "[Skull] Modell:     Bigram (1 Token Kontext)\n";
    std::cout << "[Skull] Tokens:     " << ids.size() << "\n";
    std::cout << "[Skull] Vokabular:  " << vocab << "\n";
    std::cout << "[Skull] Schritte/Epoche: " << steps_per_ep << "\n";
    std::cout << "[Skull] Parameter:  " << params << "\n\n";

    // ---- Gewichte ----
    SkullWeights w;
    w.dim      = dim;
    w.vocab    = vocab;
    w.W_embed  = xavier_init(vocab, dim,  42);
    w.W_hidden = xavier_init(dim,   dim,  43);
    w.W_out    = xavier_init(dim,   vocab, 44);
    if (cfg.use_bpe) { w.has_bpe = true; w.bpe = tok_result.bpe; }

    // Arbeitsvektoren (einmal angelegt, jeder Schritt nutzt sie wieder)
    FlatVec embed(dim), hidden(dim), d_pre(dim), d_embed(dim);
    FlatVec probs(vocab), d_logits(vocab);

    // Mini-Batch-Akkumulation (nur bei batch > 1 genutzt)
    const bool use_batch = cfg.batch > 1;
    FlatVec gW_out, gW_hidden;
    std::unordered_map<int, FlatVec> gEmbed;
    if (use_batch) { gW_out.assign(dim * vocab, 0.0); gW_hidden.assign(dim * dim, 0.0); }
    int in_batch = 0;

    auto flush_batch = [&]() {
        if (in_batch == 0) return;
        const double s = cfg.rate / (double)in_batch;
        for (size_t k = 0; k < gW_out.size(); ++k)    w.W_out[k]    -= s * gW_out[k];
        for (size_t k = 0; k < gW_hidden.size(); ++k) w.W_hidden[k] -= s * gW_hidden[k];
        for (auto& kv : gEmbed) {
            double* row = &w.W_embed[(size_t)kv.first * dim];
            for (size_t d = 0; d < dim; ++d) row[d] -= s * kv.second[d];
        }
        std::fill(gW_out.begin(), gW_out.end(), 0.0);
        std::fill(gW_hidden.begin(), gW_hidden.end(), 0.0);
        gEmbed.clear();
        in_batch = 0;
    };

    std::mt19937 window_rng(12345);
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
        double epoch_loss = 0.0;
        size_t start = 0;
        if (steps_per_ep < n_pairs)
            start = std::uniform_int_distribution<size_t>(0, n_pairs - steps_per_ep)(window_rng);

        for (size_t s = 0; s < steps_per_ep; ++s) {
            const int input_id  = ids[start + s];
            const int target_id = ids[start + s + 1];

            // --- Forward ---
            const double* erow = &w.W_embed[(size_t)input_id * dim];
            for (size_t d = 0; d < dim; ++d) embed[d] = erow[d];

            for (size_t i = 0; i < dim; ++i) {
                const double* hrow = &w.W_hidden[i * dim];
                double pre = 0.0;
                for (size_t j = 0; j < dim; ++j) pre += hrow[j] * embed[j];
                hidden[i] = pre > 0.0 ? pre : 0.0;
            }

            std::fill(probs.begin(), probs.end(), 0.0);   // probs dient zuerst als Logits
            for (size_t i = 0; i < dim; ++i) {
                const double h = hidden[i];
                if (h == 0.0) continue;
                const double* orow = &w.W_out[i * vocab];
                for (size_t j = 0; j < vocab; ++j) probs[j] += h * orow[j];
            }
            softmax_inplace(probs);
            epoch_loss += cross_entropy(probs, target_id);

            // --- Backward (von Hand) ---
            d_logits = probs;
            d_logits[(size_t)target_id] -= 1.0;

            for (size_t i = 0; i < dim; ++i) {
                if (hidden[i] <= 0.0) { d_pre[i] = 0.0; continue; }
                const double* orow = &w.W_out[i * vocab];
                double acc = 0.0;
                for (size_t j = 0; j < vocab; ++j) acc += orow[j] * d_logits[j];
                d_pre[i] = acc;
            }

            std::fill(d_embed.begin(), d_embed.end(), 0.0);
            for (size_t i = 0; i < dim; ++i) {
                const double dp = d_pre[i];
                if (dp == 0.0) continue;
                const double* hrow = &w.W_hidden[i * dim];
                for (size_t j = 0; j < dim; ++j) d_embed[j] += hrow[j] * dp;
            }

            // --- Update (SGD) ---
            if (!use_batch) {
                outer_add(w.W_out,    vocab, hidden, d_logits, -cfg.rate);
                outer_add(w.W_hidden, dim,   d_pre,  embed,    -cfg.rate);
                double* row = &w.W_embed[(size_t)input_id * dim];
                for (size_t d = 0; d < dim; ++d) row[d] -= cfg.rate * d_embed[d];
            } else {
                outer_add(gW_out,    vocab, hidden, d_logits, 1.0);
                outer_add(gW_hidden, dim,   d_pre,  embed,    1.0);
                FlatVec& g = gEmbed[input_id];
                if (g.empty()) g.assign(dim, 0.0);
                for (size_t d = 0; d < dim; ++d) g[d] += d_embed[d];
                if (++in_batch == cfg.batch) flush_batch();
            }
        }
        flush_batch();   // Rest-Batch am Epochenende

        double avg_loss = epoch_loss / (double)steps_per_ep;
        if (!std::isfinite(avg_loss))
            throw std::runtime_error(
                "train: Loss ist nicht endlich (Training divergiert) - 'rate' verkleinern");

        if (epoch == 1 || epoch % 10 == 0 || epoch == cfg.epochs) {
            double secs = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t_start).count();
            std::cout << "Epoche " << epoch << "/" << cfg.epochs
                      << "  |  Loss: " << avg_loss
                      << "  |  Zeit: " << secs << "s\n";
        }
    }

    double total = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t_start).count();

    std::cout << "\n[Skull] Training abgeschlossen! " << total << "s\n";
    std::cout << "[Skull] Parameter: " << params << "\n";

    // ---- Gewichte speichern ----
    std::string wp = weights_path_for(cfg);
    save_weights(wp, w);
    std::cout << "[Skull] Gewichte gespeichert: " << wp << "\n\n";
}
