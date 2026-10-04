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
#include "version.h"

// ============================================================
//  SKULL TRAINER
//
//  Modell: Embedding -> Hidden (ReLU) -> Softmax-Ausgabe.
//  Es sieht pro Schritt genau EIN Token (das vorige) und sagt das
//  naechste voraus, ist also ein neuronales Bigram-Modell.
//
//  Forward und Backward sind von Hand ausgeschrieben und arbeiten
//  direkt auf Vektoren (kein Autograd-Graph pro Schritt).
//  Das haelt den Speicher konstant und die Schritte schnell.
//
//  Optionen (train { ... }):
//    data, epochs, rate, batch, dim, vocab, steps, bpe, bpe_vocab,
//    gpu, prefer_amd
//  steps = 0 (Standard): jede Epoche geht durch ALLE Tokens.
//  steps > 0: pro Epoche ein zufaelliges zusammenhaengendes Fenster
//  dieser Laenge (so wird bei grossen Dateien nach und nach alles
//  gesehen, ohne dass eine Epoche ewig dauert).
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
    int         epochs     = 10;
    double      rate       = 0.001;
    int         batch      = 1;       // Tokens pro Gewichts-Update (Gradient wird gemittelt)
    size_t      dim        = 64;
    size_t      vocab      = 256;
    size_t      steps      = 0;       // 0 = alle Tokens pro Epoche
    bool        use_bpe    = false;   // BPE-Tokenisierung aktivieren
    int         bpe_vocab  = 1000;    // BPE Ziel-Vokabular
    bool        use_gpu    = false;   // GPU via OpenCL (derzeit nur initialisiert)
    bool        prefer_amd = false;   // AMD GPU bevorzugen
};

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

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Skull Trainer v" SKULL_VERSION "\n";
    std::cout << "========================================\n";
    std::cout << "  Datei:    " << cfg.data_path << "\n";
    std::cout << "  Dim:      " << cfg.dim       << "\n";
    std::cout << "  Epochen:  " << cfg.epochs    << "\n";
    std::cout << "  Rate:     " << cfg.rate      << "\n";
    std::cout << "  Batch:    " << cfg.batch     << "\n";
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

    const size_t n_pairs        = ids.size() - 1;
    const size_t steps_per_ep   = (cfg.steps > 0 && cfg.steps < n_pairs) ? cfg.steps : n_pairs;
    const size_t params         = vocab * dim + dim * dim + dim * vocab;

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
    std::string wp = cfg.data_path + ".weights";
    save_weights(wp, w);
    std::cout << "[Skull] Gewichte gespeichert: " << wp << "\n\n";
}
