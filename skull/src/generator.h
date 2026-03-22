#pragma once
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include "tensor.h"

// ============================================================
//  SKULL GENERATOR  v0.5.0
//  Laedt trainierte Gewichte und generiert Text.
//
//  Ablauf:
//    1. Gewichte aus .weights Datei laden
//    2. Prompt zeichenweise einlesen
//    3. Fuer jeden Schritt:
//       a. Letztes Zeichen -> Embedding
//       b. Hidden = relu(W_hidden * embed)
//       c. Logits = W_out^T * hidden
//       d. Softmax -> Wahrscheinlichkeiten
//       e. Naechstes Zeichen sampeln
//       f. Ausgeben und wiederholen
// ============================================================

struct GenerateConfig {
    std::string weights_path = "";
    std::string prompt       = "";
    int         tokens       = 100;
    double      temperature  = 1.0;  // > 1 = kreativer, < 1 = konservativer
    size_t      dim          = 64;
    size_t      vocab        = 256;
};

// Gewichte aus Datei laden
struct LoadedWeights {
    size_t    dim, vocab;
    FlatVec   W_embed;   // vocab x dim
    FlatVec   W_hidden;  // dim   x dim
    FlatVec   W_out;     // dim   x vocab
    bool      ok = false;
};

inline LoadedWeights load_weights(const std::string& path) {
    LoadedWeights w;
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[FEHLER] Gewichte nicht gefunden: " << path << "\n";
        std::cerr << "         Zuerst trainieren: train Modell { ... }\n";
        return w;
    }

    // Header pruefen
    char magic[6] = {};
    f.read(magic, 5);
    if (std::string(magic) != "SKULL") {
        std::cerr << "[FEHLER] Ungueltige Gewichte-Datei.\n";
        return w;
    }

    f.read(reinterpret_cast<char*>(&w.dim),   sizeof(size_t));
    f.read(reinterpret_cast<char*>(&w.vocab), sizeof(size_t));

    // Gewichte laden
    w.W_embed.resize(w.vocab * w.dim);
    w.W_hidden.resize(w.dim * w.dim);
    w.W_out.resize(w.dim * w.vocab);

    f.read(reinterpret_cast<char*>(w.W_embed.data()),
           w.W_embed.size() * sizeof(double));
    f.read(reinterpret_cast<char*>(w.W_hidden.data()),
           w.W_hidden.size() * sizeof(double));
    f.read(reinterpret_cast<char*>(w.W_out.data()),
           w.W_out.size() * sizeof(double));

    w.ok = true;
    return w;
}

// Softmax mit Temperature
inline void softmax_temp(FlatVec& v, double temperature) {
    double maxv = *std::max_element(v.begin(), v.end());
    double sum  = 0.0;
    for (auto& x : v) {
        x = std::exp((x - maxv) / temperature);
        sum += x;
    }
    for (auto& x : v) x /= sum;
}

// Naechstes Token sampeln (zufallig nach Wahrscheinlichkeit)
inline int sample(const FlatVec& probs, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng);
    double cumsum = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumsum += probs[i];
        if (r <= cumsum) return (int)i;
    }
    return (int)probs.size() - 1;
}

// Forward Pass fuer ein einzelnes Zeichen
inline FlatVec forward_char(int token_id,
                             const LoadedWeights& w) {
    size_t dim   = w.dim;
    size_t vocab = w.vocab;

    // 1. Embedding: Zeile token_id aus W_embed -> (dim)
    FlatVec embed(dim);
    for (size_t d = 0; d < dim; ++d)
        embed[d] = w.W_embed[token_id * dim + d];

    // 2. Hidden: W_hidden * embed -> (dim), dann ReLU
    FlatVec hidden(dim, 0.0);
    for (size_t i = 0; i < dim; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < dim; ++j)
            sum += w.W_hidden[i * dim + j] * embed[j];
        hidden[i] = sum > 0.0 ? sum : 0.0;  // ReLU
    }

    // 3. Logits: W_out^T * hidden -> (vocab)
    //    W_out ist (dim x vocab), also W_out^T * hidden:
    //    logits[j] = sum_i(W_out[i][j] * hidden[i])
    FlatVec logits(vocab, 0.0);
    for (size_t j = 0; j < vocab; ++j)
        for (size_t i = 0; i < dim; ++i)
            logits[j] += w.W_out[i * vocab + j] * hidden[i];

    return logits;
}

inline void skull_generate(const GenerateConfig& cfg) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Skull Generator v0.5.0\n";
    std::cout << "========================================\n";
    std::cout << "  Gewichte: " << cfg.weights_path << "\n";
    std::cout << "  Prompt:   \"" << cfg.prompt      << "\"\n";
    std::cout << "  Tokens:   " << cfg.tokens       << "\n";
    std::cout << "  Temp:     " << cfg.temperature  << "\n";
    std::cout << "========================================\n\n";

    // 1. Gewichte laden
    LoadedWeights w = load_weights(cfg.weights_path);
    if (!w.ok) return;

    std::cout << "[Skull] Gewichte geladen: "
              << w.dim << "d, vocab=" << w.vocab << "\n\n";

    // 2. Zufallsgenerator
    std::mt19937 rng(std::random_device{}());

    // 3. Prompt ausgeben
    std::cout << "--- Ausgabe ---\n";
    std::cout << cfg.prompt;
    std::cout.flush();

    // 4. Letztes Zeichen des Prompts als Startpunkt
    int current_token = 32; // Leerzeichen als Default
    if (!cfg.prompt.empty())
        current_token = (unsigned char)cfg.prompt.back();

    // 5. Text generieren
    for (int t = 0; t < cfg.tokens; ++t) {
        // Forward Pass
        FlatVec logits = forward_char(current_token, w);

        // Softmax mit Temperature
        softmax_temp(logits, cfg.temperature);

        // Naechstes Token sampeln
        int next_token = sample(logits, rng);

        // Druckbares Zeichen ausgeben
        char c = (char)next_token;
        if (c >= 32 && c <= 126) {
            std::cout << c;
        } else if (c == '\n' || c == '\t') {
            std::cout << c;
        } else {
            std::cout << ' '; // nicht-druckbare Zeichen als Leerzeichen
        }
        std::cout.flush();

        current_token = next_token;
    }

    std::cout << "\n--- Ende ---\n\n";
}
