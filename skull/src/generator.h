#pragma once
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "tensor.h"
#include "weights.h"
#include "version.h"

// ============================================================
//  SKULL GENERATOR
//  Laedt trainierte Gewichte und generiert Text.
//
//  Ablauf:
//    1. Gewichte aus .weights Datei laden und pruefen
//    2. Prompt tokenisieren (Zeichen oder, falls beim Training
//       BPE benutzt wurde, mit dem mitgespeicherten BPE-Tokenizer)
//    3. Fuer jeden Schritt:
//       a. Letztes Token -> Embedding
//       b. Hidden = relu(W_hidden * embed)
//       c. Logits = W_out^T * hidden
//       d. Naechstes Token waehlen:
//            temperature > 0: aus der Softmax-Verteilung sampeln
//            temperature <= 0: immer das wahrscheinlichste (greedy)
//       e. Ausgeben und wiederholen
// ============================================================

struct GenerateConfig {
    std::string weights_path = "";
    std::string prompt       = "";
    int         tokens       = 100;
    double      temperature  = 1.0;  // > 1 = kreativer, < 1 = konservativer, <= 0 = greedy
    // Optionale Gegenpruefung gegen die Gewichte-Datei (0 = nicht gesetzt).
    // Massgeblich sind immer die Werte aus der Datei.
    size_t      dim          = 0;
    size_t      vocab        = 0;
};

// Softmax mit Temperature (temperature muss > 0 sein)
inline void softmax_temp(FlatVec& v, double temperature) {
    double maxv = *std::max_element(v.begin(), v.end());
    double sum  = 0.0;
    for (auto& x : v) {
        x = std::exp((x - maxv) / temperature);
        sum += x;
    }
    for (auto& x : v) x /= sum;
}

// Naechstes Token sampeln (zufaellig nach Wahrscheinlichkeit)
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

inline int argmax(const FlatVec& v) {
    return (int)(std::max_element(v.begin(), v.end()) - v.begin());
}

// Forward Pass fuer ein einzelnes Token. token_id muss < w.vocab sein.
inline FlatVec forward_token(int token_id, const SkullWeights& w) {
    if (token_id < 0 || (size_t)token_id >= w.vocab)
        throw std::runtime_error("generate: Token-ID " + std::to_string(token_id) +
                                 " liegt ausserhalb des Vokabulars (" +
                                 std::to_string(w.vocab) + ")");
    const size_t dim = w.dim, vocab = w.vocab;

    const double* embed = &w.W_embed[(size_t)token_id * dim];

    FlatVec hidden(dim, 0.0);
    for (size_t i = 0; i < dim; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < dim; ++j)
            sum += w.W_hidden[i * dim + j] * embed[j];
        hidden[i] = sum > 0.0 ? sum : 0.0;  // ReLU
    }

    // logits[j] = sum_i(W_out[i][j] * hidden[i])
    FlatVec logits(vocab, 0.0);
    for (size_t i = 0; i < dim; ++i) {
        const double h = hidden[i];
        if (h == 0.0) continue;
        const double* orow = &w.W_out[i * vocab];
        for (size_t j = 0; j < vocab; ++j) logits[j] += orow[j] * h;
    }
    return logits;
}

// Nur druckbare ASCII-Zeichen, Newline und Tab ausgeben; Rest als Leerzeichen.
inline void print_safe(const std::string& s) {
    for (char c : s) {
        if ((c >= 32 && c <= 126) || c == '\n' || c == '\t') std::cout << c;
        else std::cout << ' ';
    }
}

inline void skull_generate(const GenerateConfig& cfg) {
    // Gewichte zuerst laden: Fehler (Datei fehlt/kaputt) sollen vor dem Banner kommen.
    SkullWeights w = load_weights(cfg.weights_path);

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Skull Generator v" SKULL_VERSION "\n";
    std::cout << "========================================\n";
    std::cout << "  Gewichte: " << cfg.weights_path << "\n";
    std::cout << "  Prompt:   \"" << cfg.prompt      << "\"\n";
    std::cout << "  Tokens:   " << cfg.tokens       << "\n";
    if (cfg.temperature > 0.0) std::cout << "  Temp:     " << cfg.temperature << "\n";
    else                       std::cout << "  Temp:     " << cfg.temperature << " (greedy)\n";
    std::cout << "========================================\n\n";

    if (cfg.tokens < 0)
        throw std::runtime_error("generate: 'tokens' darf nicht negativ sein");
    if (!std::isfinite(cfg.temperature))
        throw std::runtime_error("generate: 'temperature' muss eine endliche Zahl sein");

    std::cout << "[Skull] Gewichte geladen: " << w.dim << "d, vocab=" << w.vocab
              << (w.has_bpe ? ", BPE-Tokenizer" : ", Zeichen-Tokenizer") << "\n";
    if (cfg.dim   != 0 && cfg.dim   != w.dim)
        std::cout << "[WARNUNG] Modell definiert dim=" << cfg.dim
                  << ", die Gewichte haben dim=" << w.dim << " (Gewichte gelten)\n";
    if (cfg.vocab != 0 && !w.has_bpe && cfg.vocab != w.vocab)
        std::cout << "[WARNUNG] Modell definiert vocab=" << cfg.vocab
                  << ", die Gewichte haben vocab=" << w.vocab << " (Gewichte gelten)\n";
    std::cout << "\n";

    // Prompt in Token-IDs umwandeln
    std::vector<int> prompt_ids = w.has_bpe ? w.bpe.encode(cfg.prompt)
                                            : tokenize_chars(cfg.prompt);
    size_t clipped = 0;
    for (auto& id : prompt_ids)
        if (id < 0 || (size_t)id >= w.vocab) { id = 0; ++clipped; }   // wie im Training
    if (clipped > 0)
        std::cout << "[Skull] Hinweis: " << clipped << " Prompt-Token(s) liegen ausserhalb von vocab="
                  << w.vocab << " und werden als ID 0 behandelt\n";

    // Startpunkt: letztes Prompt-Token, sonst Leerzeichen (ID 32), falls im Vokabular
    int current = prompt_ids.empty() ? (w.vocab > 32 ? 32 : 0) : prompt_ids.back();

    std::mt19937 rng(std::random_device{}());

    std::cout << "--- Ausgabe ---\n";
    std::cout << cfg.prompt;
    std::cout.flush();

    for (int t = 0; t < cfg.tokens; ++t) {
        FlatVec logits = forward_token(current, w);

        int next;
        if (cfg.temperature > 0.0) {
            softmax_temp(logits, cfg.temperature);
            next = sample(logits, rng);
        } else {
            next = argmax(logits);
        }

        if (w.has_bpe) print_safe(w.bpe.vocab[(size_t)next]);
        else           print_safe(std::string(1, (char)next));
        std::cout.flush();

        current = next;
    }

    std::cout << "\n--- Ende ---\n\n";
}
