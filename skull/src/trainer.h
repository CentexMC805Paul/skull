#pragma once
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include "tensor.h"
#include "tokenizer.h"
#include "gpu.h"

// ============================================================
//  SKULL TRAINER  v0.7.0
//  Neu: Tokenizer-Integration (TXT, MD, JSON, JSONL, CSV)
//       GPU-Support (OpenCL) wenn verfuegbar
//       BPE-Tokenisierung optional
// ============================================================

struct SkullNet {
    size_t dim, vocab;
    TensorPtr W_embed, W_hidden, W_out;

    SkullNet(size_t vocab_, size_t dim_) : vocab(vocab_), dim(dim_) {
        W_embed  = tensor_rand(vocab, dim,  42);
        W_hidden = tensor_rand(dim,   dim,  43);
        W_out    = tensor_rand(dim,   vocab, 44);
        W_embed->requires_grad  = true;
        W_hidden->requires_grad = true;
        W_out->requires_grad    = true;
    }

    size_t param_count() const { return vocab*dim + dim*dim + dim*vocab; }
    void zero_all_grads() { W_embed->zero_grad(); W_hidden->zero_grad(); W_out->zero_grad(); }
    void update_all(double lr) {
        tensor_update(W_embed, lr);
        tensor_update(W_hidden, lr);
        tensor_update(W_out, lr);
    }
};

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

inline TensorPtr transpose_W_out(TensorPtr W_out, size_t dim, size_t vocab) {
    auto WT = std::make_shared<Tensor>(vocab, dim, 0.0);
    WT->requires_grad = false;
    for (size_t i = 0; i < dim; ++i)
        for (size_t j = 0; j < vocab; ++j)
            WT->data[flat_idx(j, i, dim)] = W_out->data[flat_idx(i, j, vocab)];
    return WT;
}

struct TrainConfig {
    std::string data_path = "";
    int         epochs    = 10;
    double      rate      = 0.001;
    int         batch     = 1;
    size_t      dim       = 64;
    size_t      vocab     = 256;
    bool        use_bpe   = false;    // BPE-Tokenisierung aktivieren
    int         bpe_vocab = 1000;     // BPE Ziel-Vokabular
    bool        use_gpu   = false;    // GPU via OpenCL
    bool        prefer_amd = false;   // AMD GPU bevorzugen
};

inline void skull_train(const TrainConfig& cfg) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Skull Trainer v0.7.0\n";
    std::cout << "========================================\n";
    std::cout << "  Datei:    " << cfg.data_path << "\n";
    std::cout << "  Dim:      " << cfg.dim       << "\n";
    std::cout << "  Epochen:  " << cfg.epochs    << "\n";
    std::cout << "  Rate:     " << cfg.rate      << "\n";
    std::cout << "  BPE:      " << (cfg.use_bpe ? "ja" : "nein") << "\n";
    std::cout << "  GPU:      " << (cfg.use_gpu ? "OpenCL" : "CPU AVX2") << "\n";
#if SKULL_AVX2
    std::cout << "  SIMD:     AVX2 aktiv\n";
#endif
    std::cout << "========================================\n\n";

    // GPU initialisieren falls gewuenscht
    bool gpu_active = false;
    if (cfg.use_gpu) {
        gpu_active = skull_init_gpu(cfg.prefer_amd);
        if (!gpu_active)
            std::cout << "[Skull] GPU nicht verfuegbar, Fallback: CPU\n";
    }

    // Text laden mit Tokenizer (unterstuetzt alle Formate)
    auto tok_result = skull_tokenize_file(cfg.data_path, cfg.use_bpe, cfg.bpe_vocab);

    if (tok_result.tokens.empty()) {
        std::cerr << "[FEHLER] Keine Tokens geladen!\n";
        return;
    }

    const std::vector<int>& tokens = tok_result.tokens;
    size_t actual_vocab = cfg.use_bpe ? (size_t)tok_result.vocab_size : cfg.vocab;

    std::cout << "[Skull] Tokens:     " << tokens.size() << "\n";
    std::cout << "[Skull] Vokabular:  " << actual_vocab  << "\n";

    // Netzwerk bauen
    SkullNet net(actual_vocab, cfg.dim);
    std::cout << "[Skull] Parameter:  " << net.param_count() << "\n\n";

    auto t_start = std::chrono::high_resolution_clock::now();
    size_t max_steps = std::min(tokens.size() - 1, (size_t)500);

    for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
        double epoch_loss = 0.0;

        for (size_t t = 0; t < max_steps; ++t) {
            int input_id  = tokens[t];
            int target_id = tokens[t + 1];

            // Bounds-Check fuer BPE-Vokabular
            if (input_id  >= (int)actual_vocab) input_id  = 0;
            if (target_id >= (int)actual_vocab) target_id = 0;

            net.zero_all_grads();

            // --- Forward Pass ---
            auto embed = std::make_shared<Tensor>(cfg.dim, 1, 0.0);
            embed->requires_grad = false;
            for (size_t d = 0; d < cfg.dim; ++d)
                embed->data[d] = net.W_embed->data[input_id * cfg.dim + d];

            auto hidden = tensor_relu(tensor_matmul(net.W_hidden, embed));
            auto W_out_T = transpose_W_out(net.W_out, cfg.dim, actual_vocab);
            auto logits  = tensor_matmul(W_out_T, hidden);

            FlatVec probs(logits->data.begin(), logits->data.end());
            softmax_inplace(probs);
            epoch_loss += cross_entropy(probs, target_id);

            // --- Backward Pass ---
            FlatVec d_logits(actual_vocab, 0.0);
            for (size_t i = 0; i < actual_vocab; ++i) d_logits[i] = probs[i];
            d_logits[target_id] -= 1.0;

            for (size_t i = 0; i < cfg.dim; ++i)
                for (size_t j = 0; j < actual_vocab; ++j)
                    net.W_out->grad[flat_idx(i, j, actual_vocab)] +=
                        hidden->data[i] * d_logits[j];

            FlatVec d_hidden(cfg.dim, 0.0);
            for (size_t i = 0; i < cfg.dim; ++i)
                for (size_t j = 0; j < actual_vocab; ++j)
                    d_hidden[i] += net.W_out->data[flat_idx(i, j, actual_vocab)] * d_logits[j];

            FlatVec d_pre(cfg.dim, 0.0);
            for (size_t i = 0; i < cfg.dim; ++i)
                d_pre[i] = (hidden->data[i] > 0.0) ? d_hidden[i] : 0.0;

            for (size_t i = 0; i < cfg.dim; ++i)
                for (size_t j = 0; j < cfg.dim; ++j)
                    net.W_hidden->grad[flat_idx(i, j, cfg.dim)] +=
                        d_pre[i] * embed->data[j];

            for (size_t d = 0; d < cfg.dim; ++d)
                for (size_t i = 0; i < cfg.dim; ++i)
                    net.W_embed->grad[input_id * cfg.dim + d] +=
                        net.W_hidden->data[flat_idx(i, d, cfg.dim)] * d_pre[i];

            net.update_all(cfg.rate);
        }

        double avg_loss = epoch_loss / (double)max_steps;

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
    std::cout << "[Skull] Parameter: " << net.param_count() << "\n";

    // Gewichte speichern
    std::string wp = cfg.data_path + ".weights";
    std::ofstream wf(wp, std::ios::binary);
    if (wf.is_open()) {
        wf.write("SKULL", 5);
        size_t d = cfg.dim, v = actual_vocab;
        wf.write(reinterpret_cast<char*>(&d), sizeof(size_t));
        wf.write(reinterpret_cast<char*>(&v), sizeof(size_t));
        wf.write(reinterpret_cast<char*>(net.W_embed->data.data()),
                 net.W_embed->data.size() * sizeof(double));
        wf.write(reinterpret_cast<char*>(net.W_hidden->data.data()),
                 net.W_hidden->data.size() * sizeof(double));
        wf.write(reinterpret_cast<char*>(net.W_out->data.data()),
                 net.W_out->data.size() * sizeof(double));
        std::cout << "[Skull] Gewichte gespeichert: " << wp << "\n";
    }
    std::cout << "\n";
}
