#pragma once
#include <vector>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <functional>
#include "rng.h"

// ============================================================
//  SKULL TRANSFORMER
//
//  Ein kleines GPT-artiges Sprachmodell, komplett auf der CPU,
//  mit von Hand geschriebenem Backward-Pass (kein Autograd-Graph).
//
//  Aufbau (pre-LayerNorm):
//    x   = tok_emb[token] + pos_emb[position]
//    pro Schicht:
//      x = x + Wo * Attention( LN1(x) )          kausale Multi-Head-Attention
//      x = x + W2 * GELU( W1 * LN2(x) + b1 ) + b2
//    logits = LN_f(x) * W_out
//
//  Jede Position sagt das naechste Token voraus; der Verlust ist die
//  mittlere Kreuzentropie ueber alle Positionen. "Kausal" heisst: Position i
//  sieht nur die Positionen 0..i.
//
//  Alle Parameter liegen in EINEM Vektor (params) mit passendem Gradienten-
//  vektor (grads); Adam und der Gradiententest laufen einfach ueber diesen
//  Vektor. Korrektheit des Backward prueft tests/gradcheck.cpp gegen endliche
//  Differenzen.
// ============================================================

struct TransformerConfig {
    size_t vocab   = 256;
    size_t dim     = 32;
    size_t context = 32;   // maximale Sequenzlaenge (Positionen)
    size_t heads   = 2;
    size_t layers  = 1;

    size_t ff() const { return 4 * dim; }
    size_t head_dim() const { return dim / heads; }

    void validate() const {
        if (vocab < 1)   throw std::runtime_error("transformer: vocab muss >= 1 sein");
        if (dim < 1)     throw std::runtime_error("transformer: dim muss >= 1 sein");
        if (context < 1) throw std::runtime_error("transformer: context muss >= 1 sein");
        if (heads < 1)   throw std::runtime_error("transformer: heads muss >= 1 sein");
        if (layers < 1)  throw std::runtime_error("transformer: layers muss >= 1 sein");
        if (dim % heads != 0)
            throw std::runtime_error("transformer: dim (" + std::to_string(dim) +
                                     ") muss durch heads (" + std::to_string(heads) + ") teilbar sein");
    }

    // Anzahl aller Parameter (ohne Ueberlauf-Pruefung; die Aufrufer begrenzen die Groessen)
    size_t param_count() const {
        const size_t d = dim, f = ff();
        size_t n = vocab * d + context * d;                       // tok_emb, pos_emb
        n += layers * (2 * d + 4 * d * d + 2 * d + d * f + f + f * d + d);   // ln1, Wq/k/v/o, ln2, W1, b1, W2, b2
        n += 2 * d + d * vocab;                                   // ln_f, W_out
        return n;
    }
};

class Transformer {
public:
    TransformerConfig cfg;
    std::vector<double> params;
    std::vector<double> grads;

    // Offsets in params/grads
    struct LayerOff { size_t ln1g, ln1b, wq, wk, wv, wo, ln2g, ln2b, w1, b1, w2, b2; };
    size_t off_tok = 0, off_pos = 0, off_lnfg = 0, off_lnfb = 0, off_out = 0;
    std::vector<LayerOff> L;

    explicit Transformer(const TransformerConfig& c) : cfg(c) {
        cfg.validate();
        layout();
        params.assign(cfg.param_count(), 0.0);
        grads.assign(cfg.param_count(), 0.0);
    }

    // Xavier-/Normal-artige Initialisierung. LayerNorm: gamma=1, beta=0, Biases 0.
    void init(unsigned seed) {
        std::mt19937 rng(seed);
        const size_t d = cfg.dim, f = cfg.ff();
        auto uniform = [&](size_t off, size_t n, double limit) {
            for (size_t i = 0; i < n; ++i) params[off + i] = rng_uniform(rng, -limit, limit);
        };
        auto xavier = [&](size_t off, size_t rows, size_t cols) {
            uniform(off, rows * cols, std::sqrt(6.0 / (double)(rows + cols)));
        };
        std::fill(params.begin(), params.end(), 0.0);
        uniform(off_tok, cfg.vocab * d, 0.1);
        uniform(off_pos, cfg.context * d, 0.1);
        for (const auto& l : L) {
            for (size_t i = 0; i < d; ++i) { params[l.ln1g + i] = 1.0; params[l.ln2g + i] = 1.0; }
            xavier(l.wq, d, d); xavier(l.wk, d, d); xavier(l.wv, d, d); xavier(l.wo, d, d);
            xavier(l.w1, d, f); xavier(l.w2, f, d);
        }
        for (size_t i = 0; i < d; ++i) params[off_lnfg + i] = 1.0;
        xavier(off_out, d, cfg.vocab);
    }

    void zero_grads() { std::fill(grads.begin(), grads.end(), 0.0); }

    // Arbeitsspeicher eines Vorwaerts-/Rueckwaerts-Durchlaufs (Zwischenwerte). Jeder Thread braucht
    // einen eigenen; die Parameter (params) werden von allen nur gelesen.
    struct Workspace {
        struct Layer {
            std::vector<double> x_in;                  // [t x d]
            std::vector<double> xn1, mean1, rstd1;     // LN1: normierte Ausgabe (ohne gamma/beta), Statistik
            std::vector<double> ln1;                   // LN1-Ausgabe mit gamma/beta
            std::vector<double> q, k, v;               // [t x d]
            std::vector<double> att;                   // [heads x t x t] Softmax-Wahrscheinlichkeiten
            std::vector<double> ctx;                   // [t x d] Attention-Ergebnis vor Wo
            std::vector<double> x_mid;                 // [t x d]
            std::vector<double> xn2, mean2, rstd2, ln2;
            std::vector<double> h_pre;                 // [t x ff] vor GELU
            std::vector<double> h_act;                 // [t x ff] nach GELU
            std::vector<double> h_tanh;                // [t x ff] tanh-Anteil der GELU (im Backward wiederverwendet)
        };
        std::vector<Layer> layers;
        std::vector<double> x_last;                    // Ausgabe der letzten Schicht [t x d]
        std::vector<double> xnf, meanf, rstdf, lnf;
        std::vector<double> logits, dlogits;
        size_t t_cur = 0;
    };

    // Vorwaerts: Logits [t x vocab] fuer die Token-IDs ids[0..t-1]; merkt sich die Zwischenwerte in ws.
    const std::vector<double>& forward(const int* ids, size_t t, Workspace& ws) const;

    // Verlust (mittlere Kreuzentropie) fuer targets[0..t-1] auf den zuletzt berechneten Logits von ws;
    // legt d(Verlust)/d(Logits) in ws ab.
    double loss(const int* targets, size_t t, Workspace& ws) const;

    // Rueckwaerts: ADDIERT d(Verlust)/d(params) auf g (Vektor der Laenge param_count()).
    // Nur nach forward() + loss() mit demselben ws.
    void backward(const int* ids, size_t t, Workspace& ws, double* g) const;

    // Bequeme Varianten mit einem eigenen Workspace im Objekt und den Gradienten in `grads`
    // (nicht fuer mehrere Threads gleichzeitig).
    const std::vector<double>& forward(const int* ids, size_t t) { return forward(ids, t, ws_); }
    double loss(const int* targets, size_t t) { return loss(targets, t, ws_); }
    void backward(const int* ids, size_t t) { backward(ids, t, ws_, grads.data()); }

    // Verlust fuer eine Sequenz berechnen und Gradienten in `grads` AUFADDIEREN.
    double step_loss_and_grad(const int* ids, const int* targets, size_t t) {
        forward(ids, t, ws_);
        double l = loss(targets, t, ws_);
        backward(ids, t, ws_, grads.data());
        return l;
    }

    // Wie step_loss_and_grad, aber mit explizitem Workspace und Gradientenpuffer (thread-tauglich).
    double step_loss_and_grad(const int* ids, const int* targets, size_t t, Workspace& ws, double* g) const {
        forward(ids, t, ws);
        double l = loss(targets, t, ws);
        backward(ids, t, ws, g);
        return l;
    }

private:
    Workspace ws_;

    static constexpr double LN_EPS = 1e-5;

    void layout() {
        const size_t d = cfg.dim, f = cfg.ff();
        size_t o = 0;
        auto take = [&](size_t n) { size_t r = o; o += n; return r; };
        off_tok = take(cfg.vocab * d);
        off_pos = take(cfg.context * d);
        L.clear();
        for (size_t l = 0; l < cfg.layers; ++l) {
            LayerOff x;
            x.ln1g = take(d); x.ln1b = take(d);
            x.wq = take(d * d); x.wk = take(d * d); x.wv = take(d * d); x.wo = take(d * d);
            x.ln2g = take(d); x.ln2b = take(d);
            x.w1 = take(d * f); x.b1 = take(f); x.w2 = take(f * d); x.b2 = take(d);
            L.push_back(x);
        }
        off_lnfg = take(d); off_lnfb = take(d);
        off_out = take(d * cfg.vocab);
    }

    // ---- Bausteine ----
    // Y[t x out] = X[t x in] * W[in x out]   (+ bias[out], falls bias != nullptr)
    static void linear(const double* X, const double* W, const double* bias,
                       double* Y, size_t t, size_t in, size_t out) {
        for (size_t i = 0; i < t; ++i) {
            double* y = Y + i * out;
            for (size_t j = 0; j < out; ++j) y[j] = bias ? bias[j] : 0.0;
            const double* x = X + i * in;
            for (size_t k = 0; k < in; ++k) {
                const double xv = x[k];
                const double* w = W + k * out;
                for (size_t j = 0; j < out; ++j) y[j] += xv * w[j];
            }
        }
    }
    // Skalarprodukt mit 16 festen Teilsummen. Die Summationsreihenfolge ist vorgegeben (daher auf
    // jedem System gleich), der Compiler darf die 16 Teilsummen aber auf SIMD-Register verteilen.
    // Eine einfache Schleife `acc += a[j]*b[j]` bleibt dagegen skalar: ohne -ffast-math darf der
    // Compiler Gleitkomma-Summen nicht umordnen (gemessen: 3-4 statt 12-15 GFLOP/s).
    static inline double dot(const double* a, const double* b, size_t n) {
        double s[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        size_t j = 0;
        for (; j + 16 <= n; j += 16)
            for (int l = 0; l < 16; ++l) s[l] += a[j + l] * b[j + l];
        double tail = 0.0;
        for (; j < n; ++j) tail += a[j] * b[j];
        const double r0 = (s[0]  + s[1])  + (s[2]  + s[3]);
        const double r1 = (s[4]  + s[5])  + (s[6]  + s[7]);
        const double r2 = (s[8]  + s[9])  + (s[10] + s[11]);
        const double r3 = (s[12] + s[13]) + (s[14] + s[15]);
        return ((r0 + r1) + (r2 + r3)) + tail;
    }

    // Rueckwaerts zu linear: dW += X^T dY, db += sum(dY), dX (+)= dY W^T  (dX darf nullptr sein)
    static void linear_back(const double* X, const double* W, const double* dY,
                            double* dW, double* db, double* dX,
                            size_t t, size_t in, size_t out) {
        if (db)
            for (size_t i = 0; i < t; ++i) {
                const double* dy = dY + i * out;
                for (size_t j = 0; j < out; ++j) db[j] += dy[j];
            }
        // dW += X^T dY: je 4 Zeilen von dY auf einmal, dW wird so 4x seltener geladen/gespeichert
        size_t i = 0;
        for (; i + 4 <= t; i += 4) {
            const double *dy0 = dY + i * out, *dy1 = dy0 + out, *dy2 = dy1 + out, *dy3 = dy2 + out;
            const double *x0 = X + i * in, *x1 = x0 + in, *x2 = x1 + in, *x3 = x2 + in;
            for (size_t k = 0; k < in; ++k) {
                const double a0 = x0[k], a1 = x1[k], a2 = x2[k], a3 = x3[k];
                double* dw = dW + k * out;
                for (size_t j = 0; j < out; ++j)
                    dw[j] += (a0 * dy0[j] + a1 * dy1[j]) + (a2 * dy2[j] + a3 * dy3[j]);
            }
        }
        for (; i < t; ++i) {
            const double* dy = dY + i * out;
            const double* x  = X + i * in;
            for (size_t k = 0; k < in; ++k) {
                const double xv = x[k];
                double* dw = dW + k * out;
                for (size_t j = 0; j < out; ++j) dw[j] += xv * dy[j];
            }
        }
        // dX += dY W^T: Skalarprodukte (Zeile von dY mit Zeile von W)
        if (dX)
            for (size_t r = 0; r < t; ++r) {
                const double* dy = dY + r * out;
                double* dx = dX + r * in;
                for (size_t k = 0; k < in; ++k) dx[k] += dot(dy, W + k * out, out);
            }
    }

    // LayerNorm pro Zeile: xn = (x-mean)*rstd; y = gamma*xn + beta
    static void layernorm(const double* X, const double* gamma, const double* beta,
                          double* xn, double* y, double* mean, double* rstd,
                          size_t t, size_t d) {
        for (size_t i = 0; i < t; ++i) {
            const double* x = X + i * d;
            double m = 0.0;
            for (size_t j = 0; j < d; ++j) m += x[j];
            m /= (double)d;
            double var = 0.0;
            for (size_t j = 0; j < d; ++j) { double c = x[j] - m; var += c * c; }
            var /= (double)d;
            double r = 1.0 / std::sqrt(var + LN_EPS);
            mean[i] = m; rstd[i] = r;
            for (size_t j = 0; j < d; ++j) {
                double n = (x[j] - m) * r;
                xn[i * d + j] = n;
                y[i * d + j]  = gamma[j] * n + beta[j];
            }
        }
    }
    // dX += ...,  dgamma += ..., dbeta += ...
    static void layernorm_back(const double* dY, const double* xn, const double* rstd,
                               const double* gamma, double* dX, double* dgamma, double* dbeta,
                               size_t t, size_t d) {
        for (size_t i = 0; i < t; ++i) {
            const double* dy = dY + i * d;
            const double* n  = xn + i * d;
            double s1 = 0.0, s2 = 0.0;           // mean(dxhat), mean(dxhat * xn)
            for (size_t j = 0; j < d; ++j) {
                double dxh = dy[j] * gamma[j];
                s1 += dxh;
                s2 += dxh * n[j];
                dgamma[j] += dy[j] * n[j];
                dbeta[j]  += dy[j];
            }
            s1 /= (double)d; s2 /= (double)d;
            double* dx = dX + i * d;
            for (size_t j = 0; j < d; ++j) {
                double dxh = dy[j] * gamma[j];
                dx[j] += rstd[i] * (dxh - s1 - n[j] * s2);
            }
        }
    }

    // GELU (tanh-Naeherung) und Ableitung. gelu() liefert zusaetzlich tanh(u), damit das
    // Backward es nicht noch einmal berechnen muss (tanh kostet ~14 ns pro Aufruf).
    static double gelu(double x, double& th) {
        const double c = 0.7978845608028654;     // sqrt(2/pi)
        const double u = c * (x + 0.044715 * x * x * x);
        th = std::tanh(u);
        return 0.5 * x * (1.0 + th);
    }
    static double gelu_grad(double x, double th) {
        const double c = 0.7978845608028654;
        return 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * c * (1.0 + 3.0 * 0.044715 * x * x);
    }
};

inline const std::vector<double>& Transformer::forward(const int* ids, size_t t, Workspace& ws) const {
    const size_t d = cfg.dim, f = cfg.ff(), H = cfg.heads, dh = cfg.head_dim(), V = cfg.vocab;
    if (t < 1 || t > cfg.context)
        throw std::runtime_error("transformer: Sequenzlaenge " + std::to_string(t) +
                                 " ausserhalb von 1.." + std::to_string(cfg.context));
    ws.t_cur = t;

    // Embedding
    std::vector<double> x(t * d);
    for (size_t i = 0; i < t; ++i) {
        if (ids[i] < 0 || (size_t)ids[i] >= V)
            throw std::runtime_error("transformer: Token-ID " + std::to_string(ids[i]) +
                                     " ausserhalb des Vokabulars");
        const double* te = &params[off_tok + (size_t)ids[i] * d];
        const double* pe = &params[off_pos + i * d];
        for (size_t j = 0; j < d; ++j) x[i * d + j] = te[j] + pe[j];
    }

    ws.layers.assign(cfg.layers, Workspace::Layer());
    const double scale = 1.0 / std::sqrt((double)dh);

    for (size_t l = 0; l < cfg.layers; ++l) {
        const LayerOff& o = L[l];
        Workspace::Layer& c = ws.layers[l];
        c.x_in = x;

        // --- Attention ---
        c.xn1.resize(t * d); c.ln1.resize(t * d); c.mean1.resize(t); c.rstd1.resize(t);
        layernorm(c.x_in.data(), &params[o.ln1g], &params[o.ln1b],
                  c.xn1.data(), c.ln1.data(), c.mean1.data(), c.rstd1.data(), t, d);
        c.q.resize(t * d); c.k.resize(t * d); c.v.resize(t * d);
        linear(c.ln1.data(), &params[o.wq], nullptr, c.q.data(), t, d, d);
        linear(c.ln1.data(), &params[o.wk], nullptr, c.k.data(), t, d, d);
        linear(c.ln1.data(), &params[o.wv], nullptr, c.v.data(), t, d, d);

        c.att.assign(H * t * t, 0.0);
        c.ctx.assign(t * d, 0.0);
        for (size_t h = 0; h < H; ++h) {
            for (size_t i = 0; i < t; ++i) {
                double* p = &c.att[(h * t + i) * t];
                const double* qi = &c.q[i * d + h * dh];
                double maxv = -1e300;
                for (size_t j = 0; j <= i; ++j) {            // kausal: nur j <= i
                    const double* kj = &c.k[j * d + h * dh];
                    p[j] = dot(qi, kj, dh) * scale;
                    maxv = std::max(maxv, p[j]);
                }
                double sum = 0.0;
                for (size_t j = 0; j <= i; ++j) { p[j] = std::exp(p[j] - maxv); sum += p[j]; }
                for (size_t j = 0; j <= i; ++j) p[j] /= sum;
                double* out = &c.ctx[i * d + h * dh];
                for (size_t j = 0; j <= i; ++j) {
                    const double* vj = &c.v[j * d + h * dh];
                    for (size_t e = 0; e < dh; ++e) out[e] += p[j] * vj[e];
                }
            }
        }
        std::vector<double> attn_out(t * d);
        linear(c.ctx.data(), &params[o.wo], nullptr, attn_out.data(), t, d, d);
        c.x_mid.resize(t * d);
        for (size_t i = 0; i < t * d; ++i) c.x_mid[i] = c.x_in[i] + attn_out[i];

        // --- MLP ---
        c.xn2.resize(t * d); c.ln2.resize(t * d); c.mean2.resize(t); c.rstd2.resize(t);
        layernorm(c.x_mid.data(), &params[o.ln2g], &params[o.ln2b],
                  c.xn2.data(), c.ln2.data(), c.mean2.data(), c.rstd2.data(), t, d);
        c.h_pre.resize(t * f); c.h_act.resize(t * f); c.h_tanh.resize(t * f);
        linear(c.ln2.data(), &params[o.w1], &params[o.b1], c.h_pre.data(), t, d, f);
        for (size_t i = 0; i < t * f; ++i) c.h_act[i] = gelu(c.h_pre[i], c.h_tanh[i]);
        std::vector<double> mlp_out(t * d);
        linear(c.h_act.data(), &params[o.w2], &params[o.b2], mlp_out.data(), t, f, d);
        for (size_t i = 0; i < t * d; ++i) x[i] = c.x_mid[i] + mlp_out[i];
    }

    ws.x_last = x;
    ws.xnf.resize(t * d); ws.lnf.resize(t * d); ws.meanf.resize(t); ws.rstdf.resize(t);
    layernorm(ws.x_last.data(), &params[off_lnfg], &params[off_lnfb],
              ws.xnf.data(), ws.lnf.data(), ws.meanf.data(), ws.rstdf.data(), t, d);
    ws.logits.resize(t * V);
    linear(ws.lnf.data(), &params[off_out], nullptr, ws.logits.data(), t, d, V);
    return ws.logits;
}

inline double Transformer::loss(const int* targets, size_t t, Workspace& ws) const {
    const size_t V = cfg.vocab;
    if (t != ws.t_cur) throw std::runtime_error("transformer: loss() ohne passendes forward()");
    ws.dlogits.assign(t * V, 0.0);
    double total = 0.0;
    for (size_t i = 0; i < t; ++i) {
        if (targets[i] < 0 || (size_t)targets[i] >= V)
            throw std::runtime_error("transformer: Ziel-ID " + std::to_string(targets[i]) +
                                     " ausserhalb des Vokabulars");
        const double* lg = &ws.logits[i * V];
        double maxv = *std::max_element(lg, lg + V);
        double sum = 0.0;
        double* dl = &ws.dlogits[i * V];
        for (size_t j = 0; j < V; ++j) { dl[j] = std::exp(lg[j] - maxv); sum += dl[j]; }
        for (size_t j = 0; j < V; ++j) dl[j] /= sum;
        total += -std::log(std::max(dl[(size_t)targets[i]], 1e-300));
        dl[(size_t)targets[i]] -= 1.0;
        for (size_t j = 0; j < V; ++j) dl[j] /= (double)t;     // Mittel ueber Positionen
    }
    return total / (double)t;
}

inline void Transformer::backward(const int* ids, size_t t, Workspace& ws, double* g) const {
    const size_t d = cfg.dim, f = cfg.ff(), H = cfg.heads, dh = cfg.head_dim(), V = cfg.vocab;
    if (t != ws.t_cur || ws.dlogits.size() != t * V)
        throw std::runtime_error("transformer: backward() ohne passendes forward()/loss()");
    const double scale = 1.0 / std::sqrt((double)dh);

    // Ausgabe-Projektion und finales LayerNorm
    std::vector<double> dlnf(t * d, 0.0);
    linear_back(ws.lnf.data(), &params[off_out], ws.dlogits.data(),
                (g + off_out), nullptr, dlnf.data(), t, d, V);
    std::vector<double> dx(t * d, 0.0);
    layernorm_back(dlnf.data(), ws.xnf.data(), ws.rstdf.data(), &params[off_lnfg],
                   dx.data(), (g + off_lnfg), (g + off_lnfb), t, d);

    for (size_t l = cfg.layers; l-- > 0;) {
        const LayerOff& o = L[l];
        Workspace::Layer& c = ws.layers[l];

        // x_out = x_mid + mlp_out  ->  dx fliesst unveraendert in beide Zweige
        std::vector<double> dx_mid = dx;                     // Residual-Pfad
        // --- MLP rueckwaerts ---
        std::vector<double> dh_act(t * f, 0.0);
        linear_back(c.h_act.data(), &params[o.w2], dx.data(),
                    (g + o.w2), (g + o.b2), dh_act.data(), t, f, d);
        std::vector<double> dh_pre(t * f);
        for (size_t i = 0; i < t * f; ++i) dh_pre[i] = dh_act[i] * gelu_grad(c.h_pre[i], c.h_tanh[i]);
        std::vector<double> dln2(t * d, 0.0);
        linear_back(c.ln2.data(), &params[o.w1], dh_pre.data(),
                    (g + o.w1), (g + o.b1), dln2.data(), t, d, f);
        layernorm_back(dln2.data(), c.xn2.data(), c.rstd2.data(), &params[o.ln2g],
                       dx_mid.data(), (g + o.ln2g), (g + o.ln2b), t, d);

        // x_mid = x_in + attn_out
        std::vector<double> dx_in = dx_mid;                  // Residual-Pfad
        // --- Attention rueckwaerts ---
        std::vector<double> dctx(t * d, 0.0);
        linear_back(c.ctx.data(), &params[o.wo], dx_mid.data(),
                    (g + o.wo), nullptr, dctx.data(), t, d, d);

        std::vector<double> dq(t * d, 0.0), dk(t * d, 0.0), dv(t * d, 0.0);
        std::vector<double> dp(t);
        for (size_t h = 0; h < H; ++h) {
            for (size_t i = 0; i < t; ++i) {
                const double* p   = &c.att[(h * t + i) * t];
                const double* dc  = &dctx[i * d + h * dh];
                // dp_j = dctx_i . v_j ;  dv_j += p_j * dctx_i
                double dsum = 0.0;
                for (size_t j = 0; j <= i; ++j) {
                    const double* vj = &c.v[j * d + h * dh];
                    double* dvj = &dv[j * d + h * dh];
                    for (size_t e = 0; e < dh; ++e) dvj[e] += p[j] * dc[e];
                    const double s = Transformer::dot(dc, vj, dh);
                    dp[j] = s;
                    dsum += p[j] * s;
                }
                // Softmax-Rueckwaerts: ds_j = p_j * (dp_j - sum_j' p_j' dp_j')
                const double* qi = &c.q[i * d + h * dh];
                double* dqi = &dq[i * d + h * dh];
                for (size_t j = 0; j <= i; ++j) {
                    double ds = p[j] * (dp[j] - dsum) * scale;
                    const double* kj = &c.k[j * d + h * dh];
                    double* dkj = &dk[j * d + h * dh];
                    for (size_t e = 0; e < dh; ++e) { dqi[e] += ds * kj[e]; dkj[e] += ds * qi[e]; }
                }
            }
        }
        std::vector<double> dln1(t * d, 0.0);
        linear_back(c.ln1.data(), &params[o.wq], dq.data(), (g + o.wq), nullptr, dln1.data(), t, d, d);
        linear_back(c.ln1.data(), &params[o.wk], dk.data(), (g + o.wk), nullptr, dln1.data(), t, d, d);
        linear_back(c.ln1.data(), &params[o.wv], dv.data(), (g + o.wv), nullptr, dln1.data(), t, d, d);
        layernorm_back(dln1.data(), c.xn1.data(), c.rstd1.data(), &params[o.ln1g],
                       dx_in.data(), (g + o.ln1g), (g + o.ln1b), t, d);

        dx = dx_in;
    }

    // Embedding-Gradienten
    for (size_t i = 0; i < t; ++i) {
        double* gt = g + off_tok + (size_t)ids[i] * d;
        double* gp = (g + off_pos + i * d);
        for (size_t j = 0; j < d; ++j) { gt[j] += dx[i * d + j]; gp[j] += dx[i * d + j]; }
    }
}

// ============================================================
//  Lernraten-Plan: kurzer linearer Anstieg (Warmup), dann Cosine-Abfall auf
//  min_ratio der Spitzen-Lernrate. Liefert den Faktor fuer Schritt `step`
//  (0-basiert) von `total` Schritten. Ohne Abfall bleibt das Training bei
//  konstanter Lernrate dauerhaft "verrauscht" und konvergiert schlechter.
// ============================================================
inline double lr_schedule(size_t step, size_t total, double min_ratio = 0.1) {
    if (total <= 1) return 1.0;
    const size_t warmup = std::min<size_t>(50, total / 10);
    if (step < warmup) return (double)(step + 1) / (double)(warmup + 1);
    const double progress = (double)(step - warmup) / (double)std::max<size_t>(1, total - warmup - 1);
    const double pi = 3.14159265358979323846;
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + std::cos(pi * std::min(1.0, progress)));
}

// ============================================================
//  Adam (mit globalem Gradienten-Clipping)
// ============================================================
class Adam {
public:
    double lr, beta1, beta2, eps, clip_norm;
    static constexpr size_t kChunk = 16384;   // Abschnittsgroesse fuer die (thread-unabhaengige) Summation

    explicit Adam(size_t n, double lr_, double clip = 1.0)
        : lr(lr_), beta1(0.9), beta2(0.999), eps(1e-8), clip_norm(clip), m_(n, 0.0), v_(n, 0.0) {}

    // Ein Schritt mit den (bereits gemittelten) Gradienten. Liefert die Norm vor dem Clipping.
    // lr_mult skaliert die Lernrate fuer diesen Schritt (siehe lr_schedule()).
    // run_chunks(n, fn) fuehrt fn(0..n-1) aus, evtl. parallel (Standard: nacheinander). Die Summe
    // der Gradienten-Quadrate wird in festen Abschnitten gebildet und in Abschnitts-Reihenfolge
    // addiert; das Ergebnis haengt daher nicht davon ab, wie run_chunks parallelisiert.
    template <typename RunChunks>
    double step(std::vector<double>& params, const std::vector<double>& grads, double lr_mult,
                RunChunks&& run_chunks) {
        const size_t n = params.size();
        const size_t n_chunks = (n + kChunk - 1) / kChunk;
        std::vector<double> part(n_chunks, 0.0);
        run_chunks(n_chunks, [&](size_t c) {
            const size_t lo = c * kChunk, hi = std::min(n, lo + kChunk);
            double sq = 0.0;
            for (size_t i = lo; i < hi; ++i) sq += grads[i] * grads[i];
            part[c] = sq;
        });
        double sq = 0.0;
        for (double x : part) sq += x;
        const double norm = std::sqrt(sq);
        const double s = (clip_norm > 0.0 && norm > clip_norm) ? clip_norm / norm : 1.0;
        ++t_;
        const double b1t = 1.0 - std::pow(beta1, (double)t_);
        const double b2t = 1.0 - std::pow(beta2, (double)t_);
        const double step_lr = lr * lr_mult;
        run_chunks(n_chunks, [&](size_t c) {
            const size_t lo = c * kChunk, hi = std::min(n, lo + kChunk);
            for (size_t i = lo; i < hi; ++i) {
                const double g = grads[i] * s;
                m_[i] = beta1 * m_[i] + (1.0 - beta1) * g;
                v_[i] = beta2 * v_[i] + (1.0 - beta2) * g * g;
                params[i] -= step_lr * (m_[i] / b1t) / (std::sqrt(v_[i] / b2t) + eps);
            }
        });
        return norm;
    }

    double step(std::vector<double>& params, const std::vector<double>& grads, double lr_mult = 1.0) {
        return step(params, grads, lr_mult, [](size_t n, const std::function<void(size_t)>& fn) {
            for (size_t c = 0; c < n; ++c) fn(c);
        });
    }

private:
    std::vector<double> m_, v_;
    size_t t_ = 0;
};
