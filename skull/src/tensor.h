#pragma once
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <functional>
#include <iostream>
#include <unordered_set>
#include <utility>
#include "rng.h"

// ============================================================
//  SKULL TENSOR
//
//  Autograd: Jeder Tensor kennt seine Eingaben (parents) und eine
//  backward_fn. backward() sortiert den Graphen topologisch, damit
//  jede backward_fn genau einmal laeuft (auch bei Diamant-Graphen,
//  z.B. y = h + h).
//
//  Speicher: backward_fn darf ihren eigenen Tensor NICHT per
//  shared_ptr einfangen (Referenzzyklus -> Leck). Sie bekommt
//  stattdessen einen rohen Zeiger; der Tensor lebt solange wie
//  seine backward_fn, da sie in ihm gespeichert ist.
// ============================================================

#if defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
    #include <immintrin.h>
    #define SKULL_AVX2 1
#else
    #define SKULL_AVX2 0
#endif

using FlatVec = std::vector<double>;

inline size_t flat_idx(size_t i, size_t j, size_t cols) {
    return i * cols + j;
}

// Zaehlt lebende Tensoren (auch Kopien). Dient dem Skull-Builtin
// live_tensors() und damit Regressionstests gegen Speicherlecks.
struct TensorLiveCounter {
    static inline size_t count = 0;
    TensorLiveCounter()                           { ++count; }
    TensorLiveCounter(const TensorLiveCounter&)   { ++count; }
    TensorLiveCounter& operator=(const TensorLiveCounter&) = default;
    ~TensorLiveCounter()                          { --count; }
};

struct Tensor {
    size_t  rows, cols;
    FlatVec data;
    FlatVec grad;
    bool    requires_grad = false;

    std::vector<std::shared_ptr<Tensor>> parents;
    std::function<void()>                backward_fn;
    TensorLiveCounter                    live_;

    Tensor(size_t r, size_t c, double init = 0.0)
        : rows(r), cols(c), data(r * c, init), grad(r * c, 0.0) {}

    double& at(size_t i, size_t j)           { return data[flat_idx(i, j, cols)]; }
    double  at(size_t i, size_t j) const      { return data[flat_idx(i, j, cols)]; }
    size_t  size() const                       { return rows * cols; }

    void zero_grad() { std::fill(grad.begin(), grad.end(), 0.0); }

    void accum_grad(const FlatVec& g) {
        for (size_t i = 0; i < grad.size(); ++i) grad[i] += g[i];
    }

    void backward() {
        std::fill(grad.begin(), grad.end(), 1.0);

        // Topologische Ordnung per iterativer Tiefensuche (kein Stack-Overflow
        // bei tiefen Graphen). Post-Order: Eingaben stehen vor ihren Nutzern.
        std::vector<Tensor*> order;
        std::unordered_set<Tensor*> seen;
        std::vector<std::pair<Tensor*, size_t>> stack;
        stack.push_back({this, 0});
        seen.insert(this);
        while (!stack.empty()) {
            Tensor* node = stack.back().first;
            size_t  idx  = stack.back().second++;
            if (idx < node->parents.size()) {
                Tensor* p = node->parents[idx].get();
                if (p && seen.insert(p).second) stack.push_back({p, 0});
            } else {
                order.push_back(node);
                stack.pop_back();
            }
        }

        // Rueckwaerts durchlaufen: Ergebnis zuerst, Eingaben danach.
        for (auto it = order.rbegin(); it != order.rend(); ++it)
            if ((*it)->backward_fn) (*it)->backward_fn();
    }

    void print() const {
        if (rows == 1 && cols == 1) { std::cout << data[0]; return; }
        std::cout << "tensor(" << rows << "x" << cols << ")[\n";
        for (size_t i = 0; i < std::min(rows, (size_t)6); ++i) {
            std::cout << "  [";
            for (size_t j = 0; j < std::min(cols, (size_t)6); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << at(i, j);
            }
            if (cols > 6) std::cout << " ...";
            std::cout << "]\n";
        }
        if (rows > 6) std::cout << "  ...\n";
        std::cout << "]";
    }

    std::string to_string() const {
        if (rows == 1 && cols == 1) return std::to_string(data[0]);
        return "tensor(" + std::to_string(rows) + "x" + std::to_string(cols) + ")";
    }
};

using TensorPtr = std::shared_ptr<Tensor>;

// --- Tensor erstellen ---
inline TensorPtr tensor_zeros(size_t r, size_t c) {
    return std::make_shared<Tensor>(r, c, 0.0);
}
inline TensorPtr tensor_ones(size_t r, size_t c) {
    return std::make_shared<Tensor>(r, c, 1.0);
}
inline TensorPtr tensor_rand(size_t r, size_t c, unsigned seed = 0) {
    auto t = std::make_shared<Tensor>(r, c);
    t->requires_grad = true;
    double limit = std::sqrt(6.0 / (double)(r + c));
    std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
    for (auto& val : t->data) val = rng_uniform(rng, -limit, limit);
    return t;
}

// --- SIMD Operationen ---
inline void simd_add(const FlatVec& a, const FlatVec& b, FlatVec& out) {
    size_t n = a.size(), i = 0;
#if SKULL_AVX2
    for (; i + 3 < n; i += 4) {
        _mm256_storeu_pd(&out[i],
            _mm256_add_pd(_mm256_loadu_pd(&a[i]), _mm256_loadu_pd(&b[i])));
    }
#endif
    for (; i < n; ++i) out[i] = a[i] + b[i];
}

inline void simd_sub(const FlatVec& a, const FlatVec& b, FlatVec& out) {
    size_t n = a.size(), i = 0;
#if SKULL_AVX2
    for (; i + 3 < n; i += 4) {
        _mm256_storeu_pd(&out[i],
            _mm256_sub_pd(_mm256_loadu_pd(&a[i]), _mm256_loadu_pd(&b[i])));
    }
#endif
    for (; i < n; ++i) out[i] = a[i] - b[i];
}

inline void simd_scale(const FlatVec& a, double sc, FlatVec& out) {
    size_t n = a.size(), i = 0;
#if SKULL_AVX2
    __m256d vs = _mm256_set1_pd(sc);
    for (; i + 3 < n; i += 4)
        _mm256_storeu_pd(&out[i], _mm256_mul_pd(_mm256_loadu_pd(&a[i]), vs));
#endif
    for (; i < n; ++i) out[i] = a[i] * sc;
}

inline void simd_relu(const FlatVec& a, FlatVec& out) {
    size_t n = a.size(), i = 0;
#if SKULL_AVX2
    __m256d zero = _mm256_setzero_pd();
    for (; i + 3 < n; i += 4)
        _mm256_storeu_pd(&out[i], _mm256_max_pd(_mm256_loadu_pd(&a[i]), zero));
#endif
    for (; i < n; ++i) out[i] = a[i] > 0.0 ? a[i] : 0.0;
}

inline void simd_sigmoid(const FlatVec& a, FlatVec& out) {
    for (size_t i = 0; i < a.size(); ++i)
        out[i] = 1.0 / (1.0 + std::exp(-a[i]));
}

inline void simd_tanh(const FlatVec& a, FlatVec& out) {
    for (size_t i = 0; i < a.size(); ++i)
        out[i] = std::tanh(a[i]);
}

inline void simd_sgd(FlatVec& data, const FlatVec& grad, double lr) {
    size_t n = data.size(), i = 0;
#if SKULL_AVX2
    __m256d vlr = _mm256_set1_pd(lr);
    for (; i + 3 < n; i += 4)
        _mm256_storeu_pd(&data[i],
            _mm256_fnmadd_pd(vlr, _mm256_loadu_pd(&grad[i]),
                                  _mm256_loadu_pd(&data[i])));
#endif
    for (; i < n; ++i) data[i] -= lr * grad[i];
}

// --- Matrixmultiplikation ---
inline TensorPtr tensor_matmul(TensorPtr A, TensorPtr B) {
    if (A->cols != B->rows)
        throw std::runtime_error(
            "matmul: " + std::to_string(A->rows) + "x" + std::to_string(A->cols) +
            " * " + std::to_string(B->rows) + "x" + std::to_string(B->cols) +
            " passt nicht"
        );
    size_t lR = A->rows, lC = A->cols, rC = B->cols;
    auto C = std::make_shared<Tensor>(lR, rC);
    C->requires_grad = A->requires_grad || B->requires_grad;

    FlatVec BT(B->cols * B->rows);
    for (size_t k = 0; k < B->rows; ++k)
        for (size_t j = 0; j < B->cols; ++j)
            BT[j * B->rows + k] = B->data[flat_idx(k, j, B->cols)];

    for (size_t i = 0; i < lR; ++i) {
        for (size_t j = 0; j < rC; ++j) {
            double sum = 0.0;
            size_t k = 0;
#if SKULL_AVX2
            __m256d vsum = _mm256_setzero_pd();
            for (; k + 3 < lC; k += 4)
                vsum = _mm256_fmadd_pd(
                    _mm256_loadu_pd(&A->data[flat_idx(i, k, lC)]),
                    _mm256_loadu_pd(&BT[j * B->rows + k]),
                    vsum);
            __m128d lo = _mm256_castpd256_pd128(vsum);
            __m128d hi = _mm256_extractf128_pd(vsum, 1);
            __m128d s  = _mm_add_pd(lo, hi);
            sum = _mm_cvtsd_f64(_mm_add_pd(s, _mm_unpackhi_pd(s, s)));
#endif
            for (; k < lC; ++k)
                sum += A->data[flat_idx(i, k, lC)] * BT[j * B->rows + k];
            C->data[flat_idx(i, j, rC)] = sum;
        }
    }

    C->parents = {A, B};
    Tensor* Cp = C.get();  // roh, kein shared_ptr: sonst Referenzzyklus
    C->backward_fn = [A, B, Cp, lR, lC, rC]() {
        if (A->requires_grad)
            for (size_t i = 0; i < lR; ++i)
                for (size_t k = 0; k < lC; ++k)
                    for (size_t j = 0; j < rC; ++j)
                        A->grad[flat_idx(i,k,lC)] +=
                            Cp->grad[flat_idx(i,j,rC)] * B->data[flat_idx(k,j,rC)];
        if (B->requires_grad)
            for (size_t k = 0; k < lC; ++k)
                for (size_t j = 0; j < rC; ++j)
                    for (size_t i = 0; i < lR; ++i)
                        B->grad[flat_idx(k,j,rC)] +=
                            A->data[flat_idx(i,k,lC)] * Cp->grad[flat_idx(i,j,rC)];
    };
    return C;
}

// --- Aktivierungsfunktionen ---
inline TensorPtr tensor_relu(TensorPtr A) {
    auto C = std::make_shared<Tensor>(A->rows, A->cols);
    simd_relu(A->data, C->data);
    C->requires_grad = A->requires_grad;
    C->parents = {A};
    Tensor* Cp = C.get();
    C->backward_fn = [A, Cp]() {
        if (!A->requires_grad) return;
        for (size_t i = 0; i < A->size(); ++i)
            A->grad[i] += (A->data[i] > 0.0) ? Cp->grad[i] : 0.0;
    };
    return C;
}

inline TensorPtr tensor_sigmoid(TensorPtr A) {
    auto C = std::make_shared<Tensor>(A->rows, A->cols);
    simd_sigmoid(A->data, C->data);
    C->requires_grad = A->requires_grad;
    C->parents = {A};
    Tensor* Cp = C.get();
    C->backward_fn = [A, Cp]() {
        if (!A->requires_grad) return;
        for (size_t i = 0; i < A->size(); ++i) {
            double s = Cp->data[i];
            A->grad[i] += Cp->grad[i] * s * (1.0 - s);
        }
    };
    return C;
}

inline TensorPtr tensor_tanh(TensorPtr A) {
    auto C = std::make_shared<Tensor>(A->rows, A->cols);
    simd_tanh(A->data, C->data);
    C->requires_grad = A->requires_grad;
    C->parents = {A};
    Tensor* Cp = C.get();
    C->backward_fn = [A, Cp]() {
        if (!A->requires_grad) return;
        for (size_t i = 0; i < A->size(); ++i) {
            double t = Cp->data[i];
            A->grad[i] += Cp->grad[i] * (1.0 - t * t);
        }
    };
    return C;
}

// --- Addition / Subtraktion ---
inline TensorPtr tensor_add(TensorPtr A, TensorPtr B) {
    if (A->rows != B->rows || A->cols != B->cols)
        throw std::runtime_error("add: Dimensionen passen nicht!");
    auto C = std::make_shared<Tensor>(A->rows, A->cols);
    simd_add(A->data, B->data, C->data);
    C->requires_grad = A->requires_grad || B->requires_grad;
    C->parents = {A, B};
    Tensor* Cp = C.get();
    C->backward_fn = [A, B, Cp]() {
        if (A->requires_grad) A->accum_grad(Cp->grad);
        if (B->requires_grad) B->accum_grad(Cp->grad);
    };
    return C;
}

inline TensorPtr tensor_sub(TensorPtr A, TensorPtr B) {
    if (A->rows != B->rows || A->cols != B->cols)
        throw std::runtime_error("sub: Dimensionen passen nicht!");
    auto C = std::make_shared<Tensor>(A->rows, A->cols);
    simd_sub(A->data, B->data, C->data);
    C->requires_grad = A->requires_grad || B->requires_grad;
    C->parents = {A, B};
    Tensor* Cp = C.get();
    C->backward_fn = [A, B, Cp]() {
        if (A->requires_grad) A->accum_grad(Cp->grad);
        if (B->requires_grad) {
            FlatVec neg(Cp->grad.size());
            for (size_t i = 0; i < neg.size(); ++i) neg[i] = -Cp->grad[i];
            B->accum_grad(neg);
        }
    };
    return C;
}

inline TensorPtr tensor_scale(TensorPtr A, double scalar) {
    auto C = std::make_shared<Tensor>(A->rows, A->cols);
    simd_scale(A->data, scalar, C->data);
    C->requires_grad = A->requires_grad;
    C->parents = {A};
    Tensor* Cp = C.get();
    C->backward_fn = [A, Cp, scalar]() {
        if (!A->requires_grad) return;
        FlatVec g(Cp->grad.size());
        simd_scale(Cp->grad, scalar, g);
        A->accum_grad(g);
    };
    return C;
}

// --- MSE Loss ---
inline TensorPtr tensor_mse_loss(TensorPtr pred, TensorPtr target) {
    if (pred->rows != target->rows || pred->cols != target->cols)
        throw std::runtime_error("mse_loss: Dimensionen passen nicht!");
    size_t N = pred->size();
    double loss = 0.0;
    FlatVec diff(N);
    for (size_t i = 0; i < N; ++i) {
        diff[i] = pred->data[i] - target->data[i];
        loss += diff[i] * diff[i];
    }
    loss /= (double)N;
    auto L = std::make_shared<Tensor>(1, 1);
    L->data[0] = loss;
    L->requires_grad = pred->requires_grad;
    L->parents = {pred};
    L->backward_fn = [pred, N, diff]() {
        if (!pred->requires_grad) return;
        for (size_t i = 0; i < N; ++i)
            pred->grad[i] += 2.0 * diff[i] / (double)N;
    };
    return L;
}

// --- SGD Update ---
inline void tensor_update(TensorPtr W, double lr) {
    simd_sgd(W->data, W->grad, lr);
}

// --- Info ---
inline void print_skull_info() {
    std::cout << "[Skull] Tensor-Engine geladen\n";
#if SKULL_AVX2
    std::cout << "[Skull] AVX2 aktiv (4x doubles pro Takt)\n";
#else
    std::cout << "[Skull] Scalar-Modus (kein AVX2)\n";
#endif
}
