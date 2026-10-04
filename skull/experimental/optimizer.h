#pragma once
// ============================================================
//  SKULL OPTIMIZERS v1.0.0
//  Advanced optimization algorithms for training
// ============================================================

#include "tensor.h"
#include "config.h"
#include <vector>
#include <memory>
#include <cmath>

// ============================================================
//  BASE OPTIMIZER CLASS
// ============================================================

class Optimizer {
public:
    OptimizerType type;
    float learning_rate;
    
    Optimizer(OptimizerType type, float lr = 0.001f)
        : type(type), learning_rate(lr) {}
    
    virtual ~Optimizer() = default;
    
    // Update parameters
    virtual void step(std::vector<TensorPtr>& params) = 0;
    
    // Zero gradients
    virtual void zero_grad(std::vector<TensorPtr>& params) {
        for (auto& param : params) {
            if (param->requires_grad) {
                std::fill(param->grad.begin(), param->grad.end(), 0.0f);
            }
        }
    }
    
    // Set learning rate
    virtual void set_learning_rate(float lr) {
        learning_rate = lr;
    }
    
    // Get learning rate
    virtual float get_learning_rate() const {
        return learning_rate;
    }
    
    // Get optimizer name
    virtual std::string get_name() const = 0;
};

using OptimizerPtr = std::shared_ptr<Optimizer>;

// ============================================================
//  SGD OPTIMIZER
// ============================================================

class SGDOptimizer : public Optimizer {
public:
    float momentum;
    float weight_decay;
    float nesterov;
    std::vector<TensorPtr> velocity;
    
    SGDOptimizer(float lr = 0.001f, float momentum = 0.9f, 
                 float weight_decay = 0.0f, float nesterov = false)
        : Optimizer(OptimizerType::SGD, lr),
          momentum(momentum), weight_decay(weight_decay), nesterov(nesterov) {}
    
    void step(std::vector<TensorPtr>& params) override {
        // Initialize velocity buffers if needed
        if (velocity.empty()) {
            for (auto& param : params) {
                if (param->requires_grad) {
                    velocity.push_back(tensor_zeros(param->rows, param->cols));
                } else {
                    velocity.push_back(nullptr);
                }
            }
        }
        
        for (size_t i = 0; i < params.size(); ++i) {
            auto& param = params[i];
            if (!param->requires_grad) continue;
            
            // Weight decay
            if (weight_decay > 0.0f) {
                for (size_t j = 0; j < param->data.size(); ++j) {
                    param->grad[j] += weight_decay * param->data[j];
                }
            }
            
            // Update velocity
            if (momentum > 0.0f) {
                for (size_t j = 0; j < param->data.size(); ++j) {
                    velocity[i]->data[j] = momentum * velocity[i]->data[j] + 
                                          (1.0f - momentum) * param->grad[j];
                }
            }
            
            // Update parameters
            if (momentum > 0.0f && nesterov) {
                // Nesterov momentum
                for (size_t j = 0; j < param->data.size(); ++j) {
                    param->data[j] -= learning_rate * (
                        momentum * velocity[i]->data[j] + 
                        (1.0f - momentum) * param->grad[j]);
                }
            } else if (momentum > 0.0f) {
                // Standard momentum
                for (size_t j = 0; j < param->data.size(); ++j) {
                    param->data[j] -= learning_rate * velocity[i]->data[j];
                }
            } else {
                // Standard SGD
                for (size_t j = 0; j < param->data.size(); ++j) {
                    param->data[j] -= learning_rate * param->grad[j];
                }
            }
        }
    }
    
    std::string get_name() const override {
        return "SGD(momentum=" + std::to_string(momentum) + 
               ", weight_decay=" + std::to_string(weight_decay) + 
               ", nesterov=" + std::to_string(nesterov) + ")";
    }
};

// ============================================================
//  ADAM OPTIMIZER
// ============================================================

class AdamOptimizer : public Optimizer {
public:
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    int t; // Step counter
    
    std::vector<TensorPtr> m; // First moment vector
    std::vector<TensorPtr> v; // Second moment vector
    
    AdamOptimizer(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f,
                  float eps = 1e-8f, float weight_decay = 0.0f)
        : Optimizer(OptimizerType::ADAM, lr),
          beta1(beta1), beta2(beta2), eps(eps),
          weight_decay(weight_decay), t(0) {}
    
    void step(std::vector<TensorPtr>& params) override {
        t++;
        
        // Initialize moment vectors if needed
        if (m.empty()) {
            for (auto& param : params) {
                if (param->requires_grad) {
                    m.push_back(tensor_zeros(param->rows, param->cols));
                    v.push_back(tensor_zeros(param->rows, param->cols));
                } else {
                    m.push_back(nullptr);
                    v.push_back(nullptr);
                }
            }
        }
        
        for (size_t i = 0; i < params.size(); ++i) {
            auto& param = params[i];
            if (!param->requires_grad) continue;
            
            // Weight decay
            if (weight_decay > 0.0f) {
                for (size_t j = 0; j < param->data.size(); ++j) {
                    param->grad[j] += weight_decay * param->data[j];
                }
            }
            
            // Update biased first moment estimate
            for (size_t j = 0; j < param->data.size(); ++j) {
                m[i]->data[j] = beta1 * m[i]->data[j] + (1.0f - beta1) * param->grad[j];
            }
            
            // Update biased second moment estimate
            for (size_t j = 0; j < param->data.size(); ++j) {
                v[i]->data[j] = beta2 * v[i]->data[j] + (1.0f - beta2) * 
                               (param->grad[j] * param->grad[j]);
            }
            
            // Compute bias-corrected estimates
            float m_hat_denom = 1.0f - std::pow(beta1, t);
            float v_hat_denom = 1.0f - std::pow(beta2, t);
            
            // Update parameters
            for (size_t j = 0; j < param->data.size(); ++j) {
                float m_hat = m[i]->data[j] / m_hat_denom;
                float v_hat = v[i]->data[j] / v_hat_denom;
                param->data[j] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
            }
        }
    }
    
    std::string get_name() const override {
        return "Adam(beta1=" + std::to_string(beta1) + 
               ", beta2=" + std::to_string(beta2) + 
               ", eps=" + std::to_string(eps) + 
               ", weight_decay=" + std::to_string(weight_decay) + ")";
    }
};

// ============================================================
//  ADAMW OPTIMIZER (Adam with Weight Decay)
// ============================================================

class AdamWOptimizer : public Optimizer {
public:
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    int t; // Step counter
    
    std::vector<TensorPtr> m; // First moment vector
    std::vector<TensorPtr> v; // Second moment vector
    
    AdamWOptimizer(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f,
                   float eps = 1e-8f, float weight_decay = 0.01f)
        : Optimizer(OptimizerType::ADAMW, lr),
          beta1(beta1), beta2(beta2), eps(eps),
          weight_decay(weight_decay), t(0) {}
    
    void step(std::vector<TensorPtr>& params) override {
        t++;
        
        // Initialize moment vectors if needed
        if (m.empty()) {
            for (auto& param : params) {
                if (param->requires_grad) {
                    m.push_back(tensor_zeros(param->rows, param->cols));
                    v.push_back(tensor_zeros(param->rows, param->cols));
                } else {
                    m.push_back(nullptr);
                    v.push_back(nullptr);
                }
            }
        }
        
        for (size_t i = 0; i < params.size(); ++i) {
            auto& param = params[i];
            if (!param->requires_grad) continue;
            
            // Update biased first moment estimate
            for (size_t j = 0; j < param->data.size(); ++j) {
                m[i]->data[j] = beta1 * m[i]->data[j] + (1.0f - beta1) * param->grad[j];
            }
            
            // Update biased second moment estimate
            for (size_t j = 0; j < param->data.size(); ++j) {
                v[i]->data[j] = beta2 * v[i]->data[j] + (1.0f - beta2) * 
                               (param->grad[j] * param->grad[j]);
            }
            
            // Compute bias-corrected estimates
            float m_hat_denom = 1.0f - std::pow(beta1, t);
            float v_hat_denom = 1.0f - std::pow(beta2, t);
            
            // Update parameters with weight decay
            for (size_t j = 0; j < param->data.size(); ++j) {
                float m_hat = m[i]->data[j] / m_hat_denom;
                float v_hat = v[i]->data[j] / v_hat_denom;
                
                // AdamW: weight decay is applied differently
                param->data[j] -= learning_rate * (
                    weight_decay * param->data[j] + 
                    m_hat / (std::sqrt(v_hat) + eps));
            }
        }
    }
    
    std::string get_name() const override {
        return "AdamW(beta1=" + std::to_string(beta1) + 
               ", beta2=" + std::to_string(beta2) + 
               ", eps=" + std::to_string(eps) + 
               ", weight_decay=" + std::to_string(weight_decay) + ")";
    }
};

// ============================================================
//  RMSPROP OPTIMIZER
// ============================================================

class RMSpropOptimizer : public Optimizer {
public:
    float rho;
    float eps;
    float momentum;
    
    std::vector<TensorPtr> cache; // Moving average of squared gradients
    std::vector<TensorPtr> velocity; // Momentum
    
    RMSpropOptimizer(float lr = 0.001f, float rho = 0.9f, 
                     float eps = 1e-8f, float momentum = 0.0f)
        : Optimizer(OptimizerType::RMSPROP, lr),
          rho(rho), eps(eps), momentum(momentum) {}
    
    void step(std::vector<TensorPtr>& params) override {
        // Initialize cache if needed
        if (cache.empty()) {
            for (auto& param : params) {
                if (param->requires_grad) {
                    cache.push_back(tensor_zeros(param->rows, param->cols));
                    velocity.push_back(tensor_zeros(param->rows, param->cols));
                } else {
                    cache.push_back(nullptr);
                    velocity.push_back(nullptr);
                }
            }
        }
        
        for (size_t i = 0; i < params.size(); ++i) {
            auto& param = params[i];
            if (!param->requires_grad) continue;
            
            // Update cache (moving average of squared gradients)
            for (size_t j = 0; j < param->data.size(); ++j) {
                cache[i]->data[j] = rho * cache[i]->data[j] + 
                                    (1.0f - rho) * (param->grad[j] * param->grad[j]);
            }
            
            // Update velocity if momentum is enabled
            if (momentum > 0.0f) {
                for (size_t j = 0; j < param->data.size(); ++j) {
                    velocity[i]->data[j] = momentum * velocity[i]->data[j] + 
                                          (1.0f - momentum) * param->grad[j];
                }
            }
            
            // Update parameters
            for (size_t j = 0; j < param->data.size(); ++j) {
                float update = learning_rate * param->grad[j] / 
                              (std::sqrt(cache[i]->data[j]) + eps);
                
                if (momentum > 0.0f) {
                    param->data[j] -= update + momentum * velocity[i]->data[j];
                } else {
                    param->data[j] -= update;
                }
            }
        }
    }
    
    std::string get_name() const override {
        return "RMSprop(rho=" + std::to_string(rho) + 
               ", eps=" + std::to_string(eps) + 
               ", momentum=" + std::to_string(momentum) + ")";
    }
};

// ============================================================
//  ADAGRAD OPTIMIZER
// ============================================================

class AdagradOptimizer : public Optimizer {
public:
    float eps;
    
    std::vector<TensorPtr> sum_sq; // Sum of squares of gradients
    
    AdagradOptimizer(float lr = 0.001f, float eps = 1e-8f)
        : Optimizer(OptimizerType::ADAGRAD, lr), eps(eps) {}
    
    void step(std::vector<TensorPtr>& params) override {
        // Initialize sum_sq if needed
        if (sum_sq.empty()) {
            for (auto& param : params) {
                if (param->requires_grad) {
                    sum_sq.push_back(tensor_zeros(param->rows, param->cols));
                } else {
                    sum_sq.push_back(nullptr);
                }
            }
        }
        
        for (size_t i = 0; i < params.size(); ++i) {
            auto& param = params[i];
            if (!param->requires_grad) continue;
            
            // Update sum of squares
            for (size_t j = 0; j < param->data.size(); ++j) {
                sum_sq[i]->data[j] += param->grad[j] * param->grad[j];
            }
            
            // Update parameters
            for (size_t j = 0; j < param->data.size(); ++j) {
                param->data[j] -= learning_rate * param->grad[j] / 
                                (std::sqrt(sum_sq[i]->data[j]) + eps);
            }
        }
    }
    
    std::string get_name() const override {
        return "Adagrad(eps=" + std::to_string(eps) + ")";
    }
};

// ============================================================
//  ADADELTA OPTIMIZER
// ============================================================

class AdadeltaOptimizer : public Optimizer {
public:
    float rho;
    float eps;
    
    std::vector<TensorPtr> E_g_sq; // Exponentially decaying average of squared gradients
    std::vector<TensorPtr> E_delta_sq; // Exponentially decaying average of squared updates
    
    AdadeltaOptimizer(float lr = 0.001f, float rho = 0.9f, float eps = 1e-8f)
        : Optimizer(OptimizerType::ADADELTA, lr), rho(rho), eps(eps) {}
    
    void step(std::vector<TensorPtr>& params) override {
        // Initialize buffers if needed
        if (E_g_sq.empty()) {
            for (auto& param : params) {
                if (param->requires_grad) {
                    E_g_sq.push_back(tensor_zeros(param->rows, param->cols));
                    E_delta_sq.push_back(tensor_zeros(param->rows, param->cols));
                } else {
                    E_g_sq.push_back(nullptr);
                    E_delta_sq.push_back(nullptr);
                }
            }
        }
        
        for (size_t i = 0; i < params.size(); ++i) {
            auto& param = params[i];
            if (!param->requires_grad) continue;
            
            // Update E_g_sq
            for (size_t j = 0; j < param->data.size(); ++j) {
                E_g_sq[i]->data[j] = rho * E_g_sq[i]->data[j] + 
                                     (1.0f - rho) * (param->grad[j] * param->grad[j]);
            }
            
            // Compute RMS of gradient
            std::vector<float> RMS_g(param->data.size());
            for (size_t j = 0; j < param->data.size(); ++j) {
                RMS_g[j] = std::sqrt(E_g_sq[i]->data[j] + eps);
            }
            
            // Compute update
            std::vector<float> delta(param->data.size());
            for (size_t j = 0; j < param->data.size(); ++j) {
                delta[j] = -learning_rate * param->grad[j] / RMS_g[j];
            }
            
            // Update E_delta_sq
            for (size_t j = 0; j < param->data.size(); ++j) {
                E_delta_sq[i]->data[j] = rho * E_delta_sq[i]->data[j] + 
                                         (1.0f - rho) * (delta[j] * delta[j]);
            }
            
            // Compute RMS of delta
            std::vector<float> RMS_delta(param->data.size());
            for (size_t j = 0; j < param->data.size(); ++j) {
                RMS_delta[j] = std::sqrt(E_delta_sq[i]->data[j] + eps);
            }
            
            // Update parameters
            for (size_t j = 0; j < param->data.size(); ++j) {
                param->data[j] += delta[j] * RMS_g[j] / RMS_delta[j];
            }
        }
    }
    
    std::string get_name() const override {
        return "Adadelta(rho=" + std::to_string(rho) + 
               ", eps=" + std::to_string(eps) + ")";
    }
};

// ============================================================
//  LION OPTIMIZER (New efficient optimizer)
// ============================================================

class LionOptimizer : public Optimizer {
public:
    float beta1;
    float beta2;
    float weight_decay;
    
    std::vector<TensorPtr> momentum; // Momentum buffer
    
    LionOptimizer(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.99f,
                  float weight_decay = 0.0f)
        : Optimizer(OptimizerType::LION, lr),
          beta1(beta1), beta2(beta2), weight_decay(weight_decay) {}
    
    void step(std::vector<TensorPtr>& params) override {
        // Initialize momentum buffer if needed
        if (momentum.empty()) {
            for (auto& param : params) {
                if (param->requires_grad) {
                    momentum.push_back(tensor_zeros(param->rows, param->cols));
                } else {
                    momentum.push_back(nullptr);
                }
            }
        }
        
        for (size_t i = 0; i < params.size(); ++i) {
            auto& param = params[i];
            if (!param->requires_grad) continue;
            
            // Update momentum
            for (size_t j = 0; j < param->data.size(); ++j) {
                momentum[i]->data[j] = (1.0f - beta1) * param->grad[j] + 
                                       beta1 * momentum[i]->data[j];
            }
            
            // Update parameters
            for (size_t j = 0; j < param->data.size(); ++j) {
                float sign = (param->grad[j] > 0.0f) ? 1.0f : -1.0f;
                param->data[j] -= learning_rate * 
                                (sign + beta2 * momentum[i]->data[j] + 
                                 weight_decay * param->data[j]);
            }
        }
    }
    
    std::string get_name() const override {
        return "Lion(beta1=" + std::to_string(beta1) + 
               ", beta2=" + std::to_string(beta2) + 
               ", weight_decay=" + std::to_string(weight_decay) + ")";
    }
};

// ============================================================
//  OPTIMIZER FACTORY
// ============================================================

class OptimizerFactory {
public:
    static OptimizerPtr create(OptimizerType type, float lr = 0.001f,
                              float beta1 = 0.9f, float beta2 = 0.999f,
                              float eps = 1e-8f, float weight_decay = 0.0f,
                              float momentum = 0.9f, float rho = 0.9f) {
        switch (type) {
            case OptimizerType::SGD:
                return std::make_shared<SGDOptimizer>(lr, momentum, weight_decay);
            case OptimizerType::ADAM:
                return std::make_shared<AdamOptimizer>(lr, beta1, beta2, eps, weight_decay);
            case OptimizerType::ADAMW:
                return std::make_shared<AdamWOptimizer>(lr, beta1, beta2, eps, weight_decay);
            case OptimizerType::RMSPROP:
                return std::make_shared<RMSpropOptimizer>(lr, rho, eps, momentum);
            case OptimizerType::ADAGRAD:
                return std::make_shared<AdagradOptimizer>(lr, eps);
            case OptimizerType::ADADELTA:
                return std::make_shared<AdadeltaOptimizer>(lr, rho, eps);
            case OptimizerType::LION:
                return std::make_shared<LionOptimizer>(lr, beta1, beta2, weight_decay);
            default:
                return std::make_shared<AdamOptimizer>(lr);
        }
    }
    
    static OptimizerPtr create_from_string(const std::string& name, 
                                          float lr = 0.001f,
                                          float beta1 = 0.9f, float beta2 = 0.999f,
                                          float eps = 1e-8f, float weight_decay = 0.0f,
                                          float momentum = 0.9f, float rho = 0.9f) {
        if (name == "sgd" || name == "SGD") {
            return create(OptimizerType::SGD, lr, beta1, beta2, eps, weight_decay, momentum);
        } else if (name == "adam" || name == "ADAM") {
            return create(OptimizerType::ADAM, lr, beta1, beta2, eps, weight_decay);
        } else if (name == "adamw" || name == "ADAMW") {
            return create(OptimizerType::ADAMW, lr, beta1, beta2, eps, weight_decay);
        } else if (name == "rmsprop" || name == "RMSPROP") {
            return create(OptimizerType::RMSPROP, lr, beta1, beta2, eps, weight_decay, momentum, rho);
        } else if (name == "adagrad" || name == "ADAGRAD") {
            return create(OptimizerType::ADAGRAD, lr);
        } else if (name == "adadelta" || name == "ADADELTA") {
            return create(OptimizerType::ADADELTA, lr, beta1, beta2, eps, weight_decay, momentum, rho);
        } else if (name == "lion" || name == "LION") {
            return create(OptimizerType::LION, lr, beta1, beta2, eps, weight_decay);
        } else {
            return create(OptimizerType::ADAM, lr);
        }
    }
};

// ============================================================
//  LEARNING RATE SCHEDULERS
// ============================================================

class LRScheduler {
public:
    virtual ~LRScheduler() = default;
    virtual float get_lr() = 0;
    virtual void step() = 0;
    virtual std::string get_name() const = 0;
};

using LRSchedulerPtr = std::shared_ptr<LRScheduler>;

class StepLRScheduler : public LRScheduler {
public:
    float initial_lr;
    float step_size;
    float gamma;
    float current_lr;
    int current_step;
    
    StepLRScheduler(float initial_lr, float step_size, float gamma)
        : initial_lr(initial_lr), step_size(step_size), gamma(gamma),
          current_lr(initial_lr), current_step(0) {}
    
    float get_lr() override {
        return current_lr;
    }
    
    void step() override {
        current_step++;
        current_lr = initial_lr * std::pow(gamma, current_step / step_size);
    }
    
    std::string get_name() const override {
        return "StepLR(initial_lr=" + std::to_string(initial_lr) + 
               ", step_size=" + std::to_string(step_size) + 
               ", gamma=" + std::to_string(gamma) + ")";
    }
};

class CosineAnnealingLRScheduler : public LRScheduler {
public:
    float initial_lr;
    float T_max;
    float eta_min;
    float current_lr;
    int current_step;
    
    CosineAnnealingLRScheduler(float initial_lr, float T_max, float eta_min = 0.0f)
        : initial_lr(initial_lr), T_max(T_max), eta_min(eta_min),
          current_lr(initial_lr), current_step(0) {}
    
    float get_lr() override {
        return current_lr;
    }
    
    void step() override {
        current_step++;
        current_lr = eta_min + (initial_lr - eta_min) * 
                     (1.0f + std::cos(M_PI * current_step / T_max)) / 2.0f;
    }
    
    std::string get_name() const override {
        return "CosineAnnealingLR(initial_lr=" + std::to_string(initial_lr) + 
               ", T_max=" + std::to_string(T_max) + 
               ", eta_min=" + std::to_string(eta_min) + ")";
    }
};

class ExponentialLRScheduler : public LRScheduler {
public:
    float initial_lr;
    float gamma;
    float current_lr;
    int current_step;
    
    ExponentialLRScheduler(float initial_lr, float gamma)
        : initial_lr(initial_lr), gamma(gamma),
          current_lr(initial_lr), current_step(0) {}
    
    float get_lr() override {
        return current_lr;
    }
    
    void step() override {
        current_step++;
        current_lr = initial_lr * std::pow(gamma, current_step);
    }
    
    std::string get_name() const override {
        return "ExponentialLR(initial_lr=" + std::to_string(initial_lr) + 
               ", gamma=" + std::to_string(gamma) + ")";
    }
};

class LinearLRScheduler : public LRScheduler {
public:
    float start_lr;
    float end_lr;
    int total_steps;
    float current_lr;
    int current_step;
    
    LinearLRScheduler(float start_lr, float end_lr, int total_steps)
        : start_lr(start_lr), end_lr(end_lr), total_steps(total_steps),
          current_lr(start_lr), current_step(0) {}
    
    float get_lr() override {
        return current_lr;
    }
    
    void step() override {
        current_step++;
        if (current_step >= total_steps) {
            current_lr = end_lr;
        } else {
            float progress = (float)current_step / total_steps;
            current_lr = start_lr + progress * (end_lr - start_lr);
        }
    }
    
    std::string get_name() const override {
        return "LinearLR(start_lr=" + std::to_string(start_lr) + 
               ", end_lr=" + std::to_string(end_lr) + 
               ", total_steps=" + std::to_string(total_steps) + ")";
    }
};

// ============================================================
//  SCHEDULER FACTORY
// ============================================================

class LRSchedulerFactory {
public:
    static LRSchedulerPtr create(const std::string& name, float initial_lr,
                               float step_size = 30, float gamma = 0.1f,
                               float T_max = 100, float eta_min = 0.0f,
                               int total_steps = 1000) {
        if (name == "steplr" || name == "StepLR") {
            return std::make_shared<StepLRScheduler>(initial_lr, step_size, gamma);
        } else if (name == "cosine" || name == "CosineAnnealingLR") {
            return std::make_shared<CosineAnnealingLRScheduler>(initial_lr, T_max, eta_min);
        } else if (name == "exponential" || name == "ExponentialLR") {
            return std::make_shared<ExponentialLRScheduler>(initial_lr, gamma);
        } else if (name == "linear" || name == "LinearLR") {
            return std::make_shared<LinearLRScheduler>(initial_lr, eta_min, total_steps);
        } else {
            return std::make_shared<StepLRScheduler>(initial_lr, step_size, gamma);
        }
    }
};
