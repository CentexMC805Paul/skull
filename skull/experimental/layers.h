#pragma once
// ============================================================
//  SKULL NEURAL NETWORK LAYERS v1.0.0
//  Flexible, configurable neural network layers
// ============================================================

#include "tensor.h"
#include "config.h"
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <functional>

// ============================================================
//  BASE LAYER CLASS
// ============================================================

class Layer {
public:
    std::string name;
    bool requires_grad = true;
    
    Layer(const std::string& name = "") : name(name) {}
    virtual ~Layer() = default;
    
    // Forward pass
    virtual TensorPtr forward(TensorPtr input) = 0;
    
    // Backward pass
    virtual TensorPtr backward(TensorPtr grad_output) = 0;
    
    // Parameter count
    virtual size_t param_count() const { return 0; }
    
    // Reset gradients
    virtual void zero_grad() {}
    
    // Update parameters
    virtual void update(float lr) {}
    
    // Get all parameters
    virtual std::vector<TensorPtr> get_parameters() { return {}; }
    
    // Set precision mode
    virtual void set_precision(PrecisionMode mode) {}
    
    // Move to GPU
    virtual void to_gpu() {}
    
    // Move to CPU
    virtual void to_cpu() {}
};

using LayerPtr = std::shared_ptr<Layer>;

// ============================================================
//  LINEAR LAYER (Dense / Fully Connected)
// ============================================================

class LinearLayer : public Layer {
public:
    size_t in_features;
    size_t out_features;
    TensorPtr weight;
    TensorPtr bias;
    TensorPtr grad_weight;
    TensorPtr grad_bias;
    TensorPtr last_input;
    
    LinearLayer(size_t in_features, size_t out_features, 
                const std::string& name = "Linear")
        : Layer(name), in_features(in_features), out_features(out_features) {
        
        // Initialize weights with Xavier/Glorot initialization
        double limit = std::sqrt(6.0 / (double)(in_features + out_features));
        weight = tensor_rand(out_features, in_features);
        
        // Scale weights
        for (auto& val : weight->data) {
            val *= limit;
        }
        
        bias = tensor_zeros(out_features, 1);
        grad_weight = tensor_zeros(out_features, in_features);
        grad_bias = tensor_zeros(out_features, 1);
        
        weight->requires_grad = true;
        bias->requires_grad = true;
    }
    
    TensorPtr forward(TensorPtr input) override {
        last_input = input;
        
        // Matrix multiplication: output = input * weight^T + bias
        auto output = tensor_matmul(weight, input);
        
        // Add bias (broadcast)
        for (size_t i = 0; i < output->rows; ++i) {
            for (size_t j = 0; j < output->cols; ++j) {
                output->data[i * output->cols + j] += bias->data[i];
            }
        }
        
        return output;
    }
    
    TensorPtr backward(TensorPtr grad_output) override {
        // Compute gradients
        // grad_weight = grad_output * last_input^T
        // grad_bias = grad_output (sum over batch)
        // grad_input = weight^T * grad_output
        
        // grad_bias
        for (size_t i = 0; i < grad_bias->rows; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < grad_output->cols; ++j) {
                sum += grad_output->data[i * grad_output->cols + j];
            }
            grad_bias->data[i] += sum;
        }
        
        // grad_weight = grad_output * last_input^T
        for (size_t i = 0; i < grad_weight->rows; ++i) {
            for (size_t j = 0; j < grad_weight->cols; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < grad_output->cols; ++k) {
                    sum += grad_output->data[i * grad_output->cols + k] *
                           last_input->data[j * last_input->cols + k];
                }
                grad_weight->data[i * grad_weight->cols + j] += sum;
            }
        }
        
        // grad_input = weight^T * grad_output
        auto grad_input = tensor_matmul_transposed(weight, grad_output);
        
        return grad_input;
    }
    
    size_t param_count() const override {
        return in_features * out_features + out_features;
    }
    
    void zero_grad() override {
        std::fill(grad_weight->data.begin(), grad_weight->data.end(), 0.0f);
        std::fill(grad_bias->data.begin(), grad_bias->data.end(), 0.0f);
    }
    
    void update(float lr) override {
        for (size_t i = 0; i < weight->data.size(); ++i) {
            weight->data[i] -= lr * grad_weight->data[i];
        }
        for (size_t i = 0; i < bias->data.size(); ++i) {
            bias->data[i] -= lr * grad_bias->data[i];
        }
    }
    
    std::vector<TensorPtr> get_parameters() override {
        return {weight, bias};
    }
};

// ============================================================
//  ACTIVATION LAYERS
// ============================================================

class ActivationLayer : public Layer {
public:
    ActivationFunction activation;
    TensorPtr last_output;
    
    ActivationLayer(ActivationFunction activation, 
                    const std::string& name = "Activation")
        : Layer(name), activation(activation) {}
    
    TensorPtr forward(TensorPtr input) override {
        last_output = input;
        
        switch (activation) {
            case ActivationFunction::RELU:
                return tensor_relu(input);
            case ActivationFunction::LEAKY_RELU:
                return tensor_leaky_relu(input, 0.01f);
            case ActivationFunction::SIGMOID:
                return tensor_sigmoid(input);
            case ActivationFunction::TANH:
                return tensor_tanh(input);
            case ActivationFunction::GELU:
                return tensor_gelu(input);
            case ActivationFunction::SWISH:
                return tensor_swish(input);
            case ActivationFunction::SOFTMAX:
                return tensor_softmax(input);
            default:
                return input;
        }
    }
    
    TensorPtr backward(TensorPtr grad_output) override {
        switch (activation) {
            case ActivationFunction::RELU:
                return tensor_relu_backward(grad_output, last_output);
            case ActivationFunction::LEAKY_RELU:
                return tensor_leaky_relu_backward(grad_output, last_output, 0.01f);
            case ActivationFunction::SIGMOID:
                return tensor_sigmoid_backward(grad_output, last_output);
            case ActivationFunction::TANH:
                return tensor_tanh_backward(grad_output, last_output);
            case ActivationFunction::GELU:
                return tensor_gelu_backward(grad_output, last_output);
            case ActivationFunction::SWISH:
                return tensor_swish_backward(grad_output, last_output);
            default:
                return grad_output;
        }
    }
};

// ============================================================
//  DROPOUT LAYER
// ============================================================

class DropoutLayer : public Layer {
public:
    float p; // Dropout probability
    TensorPtr mask;
    bool training;
    
    DropoutLayer(float p = 0.5f, const std::string& name = "Dropout")
        : Layer(name), p(p), training(true) {
        requires_grad = false;
    }
    
    TensorPtr forward(TensorPtr input) override {
        if (!training) {
            return input; // No dropout during inference
        }
        
        mask = tensor_zeros(input->rows, input->cols);
        
        // Apply dropout mask
        std::mt19937 rng(std::random_device{}());
        std::bernoulli_distribution dist(1.0 - p);
        
        for (size_t i = 0; i < input->data.size(); ++i) {
            if (dist(rng)) {
                mask->data[i] = 1.0f / (1.0f - p); // Scale to maintain expected value
            } else {
                mask->data[i] = 0.0f;
            }
        }
        
        auto output = std::make_shared<Tensor>(*input);
        for (size_t i = 0; i < output->data.size(); ++i) {
            output->data[i] *= mask->data[i];
        }
        
        return output;
    }
    
    TensorPtr backward(TensorPtr grad_output) override {
        if (!training) {
            return grad_output;
        }
        
        auto grad_input = std::make_shared<Tensor>(*grad_output);
        for (size_t i = 0; i < grad_input->data.size(); ++i) {
            grad_input->data[i] *= mask->data[i];
        }
        
        return grad_input;
    }
    
    void set_training(bool training_mode) {
        training = training_mode;
    }
};

// ============================================================
//  LAYER NORMALIZATION
// ============================================================

class LayerNormLayer : public Layer {
public:
    size_t normalized_shape;
    TensorPtr gamma;
    TensorPtr beta;
    TensorPtr grad_gamma;
    TensorPtr grad_beta;
    TensorPtr last_input;
    float eps = 1e-5f;
    
    LayerNormLayer(size_t normalized_shape, 
                   const std::string& name = "LayerNorm")
        : Layer(name), normalized_shape(normalized_shape) {
        gamma = tensor_ones(normalized_shape, 1);
        beta = tensor_zeros(normalized_shape, 1);
        grad_gamma = tensor_zeros(normalized_shape, 1);
        grad_beta = tensor_zeros(normalized_shape, 1);
        
        gamma->requires_grad = true;
        beta->requires_grad = true;
    }
    
    TensorPtr forward(TensorPtr input) override {
        last_input = input;
        
        // Compute mean and variance
        float mean = 0.0f;
        for (auto val : input->data) {
            mean += val;
        }
        mean /= input->data.size();
        
        float var = 0.0f;
        for (auto val : input->data) {
            var += (val - mean) * (val - mean);
        }
        var /= input->data.size();
        
        // Normalize
        auto output = std::make_shared<Tensor>(*input);
        for (size_t i = 0; i < output->data.size(); ++i) {
            output->data[i] = (output->data[i] - mean) / std::sqrt(var + eps);
            output->data[i] = gamma->data[i % gamma->rows] * output->data[i] + 
                             beta->data[i % beta->rows];
        }
        
        return output;
    }
    
    TensorPtr backward(TensorPtr grad_output) override {
        // Simplified backward pass
        auto grad_input = std::make_shared<Tensor>(*grad_output);
        
        // Compute gradients for gamma and beta
        for (size_t i = 0; i < grad_gamma->rows; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < grad_output->data.size(); ++j) {
                if (j % grad_gamma->rows == i) {
                    sum += grad_output->data[j] * last_input->data[j];
                }
            }
            grad_gamma->data[i] += sum;
        }
        
        for (size_t i = 0; i < grad_beta->rows; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < grad_output->data.size(); ++j) {
                if (j % grad_beta->rows == i) {
                    sum += grad_output->data[j];
                }
            }
            grad_beta->data[i] += sum;
        }
        
        return grad_input;
    }
    
    void update(float lr) override {
        for (size_t i = 0; i < gamma->data.size(); ++i) {
            gamma->data[i] -= lr * grad_gamma->data[i];
        }
        for (size_t i = 0; i < beta->data.size(); ++i) {
            beta->data[i] -= lr * grad_beta->data[i];
        }
    }
    
    std::vector<TensorPtr> get_parameters() override {
        return {gamma, beta};
    }
};

// ============================================================
//  ATTENTION LAYER (Self-Attention)
// ============================================================

class AttentionLayer : public Layer {
public:
    size_t d_model;
    size_t n_heads;
    size_t d_k;
    size_t d_v;
    
    // Projection matrices
    LayerPtr W_q;
    LayerPtr W_k;
    LayerPtr W_v;
    LayerPtr W_o;
    
    TensorPtr last_attention_weights;
    
    AttentionLayer(size_t d_model, size_t n_heads, 
                  const std::string& name = "Attention")
        : Layer(name), d_model(d_model), n_heads(n_heads) {
        
        if (d_model % n_heads != 0) {
            throw std::runtime_error("d_model must be divisible by n_heads");
        }
        
        d_k = d_model / n_heads;
        d_v = d_model / n_heads;
        
        W_q = std::make_shared<LinearLayer>(d_model, d_model, name + ".W_q");
        W_k = std::make_shared<LinearLayer>(d_model, d_model, name + ".W_k");
        W_v = std::make_shared<LinearLayer>(d_model, d_model, name + ".W_v");
        W_o = std::make_shared<LinearLayer>(d_model, d_model, name + ".W_o");
    }
    
    TensorPtr forward(TensorPtr input) override {
        // Project queries, keys, values
        auto Q = W_q->forward(input);
        auto K = W_k->forward(input);
        auto V = W_v->forward(input);
        
        // Split into multiple heads
        auto Q_split = split_heads(Q, n_heads, d_k);
        auto K_split = split_heads(K, n_heads, d_k);
        auto V_split = split_heads(V, n_heads, d_v);
        
        // Compute attention scores
        auto attention_weights = scaled_dot_product_attention(
            Q_split, K_split, V_split, d_k);
        
        last_attention_weights = attention_weights;
        
        // Concatenate heads
        auto output = concatenate_heads(attention_weights, n_heads, d_v);
        
        // Final projection
        return W_o->forward(output);
    }
    
    TensorPtr backward(TensorPtr grad_output) override {
        // Backpropagate through W_o
        auto grad_W_o = W_o->backward(grad_output);
        
        // Split gradients into heads
        auto grad_attention = split_heads(grad_W_o, n_heads, d_v);
        
        // Backpropagate through attention
        // (This is simplified - full attention backward is more complex)
        
        // Backpropagate through W_v, W_k, W_q
        auto grad_V = W_v->backward(grad_attention);
        auto grad_K = W_k->backward(grad_attention);
        auto grad_Q = W_q->backward(grad_attention);
        
        // Sum gradients
        auto grad_input = tensor_add(grad_Q, tensor_add(grad_K, grad_V));
        
        return grad_input;
    }
    
    size_t param_count() const override {
        return W_q->param_count() + W_k->param_count() + 
               W_v->param_count() + W_o->param_count();
    }
    
    void zero_grad() override {
        W_q->zero_grad();
        W_k->zero_grad();
        W_v->zero_grad();
        W_o->zero_grad();
    }
    
    void update(float lr) override {
        W_q->update(lr);
        W_k->update(lr);
        W_v->update(lr);
        W_o->update(lr);
    }
    
    std::vector<TensorPtr> get_parameters() override {
        auto params = W_q->get_parameters();
        auto params_k = W_k->get_parameters();
        auto params_v = W_v->get_parameters();
        auto params_o = W_o->get_parameters();
        params.insert(params.end(), params_k.begin(), params_k.end());
        params.insert(params.end(), params_v.begin(), params_v.end());
        params.insert(params.end(), params_o.begin(), params_o.end());
        return params;
    }
    
private:
    std::vector<TensorPtr> split_heads(TensorPtr x, size_t num_heads, size_t depth) {
        std::vector<TensorPtr> heads;
        size_t batch_size = x->rows;
        size_t seq_len = x->cols;
        
        for (size_t h = 0; h < num_heads; ++h) {
            auto head = std::make_shared<Tensor>(batch_size * seq_len, depth);
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t s = 0; s < seq_len; ++s) {
                    for (size_t d = 0; d < depth; ++d) {
                        head->data[(b * seq_len + s) * depth + d] = 
                            x->data[b * (seq_len * num_heads * depth) + 
                                    s * (num_heads * depth) + 
                                    h * depth + d];
                    }
                }
            }
            heads.push_back(head);
        }
        
        return heads;
    }
    
    TensorPtr concatenate_heads(const std::vector<TensorPtr>& heads, 
                                 size_t num_heads, size_t depth) {
        size_t batch_size = heads[0]->rows / heads[0]->cols;
        size_t seq_len = heads[0]->cols;
        
        auto output = std::make_shared<Tensor>(batch_size, seq_len * num_heads * depth);
        
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t s = 0; s < seq_len; ++s) {
                    for (size_t d = 0; d < depth; ++d) {
                        output->data[b * (seq_len * num_heads * depth) + 
                                     s * (num_heads * depth) + 
                                     h * depth + d] = 
                            heads[h]->data[(b * seq_len + s) * depth + d];
                    }
                }
            }
        }
        
        return output;
    }
    
    TensorPtr scaled_dot_product_attention(
        const std::vector<TensorPtr>& Q,
        const std::vector<TensorPtr>& K,
        const std::vector<TensorPtr>& V,
        size_t d_k) {
        
        size_t batch_size = Q[0]->rows;
        size_t seq_len = Q[0]->cols;
        size_t n_heads = Q.size();
        
        auto output = std::make_shared<Tensor>(batch_size * seq_len, n_heads * d_k);
        
        float scale = 1.0f / std::sqrt(d_k);
        
        for (size_t h = 0; h < n_heads; ++h) {
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t i = 0; i < seq_len; ++i) {
                    // Compute attention scores
                    std::vector<float> scores(seq_len, 0.0f);
                    for (size_t j = 0; j < seq_len; ++j) {
                        float score = 0.0f;
                        for (size_t k = 0; k < d_k; ++k) {
                            score += Q[h]->data[(b * seq_len + i) * d_k + k] *
                                     K[h]->data[(b * seq_len + j) * d_k + k];
                        }
                        scores[j] = score * scale;
                    }
                    
                    // Softmax
                    float max_score = *std::max_element(scores.begin(), scores.end());
                    float sum = 0.0f;
                    for (auto& s : scores) {
                        s = std::exp(s - max_score);
                        sum += s;
                    }
                    for (auto& s : scores) {
                        s /= sum;
                    }
                    
                    // Weighted sum of values
                    for (size_t k = 0; k < d_k; ++k) {
                        float sum_val = 0.0f;
                        for (size_t j = 0; j < seq_len; ++j) {
                            sum_val += scores[j] * V[h]->data[(b * seq_len + j) * d_k + k];
                        }
                        output->data[(b * seq_len + i) * (n_heads * d_k) + h * d_k + k] = sum_val;
                    }
                }
            }
        }
        
        return output;
    }
};

// ============================================================
//  TRANSFORMER LAYER
// ============================================================

class TransformerLayer : public Layer {
public:
    size_t d_model;
    size_t n_heads;
    size_t d_ff;
    float dropout_p;
    
    LayerPtr attn;
    LayerPtr ff1;
    LayerPtr ff2;
    LayerPtr norm1;
    LayerPtr norm2;
    DropoutLayer dropout;
    
    TransformerLayer(size_t d_model, size_t n_heads, size_t d_ff = 2048, 
                     float dropout_p = 0.1f, 
                     const std::string& name = "Transformer")
        : Layer(name), d_model(d_model), n_heads(n_heads), d_ff(d_ff), 
          dropout_p(dropout_p), dropout(dropout_p, name + ".dropout") {
        
        attn = std::make_shared<AttentionLayer>(d_model, n_heads, name + ".attn");
        ff1 = std::make_shared<LinearLayer>(d_model, d_ff, name + ".ff1");
        ff2 = std::make_shared<LinearLayer>(d_ff, d_model, name + ".ff2");
        norm1 = std::make_shared<LayerNormLayer>(d_model, name + ".norm1");
        norm2 = std::make_shared<LayerNormLayer>(d_model, name + ".norm2");
    }
    
    TensorPtr forward(TensorPtr input) override {
        // Self-attention with residual connection
        auto attn_output = attn->forward(input);
        auto attn_residual = tensor_add(input, attn_output);
        auto norm1_output = norm1->forward(attn_residual);
        
        // Feed-forward with residual connection
        auto ff_output = ff2->forward(
            std::make_shared<ActivationLayer>(ActivationFunction::GELU)->
                forward(ff1->forward(norm1_output)));
        auto ff_residual = tensor_add(norm1_output, ff_output);
        auto output = norm2->forward(ff_residual);
        
        return output;
    }
    
    TensorPtr backward(TensorPtr grad_output) override {
        // Backpropagate through norm2
        auto grad_norm2 = norm2->backward(grad_output);
        
        // Backpropagate through residual (ff + norm1_output)
        auto grad_ff = ff2->backward(
            std::make_shared<ActivationLayer>(ActivationFunction::GELU)->
                backward(ff1->backward(grad_norm2)));
        
        auto grad_norm1 = norm1->backward(
            tensor_add(grad_norm2, grad_ff));
        
        // Backpropagate through residual (attn + input)
        auto grad_attn = attn->backward(grad_norm1);
        
        // Backpropagate through input
        return tensor_add(grad_norm1, grad_attn);
    }
    
    size_t param_count() const override {
        return attn->param_count() + 
               ff1->param_count() + 
               ff2->param_count() + 
               norm1->param_count() + 
               norm2->param_count();
    }
    
    void zero_grad() override {
        attn->zero_grad();
        ff1->zero_grad();
        ff2->zero_grad();
        norm1->zero_grad();
        norm2->zero_grad();
    }
    
    void update(float lr) override {
        attn->update(lr);
        ff1->update(lr);
        ff2->update(lr);
        norm1->update(lr);
        norm2->update(lr);
    }
    
    std::vector<TensorPtr> get_parameters() override {
        auto params = attn->get_parameters();
        auto params_ff1 = ff1->get_parameters();
        auto params_ff2 = ff2->get_parameters();
        auto params_norm1 = norm1->get_parameters();
        auto params_norm2 = norm2->get_parameters();
        params.insert(params.end(), params_ff1.begin(), params_ff1.end());
        params.insert(params.end(), params_ff2.begin(), params_ff2.end());
        params.insert(params.end(), params_norm1.begin(), params_norm1.end());
        params.insert(params.end(), params_norm2.begin(), params_norm2.end());
        return params;
    }
    
    void set_training(bool training_mode) {
        dropout.set_training(training_mode);
    }
};

// ============================================================
//  LSTM LAYER
// ============================================================

class LSTMLayer : public Layer {
public:
    size_t input_size;
    size_t hidden_size;
    
    // Gates: input, forget, cell, output
    LayerPtr W_i;
    LayerPtr W_f;
    LayerPtr W_c;
    LayerPtr W_o;
    LayerPtr U_i;
    LayerPtr U_f;
    LayerPtr U_c;
    LayerPtr U_o;
    TensorPtr b_i;
    TensorPtr b_f;
    TensorPtr b_c;
    TensorPtr b_o;
    
    TensorPtr last_hidden;
    TensorPtr last_cell;
    
    LSTMLayer(size_t input_size, size_t hidden_size, 
              const std::string& name = "LSTM")
        : Layer(name), input_size(input_size), hidden_size(hidden_size) {
        
        W_i = std::make_shared<LinearLayer>(input_size, hidden_size, name + ".W_i");
        W_f = std::make_shared<LinearLayer>(input_size, hidden_size, name + ".W_f");
        W_c = std::make_shared<LinearLayer>(input_size, hidden_size, name + ".W_c");
        W_o = std::make_shared<LinearLayer>(input_size, hidden_size, name + ".W_o");
        
        U_i = std::make_shared<LinearLayer>(hidden_size, hidden_size, name + ".U_i");
        U_f = std::make_shared<LinearLayer>(hidden_size, hidden_size, name + ".U_f");
        U_c = std::make_shared<LinearLayer>(hidden_size, hidden_size, name + ".U_c");
        U_o = std::make_shared<LinearLayer>(hidden_size, hidden_size, name + ".U_o");
        
        b_i = tensor_zeros(hidden_size, 1);
        b_f = tensor_zeros(hidden_size, 1);
        b_c = tensor_zeros(hidden_size, 1);
        b_o = tensor_zeros(hidden_size, 1);
        
        last_hidden = tensor_zeros(hidden_size, 1);
        last_cell = tensor_zeros(hidden_size, 1);
    }
    
    TensorPtr forward(TensorPtr input) override {
        // Input gate
        auto i_t = tensor_sigmoid(
            tensor_add(W_i->forward(input), U_i->forward(last_hidden)));
        
        // Forget gate
        auto f_t = tensor_sigmoid(
            tensor_add(W_f->forward(input), U_f->forward(last_hidden)));
        
        // Cell gate
        auto c_t = tensor_tanh(
            tensor_add(W_c->forward(input), U_c->forward(last_hidden)));
        
        // Output gate
        auto o_t = tensor_sigmoid(
            tensor_add(W_o->forward(input), U_o->forward(last_hidden)));
        
        // Update cell state
        auto new_cell = tensor_add(
            tensor_element_mul(f_t, last_cell),
            tensor_element_mul(i_t, c_t));
        
        // Update hidden state
        auto new_hidden = tensor_element_mul(o_t, tensor_tanh(new_cell));
        
        last_hidden = new_hidden;
        last_cell = new_cell;
        
        return new_hidden;
    }
    
    TensorPtr backward(TensorPtr grad_output) override {
        // Simplified backward pass for LSTM
        // Full LSTM backward is complex due to temporal dependencies
        
        // For now, just backpropagate through the gates
        auto grad_W_o = W_o->backward(grad_output);
        auto grad_U_o = U_o->backward(grad_output);
        
        // This is a placeholder - full LSTM backward would need to handle
        // the recurrent connections properly
        
        return grad_W_o;
    }
    
    size_t param_count() const override {
        return W_i->param_count() + W_f->param_count() + 
               W_c->param_count() + W_o->param_count() +
               U_i->param_count() + U_f->param_count() +
               U_c->param_count() + U_o->param_count() +
               4 * hidden_size; // biases
    }
    
    void zero_grad() override {
        W_i->zero_grad();
        W_f->zero_grad();
        W_c->zero_grad();
        W_o->zero_grad();
        U_i->zero_grad();
        U_f->zero_grad();
        U_c->zero_grad();
        U_o->zero_grad();
    }
    
    void update(float lr) override {
        W_i->update(lr);
        W_f->update(lr);
        W_c->update(lr);
        W_o->update(lr);
        U_i->update(lr);
        U_f->update(lr);
        U_c->update(lr);
        U_o->update(lr);
    }
    
    std::vector<TensorPtr> get_parameters() override {
        auto params = W_i->get_parameters();
        auto params_f = W_f->get_parameters();
        auto params_c = W_c->get_parameters();
        auto params_o = W_o->get_parameters();
        auto params_ui = U_i->get_parameters();
        auto params_uf = U_f->get_parameters();
        auto params_uc = U_c->get_parameters();
        auto params_uo = U_o->get_parameters();
        
        params.insert(params.end(), params_f.begin(), params_f.end());
        params.insert(params.end(), params_c.begin(), params_c.end());
        params.insert(params.end(), params_o.begin(), params_o.end());
        params.insert(params.end(), params_ui.begin(), params_ui.end());
        params.insert(params.end(), params_uf.begin(), params_uf.end());
        params.insert(params.end(), params_uc.begin(), params_uc.end());
        params.insert(params.end(), params_uo.begin(), params_uo.end());
        
        params.push_back(b_i);
        params.push_back(b_f);
        params.push_back(b_c);
        params.push_back(b_o);
        
        return params;
    }
    
    void reset_state() {
        std::fill(last_hidden->data.begin(), last_hidden->data.end(), 0.0f);
        std::fill(last_cell->data.begin(), last_cell->data.end(), 0.0f);
    }
};

// ============================================================
//  CONVOLUTIONAL 1D LAYER
// ============================================================

class Conv1DLayer : public Layer {
public:
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;
    
    TensorPtr weight;
    TensorPtr bias;
    TensorPtr grad_weight;
    TensorPtr grad_bias;
    
    Conv1DLayer(size_t in_channels, size_t out_channels, size_t kernel_size,
                size_t stride = 1, size_t padding = 0,
                const std::string& name = "Conv1D")
        : Layer(name), in_channels(in_channels), out_channels(out_channels),
          kernel_size(kernel_size), stride(stride), padding(padding) {
        
        weight = tensor_rand(out_channels, in_channels * kernel_size);
        bias = tensor_zeros(out_channels, 1);
        grad_weight = tensor_zeros(out_channels, in_channels * kernel_size);
        grad_bias = tensor_zeros(out_channels, 1);
        
        weight->requires_grad = true;
        bias->requires_grad = true;
    }
    
    TensorPtr forward(TensorPtr input) override {
        // Simplified Conv1D implementation
        size_t seq_len = input->cols;
        size_t out_len = (seq_len + 2 * padding - kernel_size) / stride + 1;
        
        auto output = tensor_zeros(out_channels, out_len);
        
        // Apply padding
        auto padded_input = tensor_zeros(in_channels, seq_len + 2 * padding);
        for (size_t i = 0; i < in_channels; ++i) {
            for (size_t j = 0; j < seq_len; ++j) {
                padded_input->data[i * (seq_len + 2 * padding) + j + padding] = 
                    input->data[i * seq_len + j];
            }
        }
        
        // Convolution
        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t i = 0; i < out_len; ++i) {
                float sum = 0.0f;
                for (size_t ic = 0; ic < in_channels; ++ic) {
                    for (size_t k = 0; k < kernel_size; ++k) {
                        size_t input_pos = i * stride + k;
                        sum += padded_input->data[ic * (seq_len + 2 * padding) + input_pos] *
                               weight->data[oc * (in_channels * kernel_size) + ic * kernel_size + k];
                    }
                }
                output->data[oc * out_len + i] = sum + bias->data[oc];
            }
        }
        
        return output;
    }
    
    TensorPtr backward(TensorPtr grad_output) override {
        // Simplified backward pass
        // Full Conv1D backward would need to implement the transposed convolution
        
        auto grad_input = tensor_zeros(in_channels, grad_output->cols * stride);
        
        // This is a placeholder - full Conv1D backward is more complex
        
        return grad_input;
    }
    
    size_t param_count() const override {
        return out_channels * in_channels * kernel_size + out_channels;
    }
    
    void zero_grad() override {
        std::fill(grad_weight->data.begin(), grad_weight->data.end(), 0.0f);
        std::fill(grad_bias->data.begin(), grad_bias->data.end(), 0.0f);
    }
    
    void update(float lr) override {
        for (size_t i = 0; i < weight->data.size(); ++i) {
            weight->data[i] -= lr * grad_weight->data[i];
        }
        for (size_t i = 0; i < bias->data.size(); ++i) {
            bias->data[i] -= lr * grad_bias->data[i];
        }
    }
    
    std::vector<TensorPtr> get_parameters() override {
        return {weight, bias};
    }
};

// ============================================================
//  SEQUENTIAL LAYER (Container for multiple layers)
// ============================================================

class SequentialLayer : public Layer {
public:
    std::vector<LayerPtr> layers;
    
    SequentialLayer(const std::vector<LayerPtr>& layers, 
                   const std::string& name = "Sequential")
        : Layer(name), layers(layers) {}
    
    TensorPtr forward(TensorPtr input) override {
        TensorPtr output = input;
        for (auto& layer : layers) {
            output = layer->forward(output);
        }
        return output;
    }
    
    TensorPtr backward(TensorPtr grad_output) override {
        TensorPtr grad_input = grad_output;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            grad_input = (*it)->backward(grad_input);
        }
        return grad_input;
    }
    
    size_t param_count() const override {
        size_t count = 0;
        for (const auto& layer : layers) {
            count += layer->param_count();
        }
        return count;
    }
    
    void zero_grad() override {
        for (auto& layer : layers) {
            layer->zero_grad();
        }
    }
    
    void update(float lr) override {
        for (auto& layer : layers) {
            layer->update(lr);
        }
    }
    
    std::vector<TensorPtr> get_parameters() override {
        std::vector<TensorPtr> params;
        for (auto& layer : layers) {
            auto layer_params = layer->get_parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }
    
    void add_layer(LayerPtr layer) {
        layers.push_back(layer);
    }
    
    size_t size() const {
        return layers.size();
    }
    
    LayerPtr operator[](size_t index) {
        return layers[index];
    }
};

// ============================================================
//  LAYER FACTORY
// ============================================================

class LayerFactory {
public:
    static LayerPtr create_linear(size_t in_features, size_t out_features, 
                                   const std::string& name = "") {
        return std::make_shared<LinearLayer>(in_features, out_features, name);
    }
    
    static LayerPtr create_activation(ActivationFunction activation, 
                                       const std::string& name = "") {
        return std::make_shared<ActivationLayer>(activation, name);
    }
    
    static LayerPtr create_dropout(float p, const std::string& name = "") {
        return std::make_shared<DropoutLayer>(p, name);
    }
    
    static LayerPtr create_layer_norm(size_t normalized_shape, 
                                       const std::string& name = "") {
        return std::make_shared<LayerNormLayer>(normalized_shape, name);
    }
    
    static LayerPtr create_attention(size_t d_model, size_t n_heads, 
                                     const std::string& name = "") {
        return std::make_shared<AttentionLayer>(d_model, n_heads, name);
    }
    
    static LayerPtr create_transformer(size_t d_model, size_t n_heads, 
                                        size_t d_ff = 2048, float dropout_p = 0.1f,
                                        const std::string& name = "") {
        return std::make_shared<TransformerLayer>(d_model, n_heads, d_ff, dropout_p, name);
    }
    
    static LayerPtr create_lstm(size_t input_size, size_t hidden_size, 
                                 const std::string& name = "") {
        return std::make_shared<LSTMLayer>(input_size, hidden_size, name);
    }
    
    static LayerPtr create_conv1d(size_t in_channels, size_t out_channels, 
                                  size_t kernel_size, size_t stride = 1, 
                                  size_t padding = 0, const std::string& name = "") {
        return std::make_shared<Conv1DLayer>(in_channels, out_channels, kernel_size, 
                                              stride, padding, name);
    }
    
    static LayerPtr create_sequential(const std::vector<LayerPtr>& layers, 
                                      const std::string& name = "") {
        return std::make_shared<SequentialLayer>(layers, name);
    }
};

// ============================================================
//  HELPER FUNCTIONS FOR TENSOR OPERATIONS
// ============================================================

// Element-wise multiplication
TensorPtr tensor_element_mul(TensorPtr a, TensorPtr b) {
    if (a->data.size() != b->data.size()) {
        throw std::runtime_error("Tensor sizes must match for element-wise multiplication");
    }
    
    auto result = std::make_shared<Tensor>(a->rows, a->cols);
    for (size_t i = 0; i < a->data.size(); ++i) {
        result->data[i] = a->data[i] * b->data[i];
    }
    return result;
}

// Tensor addition
TensorPtr tensor_add(TensorPtr a, TensorPtr b) {
    if (a->data.size() != b->data.size()) {
        throw std::runtime_error("Tensor sizes must match for addition");
    }
    
    auto result = std::make_shared<Tensor>(a->rows, a->cols);
    for (size_t i = 0; i < a->data.size(); ++i) {
        result->data[i] = a->data[i] + b->data[i];
    }
    return result;
}

// Leaky ReLU
TensorPtr tensor_leaky_relu(TensorPtr x, float alpha = 0.01f) {
    auto result = std::make_shared<Tensor>(x->rows, x->cols);
    for (size_t i = 0; i < x->data.size(); ++i) {
        result->data[i] = (x->data[i] > 0.0f) ? x->data[i] : x->data[i] * alpha;
    }
    return result;
}

// Leaky ReLU backward
TensorPtr tensor_leaky_relu_backward(TensorPtr grad_output, TensorPtr input, float alpha = 0.01f) {
    auto result = std::make_shared<Tensor>(grad_output->rows, grad_output->cols);
    for (size_t i = 0; i < grad_output->data.size(); ++i) {
        result->data[i] = (input->data[i] > 0.0f) ? grad_output->data[i] : grad_output->data[i] * alpha;
    }
    return result;
}

// Sigmoid backward
TensorPtr tensor_sigmoid_backward(TensorPtr grad_output, TensorPtr input) {
    auto result = std::make_shared<Tensor>(grad_output->rows, grad_output->cols);
    for (size_t i = 0; i < grad_output->data.size(); ++i) {
        float s = 1.0f / (1.0f + std::exp(-input->data[i]));
        result->data[i] = grad_output->data[i] * s * (1.0f - s);
    }
    return result;
}

// Tanh backward
TensorPtr tensor_tanh_backward(TensorPtr grad_output, TensorPtr input) {
    auto result = std::make_shared<Tensor>(grad_output->rows, grad_output->cols);
    for (size_t i = 0; i < grad_output->data.size(); ++i) {
        float t = std::tanh(input->data[i]);
        result->data[i] = grad_output->data[i] * (1.0f - t * t);
    }
    return result;
}

// GELU backward
TensorPtr tensor_gelu_backward(TensorPtr grad_output, TensorPtr input) {
    auto result = std::make_shared<Tensor>(grad_output->rows, grad_output->cols);
    for (size_t i = 0; i < grad_output->data.size(); ++i) {
        float x = input->data[i];
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        float gelu = x * 0.5f * (1.0f + std::tanh(inner));
        float grad = 0.5f * (1.0f + std::tanh(inner)) + 
                     x * 0.5f * (1.0f - std::tanh(inner) * std::tanh(inner)) * 
                     (0.7978845608f + 0.044715f * 3.0f * x * x);
        result->data[i] = grad_output->data[i] * grad;
    }
    return result;
}

// Swish backward
TensorPtr tensor_swish_backward(TensorPtr grad_output, TensorPtr input) {
    auto result = std::make_shared<Tensor>(grad_output->rows, grad_output->cols);
    for (size_t i = 0; i < grad_output->data.size(); ++i) {
        float x = input->data[i];
        float sigmoid = 1.0f / (1.0f + std::exp(-x));
        float swish = x * sigmoid;
        float grad = sigmoid + x * sigmoid * (1.0f - sigmoid);
        result->data[i] = grad_output->data[i] * grad;
    }
    return result;
}

// Matrix multiplication with transposed B
TensorPtr tensor_matmul_transposed(TensorPtr A, TensorPtr B) {
    if (A->cols != B->cols) {
        throw std::runtime_error("Matrix dimensions incompatible for multiplication");
    }
    
    auto result = std::make_shared<Tensor>(A->rows, B->rows);
    for (size_t i = 0; i < A->rows; ++i) {
        for (size_t j = 0; j < B->rows; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < A->cols; ++k) {
                sum += A->data[i * A->cols + k] * B->data[j * B->cols + k];
            }
            result->data[i * result->cols + j] = sum;
        }
    }
    return result;
}

// Softmax
TensorPtr tensor_softmax(TensorPtr x) {
    auto result = std::make_shared<Tensor>(x->rows, x->cols);
    
    for (size_t i = 0; i < x->rows; ++i) {
        float max_val = -FLT_MAX;
        for (size_t j = 0; j < x->cols; ++j) {
            if (x->data[i * x->cols + j] > max_val) {
                max_val = x->data[i * x->cols + j];
            }
        }
        
        float sum = 0.0f;
        for (size_t j = 0; j < x->cols; ++j) {
            result->data[i * x->cols + j] = std::exp(x->data[i * x->cols + j] - max_val);
            sum += result->data[i * x->cols + j];
        }
        
        for (size_t j = 0; j < x->cols; ++j) {
            result->data[i * x->cols + j] /= sum;
        }
    }
    
    return result;
}

// GELU activation
TensorPtr tensor_gelu(TensorPtr x) {
    auto result = std::make_shared<Tensor>(x->rows, x->cols);
    for (size_t i = 0; i < x->data.size(); ++i) {
        float xi = x->data[i];
        float x3 = xi * xi * xi;
        float inner = 0.7978845608f * (xi + 0.044715f * x3);
        result->data[i] = xi * 0.5f * (1.0f + std::tanh(inner));
    }
    return result;
}

// Swish activation
TensorPtr tensor_swish(TensorPtr x) {
    auto result = std::make_shared<Tensor>(x->rows, x->cols);
    for (size_t i = 0; i < x->data.size(); ++i) {
        result->data[i] = x->data[i] / (1.0f + std::exp(-x->data[i]));
    }
    return result;
}
