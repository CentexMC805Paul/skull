#pragma once
// ============================================================
//  SKULL MODEL ARCHITECTURES v1.0.0
//  Complete neural network models with configurable architectures
// ============================================================

#include "layers.h"
#include "optimizer.h"
#include "tokenizer.h"
#include "config.h"
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>

// ============================================================
//  BASE MODEL CLASS
// ============================================================

class Model {
public:
    std::string name;
    ModelArchitecture architecture;
    
    // Model components
    std::vector<LayerPtr> layers;
    LayerPtr embedding;
    LayerPtr output;
    
    // Configuration
    size_t vocab_size;
    size_t dim;
    size_t max_seq_len;
    
    // Training state
    bool training_mode;
    int current_epoch;
    int current_step;
    
    // Optimizer and scheduler
    OptimizerPtr optimizer;
    LRSchedulerPtr scheduler;
    
    Model(const std::string& name = "Model",
          ModelArchitecture architecture = ModelArchitecture::FEEDFORWARD)
        : name(name), architecture(architecture),
          vocab_size(0), dim(0), max_seq_len(512),
          training_mode(true), current_epoch(0), current_step(0) {}
    
    virtual ~Model() = default;
    
    // Forward pass
    virtual TensorPtr forward(TensorPtr input) = 0;
    
    // Backward pass
    virtual TensorPtr backward(TensorPtr grad_output) = 0;
    
    // Training step
    virtual float train_step(TensorPtr input, TensorPtr target) = 0;
    
    // Evaluation step
    virtual TensorPtr eval_step(TensorPtr input) = 0;
    
    // Generate text
    virtual std::string generate_text(const std::string& prompt, 
                                      size_t max_tokens = 100,
                                      float temperature = 0.8f) = 0;
    
    // Get parameter count
    virtual size_t param_count() const {
        size_t count = 0;
        for (const auto& layer : layers) {
            count += layer->param_count();
        }
        if (embedding) count += embedding->param_count();
        if (output) count += output->param_count();
        return count;
    }
    
    // Get all parameters
    virtual std::vector<TensorPtr> get_parameters() {
        std::vector<TensorPtr> params;
        for (const auto& layer : layers) {
            auto layer_params = layer->get_parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        if (embedding) {
            auto emb_params = embedding->get_parameters();
            params.insert(params.end(), emb_params.begin(), emb_params.end());
        }
        if (output) {
            auto out_params = output->get_parameters();
            params.insert(params.end(), out_params.begin(), out_params.end());
        }
        return params;
    }
    
    // Zero gradients
    virtual void zero_grad() {
        for (const auto& layer : layers) {
            layer->zero_grad();
        }
        if (embedding) embedding->zero_grad();
        if (output) output->zero_grad();
    }
    
    // Update parameters
    virtual void update_parameters() {
        if (scheduler) {
            optimizer->set_learning_rate(scheduler->get_lr());
            scheduler->step();
        }
        
        auto params = get_parameters();
        optimizer->step(params);
        optimizer->zero_grad(params);
    }
    
    // Set training mode
    virtual void set_training(bool training) {
        training_mode = training;
        for (const auto& layer : layers) {
            if (auto dropout = std::dynamic_pointer_cast<DropoutLayer>(layer)) {
                dropout->set_training(training);
            }
        }
    }
    
    // Save model
    virtual void save(const std::string& filepath) {
        std::ofstream out(filepath, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Could not open file for writing: " + filepath);
        }
        
        // Save model metadata
        out.write(name.c_str(), name.size());
        out.write("\n", 1);
        out.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
        out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        out.write(reinterpret_cast<const char*>(&max_seq_len), sizeof(max_seq_len));
        out.write(reinterpret_cast<const char*>(&architecture), sizeof(architecture));
        
        // Save parameters
        auto params = get_parameters();
        for (const auto& param : params) {
            out.write(reinterpret_cast<const char*>(&param->rows), sizeof(param->rows));
            out.write(reinterpret_cast<const char*>(&param->cols), sizeof(param->cols));
            out.write(reinterpret_cast<const char*>(param->data.data()), 
                     param->data.size() * sizeof(float));
        }
        
        out.close();
    }
    
    // Load model
    virtual void load(const std::string& filepath) {
        std::ifstream in(filepath, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Could not open file for reading: " + filepath);
        }
        
        // Load model metadata
        std::string loaded_name;
        std::getline(in, loaded_name);
        in.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
        in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        in.read(reinterpret_cast<char*>(&max_seq_len), sizeof(max_seq_len));
        in.read(reinterpret_cast<char*>(&architecture), sizeof(architecture));
        
        // Load parameters
        auto params = get_parameters();
        for (const auto& param : params) {
            size_t rows, cols;
            in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
            in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
            
            if (param->rows != rows || param->cols != cols) {
                throw std::runtime_error("Model parameter dimensions mismatch");
            }
            
            in.read(reinterpret_cast<char*>(param->data.data()), 
                   param->data.size() * sizeof(float));
        }
        
        in.close();
    }
    
    // Set optimizer
    virtual void set_optimizer(OptimizerPtr opt) {
        optimizer = opt;
    }
    
    // Set scheduler
    virtual void set_scheduler(LRSchedulerPtr sched) {
        scheduler = sched;
    }
    
    // Get model info
    virtual std::string info() const {
        std::stringstream ss;
        ss << "Model: " << name << "\n";
        ss << "  Architecture: ";
        switch (architecture) {
            case ModelArchitecture::FEEDFORWARD: ss << "Feedforward"; break;
            case ModelArchitecture::TRANSFORMER: ss << "Transformer"; break;
            case ModelArchitecture::LSTM: ss << "LSTM"; break;
            case ModelArchitecture::GRU: ss << "GRU"; break;
            case ModelArchitecture::CONV1D: ss << "Conv1D"; break;
            case ModelArchitecture::CUSTOM: ss << "Custom"; break;
        }
        ss << "\n";
        ss << "  Parameters: " << param_count() << "\n";
        ss << "  Vocab Size: " << vocab_size << "\n";
        ss << "  Dimension: " << dim << "\n";
        ss << "  Max Seq Len: " << max_seq_len << "\n";
        
        if (optimizer) {
            ss << "  Optimizer: " << optimizer->get_name() << "\n";
        }
        if (scheduler) {
            ss << "  Scheduler: " << scheduler->get_name() << "\n";
        }
        
        return ss.str();
    }
};

using ModelPtr = std::shared_ptr<Model>;

// ============================================================
//  FEEDFORWARD MODEL
// ============================================================

class FeedforwardModel : public Model {
public:
    size_t num_layers;
    size_t hidden_dim;
    ActivationFunction activation;
    
    FeedforwardModel(size_t vocab_size, size_t dim, size_t num_layers = 2,
                     size_t hidden_dim = 0, ActivationFunction activation = ActivationFunction::GELU,
                     const std::string& name = "Feedforward")
        : Model(name, ModelArchitecture::FEEDFORWARD),
          vocab_size(vocab_size), dim(dim),
          num_layers(num_layers), hidden_dim(hidden_dim ? hidden_dim : dim),
          activation(activation) {
        
        // Create embedding layer
        embedding = std::make_shared<LinearLayer>(vocab_size, dim, name + ".embedding");
        
        // Create hidden layers
        for (size_t i = 0; i < num_layers; ++i) {
            layers.push_back(std::make_shared<LinearLayer>(dim, hidden_dim, name + ".layer_" + std::to_string(i)));
            layers.push_back(std::make_shared<ActivationLayer>(activation, name + ".act_" + std::to_string(i)));
            dim = hidden_dim; // Update dim for next layer
        }
        
        // Create output layer
        output = std::make_shared<LinearLayer>(dim, vocab_size, name + ".output");
        
        // Set optimizer
        optimizer = OptimizerFactory::create(OptimizerType::ADAM, 0.001f);
    }
    
    TensorPtr forward(TensorPtr input) override {
        // Embedding lookup
        auto x = embedding->forward(input);
        
        // Pass through hidden layers
        for (const auto& layer : layers) {
            x = layer->forward(x);
        }
        
        // Output layer
        return output->forward(x);
    }
    
    TensorPtr backward(TensorPtr grad_output) override {
        // Backpropagate through output layer
        auto grad = output->backward(grad_output);
        
        // Backpropagate through hidden layers
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            grad = (*it)->backward(grad);
        }
        
        // Backpropagate through embedding
        return embedding->backward(grad);
    }
    
    float train_step(TensorPtr input, TensorPtr target) override {
        // Forward pass
        auto output = forward(input);
        
        // Compute loss (cross-entropy)
        float loss = 0.0f;
        for (size_t i = 0; i < output->rows; ++i) {
            for (size_t j = 0; j < output->cols; ++j) {
                // Simple cross-entropy (simplified)
                float target_val = (i * output->cols + j == target->data[0]) ? 1.0f : 0.0f;
                loss -= target_val * std::log(std::max(output->data[i * output->cols + j], 1e-10f));
            }
        }
        loss /= output->rows * output->cols;
        
        // Backward pass
        auto grad_output = std::make_shared<Tensor>(output->rows, output->cols);
        for (size_t i = 0; i < output->rows; ++i) {
            for (size_t j = 0; j < output->cols; ++j) {
                float target_val = (i * output->cols + j == target->data[0]) ? 1.0f : 0.0f;
                grad_output->data[i * output->cols + j] = (output->data[i * output->cols + j] - target_val) / 
                                                         (output->rows * output->cols);
            }
        }
        
        backward(grad_output);
        
        // Update parameters
        update_parameters();
        
        return loss;
    }
    
    TensorPtr eval_step(TensorPtr input) override {
        set_training(false);
        auto output = forward(input);
        set_training(true);
        return output;
    }
    
    std::string generate_text(const std::string& prompt, size_t max_tokens, float temperature) override {
        // Tokenize prompt
        auto tokenizer = std::make_shared<BPETokenizer>(vocab_size);
        auto input_ids = tokenizer->encode(prompt);
        
        std::string generated_text = prompt;
        
        for (size_t i = 0; i < max_tokens; ++i) {
            // Convert input to tensor
            auto input_tensor = std::make_shared<Tensor>(1, input_ids.size());
            for (size_t j = 0; j < input_ids.size(); ++j) {
                input_tensor->data[j] = static_cast<float>(input_ids[j]);
            }
            
            // Forward pass
            auto output = eval_step(input_tensor);
            
            // Sample next token
            float max_logit = -FLT_MAX;
            for (float val : output->data) {
                if (val > max_logit) max_logit = val;
            }
            
            std::vector<float> probs(output->data.size());
            float sum = 0.0f;
            for (size_t j = 0; j < output->data.size(); ++j) {
                probs[j] = std::exp((output->data[j] - max_logit) / temperature);
                sum += probs[j];
            }
            
            for (size_t j = 0; j < probs.size(); ++j) {
                probs[j] /= sum;
            }
            
            // Sample from distribution
            std::random_device rd;
            std::mt19937 gen(rd());
            std::discrete_distribution<> dist(probs.begin(), probs.end());
            size_t next_token = dist(gen);
            
            // Add to generated text
            auto token_str = tokenizer->decode({static_cast<int>(next_token)});
            generated_text += token_str;
            
            // Update input for next step
            input_ids = {static_cast<int>(next_token)};
        }
        
        return generated_text;
    }
};

// ============================================================
//  TRANSFORMER MODEL
// ============================================================

class TransformerModel : public Model {
public:
    size_t num_layers;
    size_t num_heads;
    size_t d_ff;
    float dropout_p;
    
    LayerPtr positional_embedding;
    std::vector<LayerPtr> transformer_layers;
    LayerPtr final_norm;
    
    TransformerModel(size_t vocab_size, size_t dim, size_t num_layers = 6,
                     size_t num_heads = 8, size_t d_ff = 2048, float dropout_p = 0.1f,
                     const std::string& name = "Transformer")
        : Model(name, ModelArchitecture::TRANSFORMER),
          vocab_size(vocab_size), dim(dim),
          num_layers(num_layers), num_heads(num_heads),
          d_ff(d_ff), dropout_p(dropout_p) {
        
        // Create embedding layer
        embedding = std::make_shared<LinearLayer>(vocab_size, dim, name + ".embedding");
        
        // Create positional embedding (simple sinusoidal)
        positional_embedding = std::make_shared<LinearLayer>(max_seq_len, dim, name + ".pos_embedding");
        
        // Create transformer layers
        for (size_t i = 0; i < num_layers; ++i) {
            transformer_layers.push_back(
                std::make_shared<TransformerLayer>(dim, num_heads, d_ff, dropout_p, 
                                                  name + ".layer_" + std::to_string(i)));
        }
        
        // Final layer norm
        final_norm = std::make_shared<LayerNormLayer>(dim, name + ".final_norm");
        
        // Output layer
        output = std::make_shared<LinearLayer>(dim, vocab_size, name + ".output");
        
        // Set optimizer
        optimizer = OptimizerFactory::create(OptimizerType::ADAM, 0.001f);
        scheduler = LRSchedulerFactory::create("cosine", 0.001f, 1000, 0.0f);
    }
    
    TensorPtr forward(TensorPtr input) override {
        // Embedding lookup
        auto x = embedding->forward(input);
        
        // Add positional embedding
        auto pos_emb = positional_embedding->forward(input);
        x = tensor_add(x, pos_emb);
        
        // Pass through transformer layers
        for (const auto& layer : transformer_layers) {
            x = layer->forward(x);
        }
        
        // Final layer norm
        x = final_norm->forward(x);
        
        // Output layer
        return output->forward(x);
    }
    
    TensorPtr backward(TensorPtr grad_output) override {
        // Backpropagate through output layer
        auto grad = output->backward(grad_output);
        
        // Backpropagate through final norm
        grad = final_norm->backward(grad);
        
        // Backpropagate through transformer layers
        for (auto it = transformer_layers.rbegin(); it != transformer_layers.rend(); ++it) {
            grad = (*it)->backward(grad);
        }
        
        // Backpropagate through positional embedding
        // (This is simplified - in practice we'd need to handle positional embedding gradients)
        
        // Backpropagate through embedding
        return embedding->backward(grad);
    }
    
    float train_step(TensorPtr input, TensorPtr target) override {
        // Forward pass
        auto output = forward(input);
        
        // Compute loss (cross-entropy)
        float loss = 0.0f;
        for (size_t i = 0; i < output->rows; ++i) {
            for (size_t j = 0; j < output->cols; ++j) {
                float target_val = (i * output->cols + j == target->data[0]) ? 1.0f : 0.0f;
                loss -= target_val * std::log(std::max(output->data[i * output->cols + j], 1e-10f));
            }
        }
        loss /= output->rows * output->cols;
        
        // Backward pass
        auto grad_output = std::make_shared<Tensor>(output->rows, output->cols);
        for (size_t i = 0; i < output->rows; ++i) {
            for (size_t j = 0; j < output->cols; ++j) {
                float target_val = (i * output->cols + j == target->data[0]) ? 1.0f : 0.0f;
                grad_output->data[i * output->cols + j] = (output->data[i * output->cols + j] - target_val) / 
                                                         (output->rows * output->cols);
            }
        }
        
        backward(grad_output);
        
        // Update parameters
        update_parameters();
        
        return loss;
    }
    
    TensorPtr eval_step(TensorPtr input) override {
        set_training(false);
        auto output = forward(input);
        set_training(true);
        return output;
    }
    
    std::string generate_text(const std::string& prompt, size_t max_tokens, float temperature) override {
        // Tokenize prompt
        auto tokenizer = std::make_shared<BPETokenizer>(vocab_size);
        auto input_ids = tokenizer->encode(prompt);
        
        std::string generated_text = prompt;
        
        for (size_t i = 0; i < max_tokens; ++i) {
            // Convert input to tensor
            auto input_tensor = std::make_shared<Tensor>(1, input_ids.size());
            for (size_t j = 0; j < input_ids.size(); ++j) {
                input_tensor->data[j] = static_cast<float>(input_ids[j]);
            }
            
            // Forward pass
            auto output = eval_step(input_tensor);
            
            // Sample next token (only use last token's output)
            size_t last_pos = output->rows * output->cols - output->cols;
            
            float max_logit = -FLT_MAX;
            for (size_t j = 0; j < output->cols; ++j) {
                if (output->data[last_pos + j] > max_logit) {
                    max_logit = output->data[last_pos + j];
                }
            }
            
            std::vector<float> probs(output->cols);
            float sum = 0.0f;
            for (size_t j = 0; j < output->cols; ++j) {
                probs[j] = std::exp((output->data[last_pos + j] - max_logit) / temperature);
                sum += probs[j];
            }
            
            for (size_t j = 0; j < probs.size(); ++j) {
                probs[j] /= sum;
            }
            
            // Sample from distribution
            std::random_device rd;
            std::mt19937 gen(rd());
            std::discrete_distribution<> dist(probs.begin(), probs.end());
            size_t next_token = dist(gen);
            
            // Add to generated text
            auto token_str = tokenizer->decode({static_cast<int>(next_token)});
            generated_text += token_str;
            
            // Update input for next step
            input_ids = {static_cast<int>(next_token)};
        }
        
        return generated_text;
    }
};

// ============================================================
//  LSTM MODEL
// ============================================================

class LSTMModel : public Model {
public:
    size_t num_layers;
    size_t hidden_size;
    
    std::vector<LayerPtr> lstm_layers;
    LayerPtr final_norm;
    
    LSTMModel(size_t vocab_size, size_t dim, size_t num_layers = 2,
              size_t hidden_size = 0, const std::string& name = "LSTM")
        : Model(name, ModelArchitecture::LSTM),
          vocab_size(vocab_size), dim(dim),
          num_layers(num_layers), hidden_size(hidden_size ? hidden_size : dim) {
        
        // Create embedding layer
        embedding = std::make_shared<LinearLayer>(vocab_size, dim, name + ".embedding");
        
        // Create LSTM layers
        for (size_t i = 0; i < num_layers; ++i) {
            lstm_layers.push_back(
                std::make_shared<LSTMLayer>(i == 0 ? dim : hidden_size, hidden_size, 
                                             name + ".layer_" + std::to_string(i)));
        }
        
        // Final layer norm
        final_norm = std::make_shared<LayerNormLayer>(hidden_size, name + ".final_norm");
        
        // Output layer
        output = std::make_shared<LinearLayer>(hidden_size, vocab_size, name + ".output");
        
        // Set optimizer
        optimizer = OptimizerFactory::create(OptimizerType::ADAM, 0.001f);
    }
    
    TensorPtr forward(TensorPtr input) override {
        // Embedding lookup
        auto x = embedding->forward(input);
        
        // Pass through LSTM layers
        for (const auto& layer : lstm_layers) {
            x = layer->forward(x);
        }
        
        // Final layer norm
        x = final_norm->forward(x);
        
        // Output layer
        return output->forward(x);
    }
    
    TensorPtr backward(TensorPtr grad_output) override {
        // Backpropagate through output layer
        auto grad = output->backward(grad_output);
        
        // Backpropagate through final norm
        grad = final_norm->backward(grad);
        
        // Backpropagate through LSTM layers
        for (auto it = lstm_layers.rbegin(); it != lstm_layers.rend(); ++it) {
            grad = (*it)->backward(grad);
        }
        
        // Backpropagate through embedding
        return embedding->backward(grad);
    }
    
    float train_step(TensorPtr input, TensorPtr target) override {
        // Reset LSTM states
        for (const auto& layer : lstm_layers) {
            if (auto lstm = std::dynamic_pointer_cast<LSTMLayer>(layer)) {
                lstm->reset_state();
            }
        }
        
        // Forward pass
        auto output = forward(input);
        
        // Compute loss (cross-entropy)
        float loss = 0.0f;
        for (size_t i = 0; i < output->rows; ++i) {
            for (size_t j = 0; j < output->cols; ++j) {
                float target_val = (i * output->cols + j == target->data[0]) ? 1.0f : 0.0f;
                loss -= target_val * std::log(std::max(output->data[i * output->cols + j], 1e-10f));
            }
        }
        loss /= output->rows * output->cols;
        
        // Backward pass
        auto grad_output = std::make_shared<Tensor>(output->rows, output->cols);
        for (size_t i = 0; i < output->rows; ++i) {
            for (size_t j = 0; j < output->cols; ++j) {
                float target_val = (i * output->cols + j == target->data[0]) ? 1.0f : 0.0f;
                grad_output->data[i * output->cols + j] = (output->data[i * output->cols + j] - target_val) / 
                                                         (output->rows * output->cols);
            }
        }
        
        backward(grad_output);
        
        // Update parameters
        update_parameters();
        
        return loss;
    }
    
    TensorPtr eval_step(TensorPtr input) override {
        // Reset LSTM states
        for (const auto& layer : lstm_layers) {
            if (auto lstm = std::dynamic_pointer_cast<LSTMLayer>(layer)) {
                lstm->reset_state();
            }
        }
        
        set_training(false);
        auto output = forward(input);
        set_training(true);
        return output;
    }
    
    std::string generate_text(const std::string& prompt, size_t max_tokens, float temperature) override {
        // Tokenize prompt
        auto tokenizer = std::make_shared<BPETokenizer>(vocab_size);
        auto input_ids = tokenizer->encode(prompt);
        
        std::string generated_text = prompt;
        
        for (size_t i = 0; i < max_tokens; ++i) {
            // Convert input to tensor
            auto input_tensor = std::make_shared<Tensor>(1, input_ids.size());
            for (size_t j = 0; j < input_ids.size(); ++j) {
                input_tensor->data[j] = static_cast<float>(input_ids[j]);
            }
            
            // Forward pass
            auto output = eval_step(input_tensor);
            
            // Sample next token (only use last token's output)
            size_t last_pos = output->rows * output->cols - output->cols;
            
            float max_logit = -FLT_MAX;
            for (size_t j = 0; j < output->cols; ++j) {
                if (output->data[last_pos + j] > max_logit) {
                    max_logit = output->data[last_pos + j];
                }
            }
            
            std::vector<float> probs(output->cols);
            float sum = 0.0f;
            for (size_t j = 0; j < output->cols; ++j) {
                probs[j] = std::exp((output->data[last_pos + j] - max_logit) / temperature);
                sum += probs[j];
            }
            
            for (size_t j = 0; j < probs.size(); ++j) {
                probs[j] /= sum;
            }
            
            // Sample from distribution
            std::random_device rd;
            std::mt19937 gen(rd());
            std::discrete_distribution<> dist(probs.begin(), probs.end());
            size_t next_token = dist(gen);
            
            // Add to generated text
            auto token_str = tokenizer->decode({static_cast<int>(next_token)});
            generated_text += token_str;
            
            // Update input for next step
            input_ids = {static_cast<int>(next_token)};
        }
        
        return generated_text;
    }
};

// ============================================================
//  MODEL FACTORY
// ============================================================

class ModelFactory {
public:
    static ModelPtr create(ModelArchitecture architecture,
                          size_t vocab_size, size_t dim,
                          const std::string& name = "Model",
                          size_t num_layers = 2, size_t num_heads = 8,
                          size_t d_ff = 2048, size_t hidden_size = 0,
                          float dropout_p = 0.1f) {
        switch (architecture) {
            case ModelArchitecture::FEEDFORWARD:
                return std::make_shared<FeedforwardModel>(vocab_size, dim, num_layers, 
                                                           hidden_size, ActivationFunction::GELU, name);
            case ModelArchitecture::TRANSFORMER:
                return std::make_shared<TransformerModel>(vocab_size, dim, num_layers, 
                                                           num_heads, d_ff, dropout_p, name);
            case ModelArchitecture::LSTM:
                return std::make_shared<LSTMModel>(vocab_size, dim, num_layers, 
                                                    hidden_size, name);
            case ModelArchitecture::GRU:
            case ModelArchitecture::CONV1D:
            case ModelArchitecture::CUSTOM:
            default:
                throw std::runtime_error("Unsupported architecture");
        }
    }
    
    static ModelPtr create_from_string(const std::string& arch_name,
                                       size_t vocab_size, size_t dim,
                                       const std::string& name = "Model",
                                       size_t num_layers = 2, size_t num_heads = 8,
                                       size_t d_ff = 2048, size_t hidden_size = 0,
                                       float dropout_p = 0.1f) {
        if (arch_name == "feedforward" || arch_name == "Feedforward" || arch_name == "FF") {
            return create(ModelArchitecture::FEEDFORWARD, vocab_size, dim, name, 
                         num_layers, num_heads, d_ff, hidden_size, dropout_p);
        } else if (arch_name == "transformer" || arch_name == "Transformer") {
            return create(ModelArchitecture::TRANSFORMER, vocab_size, dim, name, 
                         num_layers, num_heads, d_ff, hidden_size, dropout_p);
        } else if (arch_name == "lstm" || arch_name == "LSTM") {
            return create(ModelArchitecture::LSTM, vocab_size, dim, name, 
                         num_layers, num_heads, d_ff, hidden_size, dropout_p);
        } else {
            return create(ModelArchitecture::FEEDFORWARD, vocab_size, dim, name, 
                         num_layers, num_heads, d_ff, hidden_size, dropout_p);
        }
    }
};

// ============================================================
//  MODEL CONFIGURATION
// ============================================================

struct ModelConfig {
    std::string name = "MyModel";
    ModelArchitecture architecture = ModelArchitecture::TRANSFORMER;
    size_t vocab_size = 256;
    size_t dim = 512;
    size_t num_layers = 6;
    size_t num_heads = 8;
    size_t d_ff = 2048;
    size_t hidden_size = 0;
    float dropout_p = 0.1f;
    
    // Training configuration
    OptimizerType optimizer_type = OptimizerType::ADAM;
    float learning_rate = 0.001f;
    std::string scheduler_type = "cosine";
    
    // Precision
    PrecisionMode precision = PrecisionMode::FP32;
    
    // GPU
    GPUBackend gpu_backend = GPUBackend::AUTO;
    
    // Save/load
    std::string save_path = "";
    std::string load_path = "";
    
    // Create model from config
    ModelPtr create_model() const {
        auto model = ModelFactory::create(architecture, vocab_size, dim, name,
                                         num_layers, num_heads, d_ff, hidden_size, dropout_p);
        
        // Set optimizer
        model->optimizer = OptimizerFactory::create(optimizer_type, learning_rate);
        
        // Set scheduler
        model->scheduler = LRSchedulerFactory::create(scheduler_type, learning_rate);
        
        return model;
    }
};

// ============================================================
//  MODEL UTILITIES
// ============================================================

// Train a model
inline void train_model(ModelPtr model, const std::vector<std::string>& train_files,
                       size_t epochs = 10, size_t batch_size = 32, 
                       const std::string& save_path = "") {
    
    std::cout << "Training model: " << model->name << "\n";
    std::cout << "Epochs: " << epochs << "\n";
    std::cout << "Batch size: " << batch_size << "\n";
    std::cout << "Parameters: " << model->param_count() << "\n";
    
    // Load and preprocess data
    std::vector<TensorPtr> train_data;
    for (const auto& file : train_files) {
        // Load file and convert to tensor
        // This is a placeholder - actual implementation would use the tokenizer
        auto tensor = tensor_zeros(1, 1);
        train_data.push_back(tensor);
    }
    
    // Training loop
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        size_t num_batches = 0;
        
        for (size_t i = 0; i < train_data.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, train_data.size());
            
            // Create batch
            auto batch_input = tensor_zeros(batch_size, train_data[i]->cols);
            auto batch_target = tensor_zeros(batch_size, 1);
            
            for (size_t j = i; j < end; ++j) {
                // Copy data to batch
                for (size_t k = 0; k < train_data[j]->cols; ++k) {
                    batch_input->data[(j - i) * batch_input->cols + k] = train_data[j]->data[k];
                }
                // Set target (simplified)
                batch_target->data[j - i] = 0; // Placeholder
            }
            
            // Train step
            float loss = model->train_step(batch_input, batch_target);
            epoch_loss += loss;
            num_batches++;
        }
        
        epoch_loss /= num_batches;
        std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                  << " - Loss: " << epoch_loss << "\n";
    }
    
    // Save model if path is provided
    if (!save_path.empty()) {
        model->save(save_path);
        std::cout << "Model saved to: " << save_path << "\n";
    }
}

// Evaluate a model
inline void evaluate_model(ModelPtr model, const std::vector<std::string>& eval_files,
                          size_t batch_size = 32) {
    
    std::cout << "Evaluating model: " << model->name << "\n";
    
    model->set_training(false);
    
    // Load and preprocess data
    std::vector<TensorPtr> eval_data;
    for (const auto& file : eval_files) {
        // Load file and convert to tensor
        auto tensor = tensor_zeros(1, 1);
        eval_data.push_back(tensor);
    }
    
    // Evaluation loop
    float total_loss = 0.0f;
    size_t num_batches = 0;
    
    for (size_t i = 0; i < eval_data.size(); i += batch_size) {
        size_t end = std::min(i + batch_size, eval_data.size());
        
        // Create batch
        auto batch_input = tensor_zeros(batch_size, eval_data[i]->cols);
        auto batch_target = tensor_zeros(batch_size, 1);
        
        for (size_t j = i; j < end; ++j) {
            // Copy data to batch
            for (size_t k = 0; k < eval_data[j]->cols; ++k) {
                batch_input->data[(j - i) * batch_input->cols + k] = eval_data[j]->data[k];
            }
            // Set target (simplified)
            batch_target->data[j - i] = 0; // Placeholder
        }
        
        // Eval step
        auto output = model->eval_step(batch_input);
        
        // Compute loss (simplified)
        float loss = 0.0f;
        for (size_t j = 0; j < output->data.size(); ++j) {
            loss += output->data[j];
        }
        loss /= output->data.size();
        
        total_loss += loss;
        num_batches++;
    }
    
    total_loss /= num_batches;
    std::cout << "Evaluation loss: " << total_loss << "\n";
    
    model->set_training(true);
}

// Generate text with a model
inline std::string generate_text(ModelPtr model, const std::string& prompt,
                                size_t max_tokens = 100, float temperature = 0.8f) {
    return model->generate_text(prompt, max_tokens, temperature);
}
