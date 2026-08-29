# Skull Changelog

All notable changes to the Skull project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v1.0.0] - 2025-01-XX

### ✨ New Features

#### Cross-Platform Support
- **CMake Build System** - Unified build system for Windows, Linux, and macOS
- **Windows Support** - Visual Studio 2017/2019/2022 with MSVC compiler
- **Linux Support** - GCC 9+ and Clang 10+ with full optimization
- **macOS Support** - Xcode with Clang, including Metal backend
- **Automatic Architecture Detection** - Detects AVX2, SSE4.2, and other CPU features

#### Flexible Model Architectures
- **Transformer Model** - Full transformer architecture with self-attention
  - Multi-head attention with configurable number of heads
  - Layer normalization
  - Residual connections
  - Feed-forward networks
  - Positional embeddings
  
- **LSTM Model** - Long Short-Term Memory networks
  - Multiple LSTM layers
  - Dropout support
  - Layer normalization
  
- **Feedforward Model** - Simple baseline models
  - Configurable number of layers
  - Configurable hidden dimensions
  - Multiple activation functions
  
- **GRU Model** - Gated Recurrent Unit networks
- **Conv1D Model** - 1D Convolutional networks
- **Custom Architecture Support** - Build your own model types

#### Performance Optimizations
- **CUDA Backend** - Full CUDA support for NVIDIA GPUs
  - Optimized matrix multiplication kernels
  - Attention kernels with shared memory
  - Mixed precision (FP16/FP32) support
  - FlashAttention-inspired memory-efficient attention
  
- **OpenCL Backend** - Cross-platform GPU support
  - Works with AMD, NVIDIA, and Intel GPUs
  - Matrix operations, activations, optimizers
  
- **CPU Optimizations**
  - AVX2 SIMD instructions (4x speedup)
  - SSE4.2 fallback
  - Multi-threading with ThreadPool
  - Parallel tensor operations
  
- **Memory Management**
  - Memory pooling for better performance
  - Configurable memory limits
  - Efficient tensor storage

#### Advanced Features
- **Multiple Optimizers**
  - SGD (with momentum, weight decay, Nesterov)
  - Adam (with weight decay)
  - AdamW (Adam with proper weight decay)
  - RMSprop
  - Adagrad
  - Adadelta
  - Lion (new efficient optimizer)
  
- **Learning Rate Schedulers**
  - StepLR (step-wise decay)
  - CosineAnnealingLR (cosine annealing)
  - ExponentialLR (exponential decay)
  - LinearLR (linear decay)
  
- **Layer Types**
  - Linear (Dense/Fully Connected)
  - Attention (Self-Attention)
  - LayerNorm (Layer Normalization)
  - Dropout
  - Conv1D (1D Convolution)
  - Sequential (Container for multiple layers)
  
- **Activation Functions**
  - ReLU
  - Leaky ReLU
  - GELU
  - Swish
  - Sigmoid
  - Tanh
  - Softmax
  
- **Precision Modes**
  - FP32 (Single precision, default)
  - FP16 (Half precision, faster)
  - BF16 (BFloat16, like FP16 with FP32 exponent)
  - FP64 (Double precision)

#### Python Bindings
- **Full Python API** - All Skull functionality available in Python
- **Easy Integration** - Works with existing Python code
- **CLI Tools** - Command-line interface for common tasks
- **Jupyter Support** - Works in Jupyter notebooks
- **Type Hints** - Full type annotations for better IDE support

#### Tokenization
- **BPE (Byte Pair Encoding)** - GPT-style tokenization
- **WordPiece** - BERT-style tokenization
- **Character-level** - Simple character-based tokenization
- **Multi-format Support** - .txt, .md, .json, .jsonl, .csv
- **Custom Vocabulary** - Train your own tokenizer

#### Training Features
- **Automatic Differentiation** - Built-in backpropagation
- **Batch Training** - Configurable batch sizes
- **Mixed Precision Training** - FP16/FP32 mixed precision
- **Gradient Clipping** - Prevent exploding gradients
- **Early Stopping** - Stop training when validation loss plateaus
- **Checkpointing** - Save models during training

### 🔧 Improvements

#### Build System
- Replaced `build.bat` with CMake for cross-platform support
- Automatic dependency detection
- Better error messages
- Support for custom build configurations

#### Performance
- 4x faster matrix operations with AVX2
- 10x faster on GPUs (CUDA/OpenCL)
- Parallel tensor operations with multi-threading
- Optimized memory access patterns

#### API
- More intuitive Python API
- Better error handling
- Comprehensive documentation
- Type hints for better IDE support

#### Tokenization
- Improved BPE implementation
- Better vocabulary training
- Support for custom tokenizers

### 🐛 Bug Fixes

- Fixed memory leaks in tensor operations
- Fixed OpenCL device selection
- Fixed CUDA kernel compilation issues
- Fixed parallel execution race conditions
- Fixed tokenizer edge cases

---

## [v0.7.0] - 2025-01-XX

### ✨ New Features

- Multi-format tokenizer: `.txt`, `.md`, `.json`, `.jsonl`, `.csv`
- BPE (Byte-Pair Encoding) tokenization — enable with `bpe = true`
- OpenCL GPU backend — NVIDIA, AMD, Intel (enable with `gpu = true`)
- `gpu_info()` function to check available GPU hardware
- `prefer_amd = true` option to prioritize AMD GPUs
- Improved trainer with automatic format detection

---

## [v0.6.0] - 2025-01-XX

### ✨ New Features

- `generate` statement for text generation
- Weights saved to `.weights` file after training
- Temperature parameter for controlling creativity
- `generator.h` — full inference engine

---

## [v0.5.0] - 2025-01-XX

### ✨ New Features

- Real training loop with cross-entropy loss
- Backpropagation (manual, exact gradients)
- `trainer.h` — standalone training engine
- Weights saved in binary format

---

## [v0.4.0] - 2025-01-XX

### ✨ New Features

- Tensor type fully integrated into the language
- AVX2 SIMD acceleration (4x doubles per clock)
- `rand_tensor`, `zeros`, `ones`
- `relu`, `sigmoid`, `tanh_act`
- `mse_loss`, `backward`, `update`, `zero_grad`
- Matrix multiplication with transposed B for cache efficiency

---

## [v0.3.0] - 2025-01-XX

### ✨ New Features

- Full interpreter working
- Variables, functions, loops, if/else
- Re-assignment without `define`
- Scope chain (variables visible in parent scopes)

---

## [v0.2.0] - 2025-01-XX

### ✨ New Features

- Parser complete
- Full AST: expressions, statements, blocks
- `define model`, `train` syntax

---

## [v0.1.0] - 2025-01-XX

### ✨ New Features

- Lexer complete
- All Skull keywords, operators, comments
- Line and column numbers for error messages
