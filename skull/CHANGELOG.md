# Skull Changelog

All notable changes to the Skull project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

Bug-fix and honesty release. The working program (language, trainer, generator) is the same
size as before; most of the "v1.0.0" feature list below was announced but never worked.

### 🐛 Fixed

- **Memory leak in training.** Autograd closures captured their own tensor (reference cycle), and
  every training step allocated a transposed copy of `W_out`. Peak memory grew linearly with
  steps: 3.7 GB after 50 epochs on a 281-byte file. The trainer now runs without an autograd
  graph: **constant ~10 MB, and 50 epochs take ~0.5 s instead of ~25 s** (the loss curve was
  identical to the old implementation step for step at the time of this rewrite).
- **Autograd counted gradients twice** for tensors used more than once (e.g. `h + h`).
  `backward()` now walks the graph in topological order.
- **Training only used the first 500 tokens** of any data file. Epochs now cover all tokens
  (new option `steps` limits an epoch to a random window for very large files).
- `batch` was parsed but ignored; it is now real mini-batch SGD (mean gradient).
- **Generator**: out-of-bounds read when `vocab < 256` or the prompt contained a byte `>= vocab`;
  `temperature = 0` produced NaN and empty output (now greedy decoding); truncated or corrupt
  weight files were accepted silently (header and file size are validated now).
- **BPE** training could not be used for generation (tokenizer was not saved). The BPE merges are
  now stored in the weights file (optional trailing `TOKB` section; older files stay readable).
- **Errors now end the program with exit code 1** (missing data file, bad weights, empty file, …).
  Before, `train` with a missing file printed an error and exited with 0.
- **Interpreter**: `return` outside a function crashed (abort); unbounded recursion crashed
  (segfault) — now a clean error at 1000 nested calls, with the interpreter running on a thread
  with a large stack; `1 == "1"` was `true`; wrong value types (`data = 5`) were silently turned
  into empty values; unknown fields (typos like `epoch`) and unsupported ones (`layers`, `heads`)
  were silently ignored — they now produce a warning.
- A data file without an extension threw an obscure `out_of_range` exception.
- **Build**: CMake required OpenCL even with `-DSKULL_USE_OPENCL=OFF` (hard-coded `-lOpenCL`),
  so the documented default build failed on machines without OpenCL. OpenCL is now optional
  and off by default. Duplicate CUDA blocks removed, Visual Studio builds put `skull.exe` into
  `build/` as documented, `build.bat` no longer falls through into its help text, the fragile
  `vcvars` handling is gone (CMake picks the newest Visual Studio itself), and the version number
  lives in one place (`src/version.h`).
  AVX2 can be switched off (`-DSKULL_ENABLE_AVX2=OFF`) for older CPUs.

### ➕ Added

- **Validation.** `train` holds back the last 10 % of the data (`val`, at most 50 000 tokens) and
  reports validation loss and perplexity every epoch. The weights with the best validation loss are
  saved (not the last epoch's), `patience = N` stops after N epochs without improvement. Skipped with a
  note when there is too little data. New builtins `last_loss()`, `last_val_loss()`,
  `last_best_epoch()`. The training loop is now one shared frame (`run_training`) with the two models
  behind a common interface; `tests/traincheck.cpp` tests it exactly with a scripted mock model
  (mutation-tested).
- **Transformer model** (`context > 1`): causal multi-head self-attention, pre-LayerNorm, GELU MLP,
  any number of layers, hand-written backward pass (`src/transformer.h`), Adam with gradient
  clipping. Selected via the model fields `context`, `heads`, `layers`. The backward pass is
  verified against finite differences for every parameter (`tests/gradcheck.cpp`; mutation-tested:
  it catches a missing attention scale, wrong LayerNorm/GELU derivatives, a missing residual
  path and a removed causal mask). New weights format `SKULT` (versioned, validated on load);
  the bigram format `SKULL` stays readable.
- `train { out = "..." }` sets the weights path; `examples/transformer.skull`.
- Language: `else if`, `and` / `or` / `not` (also `!`) with short-circuit evaluation, and `%`
  (Python semantics: the result takes the sign of the divisor).
- Regression tests (`ctest`, 46 cases) for all of the above, run by GitHub Actions on Linux
  (gcc, clang, AddressSanitizer/UBSan), macOS and Windows. Run locally with `./build.sh --test`.
- Builtin `live_tensors()` (number of live tensors; used by the leak test).
- `-DSKULL_SANITIZE=ON` build option.

### 🔄 Changed

- **Random numbers are now identical on every platform** (`src/rng.h`). `std::mt19937` is fixed by
  the C++ standard, but `std::uniform_real_distribution` / `std::uniform_int_distribution` are not:
  libstdc++, libc++ (macOS) and MSVC return different numbers for the same seed, so the same
  `train` produced different initial weights and training windows on different systems. Skull now
  converts the raw generator output itself (bit-identical to NumPy's `rand()` for the same seed;
  `tests/rngcheck.cpp`). Trained weights differ from earlier builds for the same data.
- **Transformer training uses a learning-rate schedule** (short linear warm-up, then cosine decay
  to 10 % of `rate`). With a constant rate the loss kept jittering around its floor and about one
  run in twenty ended with a visibly worse model; `rate` is now the peak learning rate.
- Tests are self-contained: each training test writes its own weights file. They used to share
  files, and this **did** fail on macOS CI (`train_all_data` read weights overwritten by
  `train_steps_window` when `ctest` ran tests in parallel). The examples that share one file are
  now serialised.

- `gpu = true` now says openly that training still runs on the CPU (it only detects the device).
- `examples/test.skull` uses a small model and the bundled data instead of a missing file.
- Banner/version strings of trainer and generator no longer disagree (`v0.5.0` / `v0.7.0` / `v1.0.0`).

### 📦 Moved / Removed

- The unfinished "v1.0.0" draft code (`layers.h`, `model.h`, `optimizer.h`, `parallel.h`, `config.h`,
  `gpu_cuda.cu`, `skull_python.cpp`, `python/`, `setup.py`) moved to `skull/experimental/`. It never
  compiled and was never connected to the CLI; see `experimental/README.md` for its status.
- `./build.sh --cuda` / `build.bat --cuda` removed (the CUDA code never compiled).
- `src/tensor_info.h` (an unused duplicate fragment).

---

## [v1.0.0] - 2025-01-XX

> **Note:** The feature list below describes the *plan* for v1.0.0. Of it, only the CMake build,
> the multi-format tokenizer and BPE exist in the working program. Transformer/LSTM/GRU/Conv1D
> models, the extra optimizers and schedulers, CUDA, Metal, WordPiece, mixed precision, early stopping,
> checkpointing, memory pooling, multi-threading and the Python bindings are **not** implemented
> (partly drafts in `experimental/`, partly only enum names).

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
