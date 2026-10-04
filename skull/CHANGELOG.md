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
- **BPE training silently stopped at 500 merges**: `bpe_vocab = 1000` produced only 756 tokens
  (256 bytes + 500 merges). The limit is gone; training now ends only when the vocabulary reaches
  `bpe_vocab` or no pair occurs at least twice any more (in that case it says so instead of
  stopping quietly).
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

### ⚡ Performance (Transformer training)

Measured on Tiny Shakespeare (1.1 MB), dim 64, 2 layers, 4 heads, context 64, batch 8
(4-vCPU VM; its 4th vCPU is over-subscribed, so more than ~3 threads do not scale here):

| | tokens/s |
|---|---|
| before | ~5 000 |
| faster kernels, 1 thread | ~10 000 (1.9-2.1x) |
| + 2 threads | ~17 000 (3.4x) |

- **Backward pass 3.2x faster** (profiled with callgrind: `linear_back` was 55 % of the work). The
  `dX = dY W^T` loop was a floating-point reduction, which the compiler may not reorder without
  `-ffast-math`, so it stayed scalar (3-4 GFLOP/s instead of 12-15). It now uses dot products with 16
  fixed partial sums (a fixed order, so identical on every system, but SIMD-friendly); `dW` processes 4
  rows at a time. The same dot product speeds up attention; the GELU's `tanh` is reused from the
  forward pass. (A register-tiled matrix multiply I tried first was *slower* and was dropped.)
- **Multi-threading** (`threads`): the sequences of a batch run in parallel, as does validation and the
  Adam step. Work is assigned to fixed *slots* (sequence `b` -> slot `b % 8`) and partial sums are
  combined in a fixed order, so **the trained weights are bit-identical for any thread count**
  (verified for 1, 2, 3, 4 and 8 threads, `tests/traincheck.cpp`). `-DSKULL_TSAN=ON` builds with
  ThreadSanitizer: no data race in the suite, while a deliberately injected one is reported.
- The transformer now separates read-only parameters from a per-call `Workspace`, which is what makes
  the above thread-safe.

### 📦 Added (distribution)

- **Prebuilt packages** (`.github/workflows/release.yml`): Linux x86-64 (fully static, runs on any
  distribution), macOS Apple Silicon, Windows x86-64 (static runtime, no redistributable needed), each with an
  AVX2 build and a `-compat` build for older CPUs (except macOS). Pushing a tag `v<version>` builds them,
  runs the whole test suite, unpacks every archive and smoke-tests it, and publishes a GitHub release with
  `SHA256SUMS.txt`. Manual runs only build and test. Intel Macs and Linux arm64 are not covered (build from
  source). The packages are unsigned: macOS and Windows show a warning (documented in the tutorial).
- `-DSKULL_STATIC_RUNTIME=ON` (fully static on Linux, `/MT` on MSVC); CI builds and tests it on every push.
- **CPU check at startup**: an AVX2 build on a CPU without AVX2 used to die with "Illegal instruction"; it now
  says what to do (use the `-compat` package or `-DSKULL_ENABLE_AVX2=OFF`). The failure path itself could not
  be exercised on an AVX2 machine; the check only reads the CPU feature flags.
- `docs/TUTORIAL.md` (German): download, language basics, first model, reading loss/perplexity, better
  training, pause/resume, sampling, troubleshooting. Every command and every quoted number was run.

### ➕ Added (language)

- **Lists and text indexing**: list literals `[1, "a", [2]]`, `liste[i]` (negative from the end),
  `liste[i] = wert`, `for x in liste`, `for c in text`, `liste + liste`, deep `==`; builtins `len`, `push`,
  `pop`, `substr`, `num`. Lists have reference semantics (as in Python), cannot contain themselves
  (`push(a, a)` is an error), and are limited (10 M elements, 256 MB per text, `print` abbreviates very
  long lists). Text is indexed by UTF-8 character, not by byte.
- **Parser robustness**: nesting depth (200) and chain length (20 000 terms) are limited, so a hostile or
  broken file gives an error message instead of a stack overflow; huge number literals (`1` followed by
  400 zeros) gave the message `stod` and now say what is wrong; "Ausdruck erwartet" now names line and
  column. `for i in a..b` rejects non-finite or huge bounds (casting them to `long long` was undefined).
- **Tests for the language** (`tests/fuzz.cpp`, run under all sanitizers in CI): mutation fuzzing of
  `Lexer`/`Parser` with ~11 000 mutated example scripts and 3 000 random byte strings (only clean
  `runtime_error`s allowed), 600 randomly generated programs (run twice, results must be identical),
  a model test that runs 300 list programs and 400 text programs in Skull and in a C++ model and compares
  output and error messages with line numbers, and limit tests (a 300 000-level-deep list is freed without
  recursion on a 1 MB stack; cycles, exponential comparisons and huge lists are stopped). The tests were
  mutation-checked: five deliberately injected bugs (index, destructor, cycle check, UTF-8, equality
  shortcut) were each caught.

### ➕ Added

- **Checkpoints / resume** (`checkpoint`, `stop_after`, `resume`): the complete training state
  (weights, Adam moments, RNG state, learning-rate position, best-so-far model and the early-stopping
  counter) is saved atomically (`.tmp` then rename) and restored. **A paused and resumed run produces
  byte-identical weights to an uninterrupted one** (tested for bigram and transformer, with batches,
  `steps` windows and different thread counts). A mutation test confirmed each restored piece of state
  is actually checked. Mismatching checkpoints (other model kind or size, other data / `val` / `bpe`,
  truncated or corrupt file) are rejected with a clear message.
- **`seed`** for `train` (initial weights and window order; default 42, so existing behaviour is
  unchanged) and for `generate` (reproducible sampling).
- **`top_k` / `top_p`** sampling for `generate` (deterministic tie-breaking by token id). While adding
  them I fixed `sample()`: when rounding made the cumulative sum fall short it could return the last
  token of the vocabulary even with probability 0; it now returns the last token with p > 0.
- **KV cache for generation** (`Transformer::prefill()` / `step()`): a new token only computes its own
  row. The logits are bit-identical to a full forward pass (`tests/gradcheck.cpp`). Default mode is
  exact (output unchanged; the cache helps until the context is full, after that every token needs a
  full recompute because positions are absolute). `generate { shift = true }` keeps only the younger
  half when the context is full: 300 tokens at context 64 take 0.04 s instead of 0.83 s (20x), at the
  price of sometimes seeing only `context/2` to `context` tokens.
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
- Regression tests (`ctest`, ~100 cases) for all of the above, run by GitHub Actions on Linux
  (gcc, clang, AddressSanitizer/UBSan), macOS and Windows. Run locally with `./build.sh --test`.
- Builtin `live_tensors()` (number of live tensors; used by the leak test).
- `-DSKULL_SANITIZE=ON` build option.

### 🔄 Changed

- **Reproducibility claim corrected.** The docs said that a seed gives the same weights "on every system".
  Measured: the *initial* weights and all random draws are identical everywhere, and training is
  bit-identical for the same build (any thread count, with pause/resume), but a gcc AVX2 build, a
  non-AVX2 build and a clang build write weight files that differ in the last bits (the printed loss agrees
  to 6 digits). The README now says so.

- **Random numbers are now identical on every platform** (`src/rng.h`). `std::mt19937` is fixed by
  the C++ standard, but `std::uniform_real_distribution` / `std::uniform_int_distribution` are not:
  libstdc++, libc++ (macOS) and MSVC return different numbers for the same seed, so the same
  `train` produced different initial weights and training windows on different systems. Skull now
  converts the raw generator output itself (bit-identical to NumPy's `rand()` for the same seed;
  `tests/rngcheck.cpp`). Trained weights differ from earlier builds for the same data.
- **Transformer training uses a learning-rate schedule** (short linear warm-up, then cosine decay
  to 10 % of `rate`). With a constant rate the loss kept jittering around its floor and about one
  run in twenty ended with a visibly worse model; `rate` is now the peak learning rate.
- **BPE tokenizer rewritten for speed and correctness** (`src/tokenizer.h`). Tokens are integer IDs
  instead of strings, the text lives in a doubly linked list, pair counts are updated
  incrementally around each merge (max-heap with lazy deletion) instead of being recounted from
  scratch, and `encode` uses a min-heap over merge ranks (O(n log n)) instead of applying every
  merge to the whole text, and training hands back the encoded training text instead of encoding
  it a second time. On ~250 KB of text the old code needed 13.6 s (training, capped at 500 merges)
  plus 1.5 s (encoding); the new code takes 0.05 s plus 0.05 s for a 1000-token vocabulary, and
  1 MB takes about 0.3 s plus 0.3 s. The result is exactly that of the plain definition
  (most frequent pair, ties broken by the smaller token IDs, non-overlapping left-to-right
  replacement); `tests/bpecheck.cpp` checks this against a deliberately naive reference on
  thousands of random small corpora (including `aaaa` / `abab` style repeats). The `TOKB` file
  format is unchanged, and files written by older versions load and encode exactly as before.
  One deliberate difference for new training runs: when two pairs are equally frequent, the one
  with the smaller token IDs wins (the old code compared the token strings), so a freshly trained
  vocabulary can differ from one trained by an older build on the same data.
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
