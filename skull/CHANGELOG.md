# Changelog

## v0.7.0
- Multi-format tokenizer: `.txt`, `.md`, `.json`, `.jsonl`, `.csv`
- BPE (Byte-Pair Encoding) tokenization — enable with `bpe = true`
- OpenCL GPU backend — NVIDIA, AMD, Intel (enable with `gpu = true`)
- `gpu_info()` function to check available GPU hardware
- `prefer_amd = true` option to prioritize AMD GPUs
- Improved trainer with automatic format detection

## v0.6.0
- `generate` statement for text generation
- Weights saved to `.weights` file after training
- Temperature parameter for controlling creativity
- `generator.h` — full inference engine

## v0.5.0
- Real training loop with cross-entropy loss
- Backpropagation (manual, exact gradients)
- `trainer.h` — standalone training engine
- Weights saved in binary format

## v0.4.0
- Tensor type fully integrated into the language
- AVX2 SIMD acceleration (4x doubles per clock)
- `rand_tensor`, `zeros`, `ones`
- `relu`, `sigmoid`, `tanh_act`
- `mse_loss`, `backward`, `update`, `zero_grad`
- Matrix multiplication with transposed B for cache efficiency

## v0.3.0
- Full interpreter working
- Variables, functions, loops, if/else
- Re-assignment without `define`
- Scope chain (variables visible in parent scopes)

## v0.2.0
- Parser complete
- Full AST: expressions, statements, blocks
- `define model`, `train` syntax

## v0.1.0
- Lexer complete
- All Skull keywords, operators, comments
- Line and column numbers for error messages
