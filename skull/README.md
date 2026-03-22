# Skull

**A programming language built from the ground up for LLM/SLM training.**

Skull makes training your own language model accessible to everyone — no expensive cloud, no complex setup, no deep ML knowledge required.

```skull
define model MyLLM {
    dim   = 128
    vocab = 256
}

train MyLLM {
    data   = "my_text.txt"
    epochs = 100
    rate   = 0.001
}

generate MyLLM {
    weights     = "my_text.txt.weights"
    prompt      = "Hello"
    tokens      = 100
    temperature = 0.8
}
```

That's it. Skull handles the rest.

---

## Why Skull?

Training a language model today requires expensive GPUs, complex Python/PyTorch setup, and deep knowledge of optimization tricks. Most people who want to build their own AI simply can't.

Skull changes that.

## Features

- **Simple syntax** — if you can write Python, you can write Skull
- **AVX2 SIMD acceleration** — 4x faster matrix operations on any modern CPU
- **GPU support** — OpenCL backend works on NVIDIA, AMD, and Intel GPUs
- **Multi-format training data** — `.txt`, `.md`, `.json`, `.jsonl`, `.csv`
- **BPE tokenization** — smarter tokenization like GPT uses
- **Automatic differentiation** — backprop is built in, you never write it manually
- **Saves weights** — trained models are stored and reloadable

## Quick Start

### Build

**Requirements:** Visual Studio 2017/2019/2022 with C++ tools

```
cd skull
build.bat
```

For GPU support (requires OpenCL SDK):
```
build.bat --gpu
```

### Run

```
skull.exe examples\train_demo.skull
skull.exe examples\generate_demo.skull
skull.exe examples\format_test.skull
```

## Language Reference

### Variables
```skull
define x = 42
define name = "Skull"
x = x + 1
```

### Functions
```skull
define func add(a, b) {
    return a + b
}
print(add(3, 4))
```

### Control Flow
```skull
if x > 10 {
    print("big")
} else {
    print("small")
}

for i in 1..100 {
    print(i)
}
```

### Tensors
```skull
define W = rand_tensor(128, 128)
define h = relu(W * x)
define loss = mse_loss(output, target)
backward(loss)
update(W, 0.001)
```

### Training Options
```skull
train MyModel {
    data        = "data.txt"   // .txt .md .json .jsonl .csv
    epochs      = 100
    rate        = 0.001
    bpe         = true         // BPE tokenization
    bpe_vocab   = 1000         // BPE vocabulary size
    gpu         = true         // GPU via OpenCL
    prefer_amd  = true         // prefer AMD over NVIDIA
}
```

## Project Structure

```
skull/
  src/
    lexer.h        — Tokenizer
    ast.h          — Syntax tree
    parser.h       — Parser
    interpreter.h  — Execution engine
    tensor.h       — SIMD tensor operations (AVX2)
    trainer.h      — Training loop
    generator.h    — Text generation
    tokenizer.h    — Multi-format tokenizer + BPE
    gpu.h          — OpenCL GPU backend
  examples/        — Example .skull programs
  docs/            — Documentation
```

## Sister Project

[skull-research](../skull-research) — exploring new mathematics to make LLM training fundamentally faster.

## License

MIT — free to use for anything, forever.

## Contributing

This project is just getting started. Issues, ideas, and pull requests are welcome.
