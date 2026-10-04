# Skull Project

Two repositories, one goal: make LLM training accessible to everyone.

---

## [skull/](skull/)

A programming language built from the ground up for training language models.

```skull
define model MyLLM { dim = 128  vocab = 256 }

train MyLLM {
    data   = "my_text.txt"
    epochs = 100
    rate   = 0.001
}

generate MyLLM {
    weights     = "my_text.txt.weights"
    prompt      = "Hello"
    tokens      = 100
}
```

**Status:** v1.0.0 — the language and a small trainer work; builds with CMake on Linux, macOS and Windows.
The model is deliberately tiny (it sees one token at a time), so it demonstrates training and
generation rather than producing meaningful text. Attention, GPU training and Python bindings are
not implemented yet — drafts live in [`skull/experimental/`](skull/experimental/README.md).

---

## [skull-research/](skull-research/)

Exploring new mathematics to make LLM training fundamentally faster.

**Status:** Early — collecting open questions, starting experiments

---

## Why?

Training a language model today requires money, a powerful GPU, and deep technical knowledge. Most people who want to build their own AI simply cannot.

Skull is the attempt to change that.

---

## License

MIT — free for everything, forever.
