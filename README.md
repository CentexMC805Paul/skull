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

**Status:** v0.7.0 — working, builds on Windows with Visual Studio

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
