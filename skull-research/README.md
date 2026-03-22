# Skull Research

> Exploring new mathematics to make LLM training fundamentally faster.

This is the research branch of the [Skull project](../skull).

While Skull focuses on making known optimizations accessible, Skull Research asks harder questions:

**What if the math itself could be better?**

---

## Open Questions

### Attention
- Can we approximate `softmax(QK^T / sqrt(d)) * V` with something cheaper but equally expressive?
- Is the O(seq_len^2) complexity of attention actually necessary?
- What information does attention need to preserve, and what can be discarded?

### Backpropagation
- Is there a way to compute gradients without storing the entire forward pass?
- Can we update weights with lower precision and still converge reliably?

### Loss Functions
- Cross-entropy is standard. Is it actually optimal for language modeling?
- Are there loss functions that lead to faster or more stable convergence?

### Numerical Precision
- Where exactly does FP16 hurt training, and can we design around those cases?
- Can mixed-precision strategies be made automatic?

---

## How This Works

1. An idea goes into `notes/` — raw thinking, no pressure
2. If it looks promising, it becomes an `experiments/` folder with code
3. If it works, it gets implemented in [Skull](../skull)
4. If it fails, we document why in `notes/` — failures are valuable

---

## Structure

```
skull-research/
  experiments/   — Code experiments, one folder per idea
  papers/        — Notes on relevant research papers
  notes/         — Raw ideas, dead ends, open questions
```

---

## Honesty Policy

Most experiments here will fail. We document the failures too, because knowing what does not work is as valuable as knowing what does.

---

## How to Contribute

If you have an idea for a faster or better training algorithm:

1. Open an issue describing the idea
2. We discuss whether it's worth trying
3. You or we implement it in `experiments/`
4. We measure whether it actually helps

No PhD required. Good ideas come from everywhere.

---

## Related

- [Skull](../skull) — the language where working ideas end up
- [FlashAttention paper](https://arxiv.org/abs/2205.14135) — inspiration for what better math can achieve
- [Chinchilla paper](https://arxiv.org/abs/2203.15556) — optimal compute allocation

---

## License

MIT
