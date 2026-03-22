# Skull v0.6.0 — Was funktioniert

## Die Sprache
```skull
define x = 42
define func name(args) { return args + 1 }
if x > 10 { print("gross") }
for i in 1..100 { print(i) }
while x > 0 { x = x - 1 }
```

## Tensoren & KI-Mathematik (AVX2)
```skull
define W = rand_tensor(512, 512)
define h = relu(W * x)
define loss = mse_loss(output, target)
backward(loss)
update(W, 0.001)
```

## Training
```skull
define model MeinLLM { dim = 64  vocab = 256 }
train MeinLLM { data = "text.txt"  epochs = 100  rate = 0.005 }
```

## Textgenerierung
```skull
generate MeinLLM {
    weights     = "text.txt.weights"
    prompt      = "Hallo"
    tokens      = 100
    temperature = 0.8
}
```

## Aktueller Stand der Dateien
```
skull/src/
  lexer.h        — Tokenizer          (fertig)
  ast.h          — Syntaxbaum         (fertig)
  parser.h       — Parser             (fertig)
  interpreter.h  — Ausfuehrung        (fertig)
  tensor.h       — SIMD Tensor-Engine (fertig, AVX2)
  trainer.h      — Echtes Training    (fertig)
  generator.h    — Textgenerierung    (fertig)
  main.cpp       — Einstiegspunkt     (fertig)
```

## Naechste Schritte (Prioritaet)
1. Groessere Trainingsmengen unterstuetzen
2. Mehrere Transformer-Layer
3. Attention-Mechanismus einbauen
4. GitHub veroeffentlichen
