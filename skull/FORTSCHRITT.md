# Skull v1.0.0 — Was funktioniert

## Die Sprache
```skull
define x = 42
define func name(args) { return args + 1 }
if x > 10 { print("gross") } else if x > 5 { print("mittel") } else { print("klein") }
if x % 2 == 0 and not (x > 100) { print("gerade") }
for i in 1..100 { print(i) }
while x > 0 { x = x - 1 }
```
Rekursion bis 1000 Ebenen, typsichere Vergleiche, `and`/`or`/`not` mit Kurzschluss, `%`,
Fehlermeldungen mit Zeilennummer. Fehlt: Listen.

## Tensoren & KI-Mathematik (AVX2, Autograd)
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
train MeinLLM { data = "text.txt"  epochs = 100  rate = 0.005  batch = 8 }   // Bigram, SGD

define model Mini { dim = 32  context = 32  heads = 2  layers = 2 }
train Mini { data = "text.txt"  out = "mini.weights"  epochs = 200  rate = 0.003  batch = 4 }  // Transformer, Adam
```
- `context = 1` (Standard): Embedding → ReLU-Hidden → Softmax, ein Token Kontext (Bigram), SGD.
- `context > 1`: Transformer mit kausaler Multi-Head-Attention, LayerNorm, GELU-MLP, mehreren
  Schichten, Adam mit Gradienten-Clipping. Backward von Hand, gegen endliche Differenzen geprüft.
- Alle Tokens pro Epoche (oder Fenster mit `steps`), echtes Mini-Batch, BPE optional.
- Konstanter Speicherbedarf, kein Autograd-Graph im Trainingsschritt.
- Formate: `.txt`, `.md`, `.json`, `.jsonl`, `.csv`.
- Rechnet auf der CPU. `gpu = true` erkennt nur das Gerät.

## Textgenerierung
```skull
generate MeinLLM {
    weights     = "text.txt.weights"
    prompt      = "Hallo"
    tokens      = 100
    temperature = 0.8      // 0 = greedy
}
```

## Tests
`./build.sh --test` (oder `ctest` im Build-Verzeichnis): 46 Tests, darunter Regressionstests für
Speicherleck, Autograd, Trainingsdaten-Limit, Generator-Randfälle und der Gradiententest des
Transformers. CI läuft auf Linux (gcc, clang,
Sanitizer), macOS und Windows.

## Aktueller Stand der Dateien
```
skull/src/
  version.h      — Versionsnummer (einzige Quelle)
  lexer.h        — Tokenizer der Sprache   (fertig)
  ast.h          — Syntaxbaum              (fertig)
  parser.h       — Parser                  (fertig)
  interpreter.h  — Ausführung              (fertig)
  stack.h        — Interpreter-Thread mit großem Stack
  tensor.h       — Tensor-Engine + Autograd (AVX2), für die Sprache
  transformer.h  — Transformer (Attention) + Adam, Backward von Hand
  tokenizer.h    — Dateiformate + BPE
  weights.h      — Gewichte speichern/laden (mit Prüfung)
  trainer.h      — Training
  generator.h    — Textgenerierung
  gpu.h          — OpenCL-Geräteerkennung (noch ohne Rechenarbeit)
  main.cpp       — Einstiegspunkt
skull/experimental/ — Entwürfe (Transformer, Optimizer, CUDA, Python), nicht im Build
skull/tests/        — ctest-Fälle, gradcheck.cpp (Gradiententest)
```

## Nächste Schritte (Priorität)
1. Geschwindigkeit des Transformers: mehrere Threads über die Sequenzen eines Batches, `float`
   statt `double`, gepufferte Matrixmultiplikation.
2. Generierung: Top-k/Top-p-Sampling, KV-Cache (heute wird bei jedem Token der ganze Kontext neu gerechnet).
3. Sprache: Listen, Strings indizieren.
4. GPU-Rechnen: die OpenCL-Kernel in `gpu.h` an den Transformer anbinden und gegen die CPU prüfen.
5. Validierungsdaten und Checkpoints während des Trainings.
