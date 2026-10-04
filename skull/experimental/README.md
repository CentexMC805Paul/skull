# experimental/ — Entwürfe, NICHT Teil des Builds

Hier liegt der Code aus dem Commit „Skull v1.0.0: Cross-Platform + CUDA + Python Bindings + Flexible Models“
(ca. 6.300 Zeilen). Er wurde **nie zusammen kompiliert**, ist **nicht an die CLI angebunden** und
funktioniert so, wie er ist, nicht. Er liegt hier, damit die Ideen nicht verloren gehen und als
Ausgangspunkt für spätere Arbeit — nicht, um benutzt zu werden.

Die laufende CLI (`../src/`) braucht nichts davon. `CMakeLists.txt` baut nur `src/`.

> Alle Befunde unten stammen aus einem Code-Review und Kompilierversuchen
> (g++ 13, `-std=c++17 -mavx2 -mfma`); die CUDA-Aussagen sind reine Code-Lektüre (kein `nvcc` verfügbar).

## Inhalt und Zustand

| Datei | Soll | Zustand |
|---|---|---|
| `layers.h` | Linear, Attention, LayerNorm, Dropout, LSTM, Conv1D, Transformer | Kompiliert nicht (~31 Fehler zusammen mit `model.h`). Nur Linear und Dropout haben ein brauchbares Backward; Attention, LSTM und Conv1D haben keins oder einen Platzhalter, LayerNorm eins mit falscher Mathematik. |
| `model.h` | Feedforward-/Transformer-/LSTM-Modelle, `train_model`, `generate_text` | Kompiliert nicht. `train_model`/`evaluate_model` sind Attrappen (lesen die Datei nie). `save`/`load` schreiben nur die halben Daten (`float` statt `double`). Der „Cross-Entropy“-Loss nimmt den Logarithmus roher Logits. |
| `optimizer.h` | SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta, Lion + LR-Scheduler | Adam/AdamW sind korrekt. SGD-Momentum, RMSprop, Adadelta, Lion und StepLR/Cosine-Scheduler haben Formelfehler. Die Optimizer lesen `Tensor::grad`, die Layer schreiben aber in `grad_weight` — es würde nie etwas gelernt. |
| `parallel.h` | Thread-Pool, parallele Tensor-Operationen | Kompiliert nicht. `ThreadPool::wait()` kann zu früh zurückkehren, `set_global_thread_pool_size()` wirkt nicht auf den globalen Pool, Exceptions aus Workern gehen verloren. |
| `config.h` | Plattform-/Feature-Makros, Enums | `g_precision_mode`/`g_gpu_backend` sind nur `extern` deklariert (Linkerfehler). Die meisten Optionen werden nirgends gelesen. |
| `gpu_cuda.cu` | CUDA-Kernel | Das `CUDA_CHECK`-Makro hat keine `\`-Fortsetzungszeilen. Es gibt keine Kernel-Starter und kein `cudaMalloc`/`cudaMemcpy`. Softmax-, LayerNorm- und Attention-Kernel haben Race Conditions und Shared-Memory-Überläufe. |
| `skull_python.cpp` | Python-Modul (rohe CPython-C-API) | Kompiliert nicht (u. a. falsches `PyObject_New`, falsch ausgerichtete `PyTypeObject`-Initialisierer). Das Modul heißt `PyInit_skull`, importiert wird aber `._skull`. `train()`/`generate()` sind Stubs. |
| `python/` | Python-Paket und CLI | Syntaktisch in Ordnung, ruft aber Methoden auf, die das C++-Modul nicht hat (`set_optimizer`, `set_scheduler`, `evaluate`). `--gpu`, `--batch-size`, `--bpe` ohne Wirkung. |
| `setup.py` | Build des Python-Moduls | Ruft CMake mit dem falschen Verzeichnis auf, kopiert eine Bibliothek, die CMake nie erzeugt, und `find_packages()` findet das Paket nicht. |

Zusätzlich: `layers.h` definiert `tensor_add` ein zweites Mal (steht schon in `../src/tensor.h`).
Außerdem sucht kein Include-Pfad `tensor.h`, wenn man aus diesem Ordner baut — es braucht `-I../src`.

## Wenn du hier weiterarbeiten willst

Empfohlene Reihenfolge, damit jeder Schritt prüfbar bleibt:

1. `layers.h` einzeln zum Kompilieren bringen (ohne `model.h`, `skull_python.cpp`).
2. Pro Layer einen Gradiententest gegen endliche Differenzen schreiben, bevor irgendetwas trainiert wird.
3. Erst dann ein Modell mit Kontext > 1 Token (Attention) bauen und über `define model { ... }` in
   `../src/interpreter.h` anbinden. Die Felder `layers` und `heads` werden dort heute nur mit einer
   Warnung ignoriert.
4. Python-Bindings und CUDA zuletzt.
