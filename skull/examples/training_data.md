# Skull — Eine neue Programmiersprache fuer KI

Skull ist eine Programmiersprache die speziell fuer das Training
von Sprachmodellen gebaut wurde.

## Ziele

Das Ziel von Skull ist es LLM Training zuganglich zu machen.
Jeder soll in der Lage sein sein eigenes Modell zu trainieren.

## Features

- Einfache Syntax wie Python
- Automatische Hardware-Optimierung
- AVX2 SIMD Beschleunigung
- GPU-Support fuer NVIDIA und AMD
- BPE Tokenisierung

## Beispiel

Ein einfaches Skull Programm sieht so aus:

```
define model MeinLLM {
    dim    = 512
    layers = 6
}

train MeinLLM {
    data   = "mein_text.txt"
    epochs = 100
}
```

Skull kuemmert sich um den Rest.
