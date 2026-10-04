# Skull v1.0.0 - Einfache KI-Trainings-Sprache

**Skull macht KI-Training für jeden zugänglich - ohne komplizierte Setups!**

---

## ⚠️ Was Skull heute ist (und was nicht)

Skull ist eine kleine Programmiersprache mit eingebautem Trainer. Das **Modell** ist bewusst winzig:
Embedding → Hidden-Schicht (ReLU) → Softmax. Es sieht pro Schritt **nur das vorige Token** und
sagt das nächste voraus (ein neuronales Bigram-Modell). Das reicht, um Training, Verlust und
Textgenerierung zu verstehen — aber **nicht**, um sinnvollen Text zu schreiben. Rechnen tut Skull
auf der **CPU** (AVX2, wenn vorhanden).

Noch **nicht** vorhanden: Attention/Transformer, mehrere Schichten, GPU-Training, Python-Bindings.
Entwürfe dafür liegen in [`experimental/`](experimental/README.md) und sind nicht Teil des Builds.

---

## 🚀 Schnellstart (3 Schritte)

### 1️⃣ Installieren und bauen

#### **Linux/macOS:**
```bash
# Abhängigkeiten installieren
sudo apt install build-essential cmake git   # Ubuntu/Debian
brew install cmake                            # macOS (Compiler: xcode-select --install)

# Repository klonen
git clone https://github.com/CentexMC805Paul/skull.git
cd skull/skull

# Bauen (und gleich alle Tests laufen lassen)
./build.sh --test
```

#### **Windows:**
1. **Visual Studio 2019 oder neuer** mit C++-Werkzeugen und **CMake** installieren:
   [Visual Studio](https://visualstudio.microsoft.com/downloads/), [CMake](https://cmake.org/download/)
2. **Repository klonen:**
   ```cmd
   git clone https://github.com/CentexMC805Paul/skull.git
   cd skull\skull
   ```
3. **Bauen:**
   ```cmd
   build.bat --test
   ```

Es wird **keine GPU-Bibliothek** gebraucht. Das fertige Programm liegt in `build/skull` (Windows: `build\skull.exe`).

---

### 2️⃣ Testen

```bash
# Linux/macOS
./build/skull examples/hello.skull

# Windows
build\skull.exe examples\hello.skull
```

---

### 3️⃣ KI trainieren

```bash
# Linux/macOS
./build/skull examples/train.skull

# Windows
build\skull.exe examples\train.skull
```

Beim Training wird neben die Datendatei eine `.weights`-Datei geschrieben
(z. B. `examples/training_data.txt.weights`).

---

## 📚 Einfache Beispiele

| Beispiel | Beschreibung | Befehl |
|----------|-------------|--------|
| `hello.skull` | Grundlagen (Variablen, Funktionen, Schleifen) | `./build/skull examples/hello.skull` |
| `tensor_test.skull` | Tensoren, Backpropagation | `./build/skull examples/tensor_test.skull` |
| `train.skull` | Einfaches KI-Training | `./build/skull examples/train.skull` |
| `generate.skull` | Trainieren und Text generieren | `./build/skull examples/generate.skull` |
| `format_test.skull` | Trainieren mit Markdown und JSONL | `./build/skull examples/format_test.skull` |

---

## 📖 Sprach-Referenz (Einfach erklärt)

### 1. Variablen
```skull
define x = 42
define name = "Skull"
print("x =", x)
```

### 2. Funktionen
```skull
define func add(a, b) {
    return a + b
}
print(add(3, 4))  // Ausgabe: 7
```
Rekursion ist möglich (bis 1000 verschachtelte Aufrufe). `return` gibt es nur in Funktionen.

### 3. Schleifen & Bedingungen
```skull
// Schleife (beide Grenzen eingeschlossen)
for i in 1..5 {
    print(i)
}

// Bedingung
if x > 10 {
    print("x ist groß")
} else {
    print("x ist klein")
}
```
Vergleiche: `==  !=  <  >  <=  >=`. `==` vergleicht mit Typ (`1 == "1"` ist `false`).
Es gibt (noch) kein `else if`, kein `and`/`or`/`!` und kein `%`.

### 4. KI-Modell trainieren
```skull
// Modell definieren
define model MeinModell {
    dim   = 64    // Größe des Modells
    vocab = 256   // Vokabular-Größe (256 = ein Token pro Byte)
}

// Training starten
train MeinModell {
    data   = "mein_text.txt"  // Deine Trainingsdaten (.txt, .md, .json, .jsonl, .csv)
    epochs = 50               // Anzahl Durchläufe
    rate   = 0.01             // Lernrate
}
```

### 5. Text generieren
```skull
generate MeinModell {
    weights     = "mein_text.txt.weights"  // Trainierte Gewichte
    prompt      = "Hallo"                  // Starttext
    tokens      = 50                       // Anzahl Tokens
    temperature = 0.8                      // Kreativität; 0 = immer das wahrscheinlichste Token
}
```
Maßgeblich sind Größe und Tokenizer aus der Gewichte-Datei; passt `dim` im `define model`
nicht dazu, warnt Skull.

---

## 🎯 Weitere Optionen

### `train { ... }`

| Feld | Standard | Bedeutung |
|------|----------|-----------|
| `data` | – | Pfad zur Trainingsdatei (String, Pflicht) |
| `epochs` | 10 | Durchläufe über die Daten |
| `rate` | 0.001 | Lernrate |
| `batch` | 1 | Tokens pro Gewichts-Update (Gradient wird gemittelt) |
| `steps` | 0 | 0 = jede Epoche geht durch **alle** Tokens; sonst ein zufälliges Fenster dieser Länge pro Epoche (für große Dateien) |
| `dim`, `vocab` | 64, 256 | Auch im `define model` setzbar |
| `bpe`, `bpe_vocab` | false, 1000 | Byte-Pair-Encoding statt Bytes als Token. Der Tokenizer wird in der Gewichte-Datei mitgespeichert, `generate` benutzt ihn automatisch |
| `gpu`, `prefer_amd` | false | Zeigt das OpenCL-Gerät an. **Das Training läuft trotzdem auf der CPU** (Hinweis wird ausgegeben) |

Unbekannte Felder (z. B. Tippfehler) und noch nicht unterstützte (`layers`, `heads`) werden mit
einer Warnung gemeldet statt still ignoriert.

### Fehler
Fehler (fehlende Datei, kaputte Gewichte, falscher Typ …) beenden Skull mit einer Meldung
`[FEHLER] Zeile N: …` und Exit-Code 1.

---

## 🔧 Build-Optionen

| Befehl | Beschreibung |
|--------|-------------|
| `./build.sh` | Standard: CPU mit AVX2 (falls die CPU es kann) |
| `./build.sh --test` | Bauen und alle Tests ausführen |
| `./build.sh --gpu` | Zusätzlich OpenCL-Geräteerkennung (`gpu_info()`); braucht OpenCL-Header |
| `./build.sh --debug` | Debug-Modus |
| `./build.sh --clean` | Build-Verzeichnis löschen |

Windows: dieselben Optionen mit `build.bat`.

Direkt mit CMake: `cmake -S . -B build && cmake --build build && cd build && ctest`.
Weitere CMake-Schalter: `-DSKULL_ENABLE_AVX2=OFF` (ältere CPUs), `-DSKULL_SANITIZE=ON`
(AddressSanitizer/UBSan für Tests), `-DSKULL_USE_OPENCL=ON`.

---

## 📦 Abhängigkeiten

| System | Pflicht | Optional |
|--------|---------|----------|
| Linux (Ubuntu/Debian) | `build-essential cmake` | `opencl-headers ocl-icd-opencl-dev` (nur für `--gpu`) |
| macOS | Xcode-Kommandozeilentools, `cmake` | – |
| Windows | Visual Studio 2019+ mit C++-Werkzeugen, CMake | OpenCL-SDK (nur für `--gpu`) |

---

## 🤝 Mitmachen

- **Bugs melden:** [GitHub Issues](https://github.com/CentexMC805Paul/skull/issues)
- **Fragen stellen:** [GitHub Discussions](https://github.com/CentexMC805Paul/skull/discussions)
- **Code verbessern:** Pull Requests sind willkommen! (siehe [CONTRIBUTING.md](CONTRIBUTING.md))

---

## 📜 Lizenz

MIT License - frei für alles, für immer.

---

**Skull - KI-Training für jeden!** 🚀
