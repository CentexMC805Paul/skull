# Skull v1.0.0 - Einfache KI-Trainings-Sprache

**Skull macht KI-Training für jeden zugänglich - ohne komplizierte Setups!**

---

## ⚠️ Was Skull heute ist (und was nicht)

Skull ist eine kleine Programmiersprache mit eingebautem Trainer für **winzige** Sprachmodelle.
Es gibt zwei Modelle, gewählt über das Feld `context`:

| Modell | `context` | Was es sieht | Training |
|--------|-----------|--------------|----------|
| **Bigram** (Standard) | 1 | nur das vorige Token | SGD |
| **Transformer** mit Attention | > 1 | bis zu `context` vorige Token | Adam |

Beide trainieren und generieren auf der **CPU** (AVX2, wenn vorhanden). Der Transformer
(kausale Multi-Head-Attention, LayerNorm, GELU-MLP, mehrere Schichten) lernt auf kleinen Texten
Wörter und Satzteile; das Bigram-Modell erzeugt dagegen nur Zufallsbuchstaben. Beispiel
(`tests/cases/tf_learns_context.skull`): Auf „das ist ein test. “ wiederholt trainiert, setzt der
Transformer den Satz fehlerfrei fort, das Bigram-Modell bleibt bei „t. t. t.“ hängen.

Das sind **Spielzeugmodelle** (einige zehntausend Parameter, trainiert in Sekunden), gut zum
Verstehen von Training, Verlust und Generierung — nicht für echten Text in guter Qualität.

Noch **nicht** vorhanden: GPU-Training, Python-Bindings, Gewichte-Austausch mit anderen Frameworks.
Entwürfe für GPU/Python liegen in [`experimental/`](experimental/README.md) und sind nicht Teil des Builds.

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
| `train.skull` | Einfaches KI-Training (Bigram) | `./build/skull examples/train.skull` |
| `transformer.skull` | Modell mit Attention (Transformer) trainieren und Text erzeugen | `./build/skull examples/transformer.skull` |
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

// Bedingung, auch mit else if
if x > 10 {
    print("x ist groß")
} else if x > 5 {
    print("x ist mittel")
} else {
    print("x ist klein")
}

// Logik und Rest
if x % 2 == 0 and not (x > 100) {
    print("gerade und höchstens 100")
}
```
- Vergleiche: `==  !=  <  >  <=  >=`. `==` vergleicht mit Typ (`1 == "1"` ist `false`).
- Logik: `and`, `or`, `not` (auch `!`). Rangfolge: `or` < `and` < `not` < Vergleiche. `and`/`or`
  werten die rechte Seite nur aus, wenn nötig, und liefern `true`/`false`.
- Rechnen: `+  -  *  /  %`. `%` wie in Python (`-7 % 3` ist `2`).
- Es gibt (noch) keine Listen.

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

### 5. Modell mit Attention (Transformer)
```skull
define model Mini {
    dim     = 32    // Breite
    context = 32    // so viele Zeichen sieht das Modell gleichzeitig (> 1 = Transformer)
    heads   = 2     // Attention-Köpfe (dim muss durch heads teilbar sein)
    layers  = 2     // Anzahl Schichten
}
train Mini {
    data = "text.txt"
    out  = "mini.weights"   // optional: Zielpfad der Gewichte
    epochs = 200
    rate   = 0.003          // beim Transformer die Spitzen-Lernrate von Adam
    batch  = 4              // Sequenzen pro Update
}
```
Mehr dazu in `examples/transformer.skull`. Ist die Datei kürzer als `context`, wird `context`
verkleinert (mit Hinweis).

### 6. Text generieren
```skull
generate MeinModell {
    weights     = "mein_text.txt.weights"  // Trainierte Gewichte
    prompt      = "Hallo"                  // Starttext
    tokens      = 50                       // Anzahl Tokens
    temperature = 0.8                      // Kreativität; 0 = immer das wahrscheinlichste Token
}
```
Maßgeblich sind Modellart, Größe und Tokenizer aus der Gewichte-Datei; passt `dim`, `context`,
`heads` oder `layers` im `define model` nicht dazu, warnt Skull. Beim Transformer sieht das Modell
beim Generieren immer die letzten `context` Token.

---

## 🎯 Weitere Optionen

### `train { ... }`

| Feld | Standard | Bedeutung |
|------|----------|-----------|
| `data` | – | Pfad zur Trainingsdatei (String, Pflicht) |
| `out` | `data` + `.weights` | Zielpfad der Gewichte-Datei |
| `epochs` | 10 | Durchläufe über die Daten |
| `rate` | 0.001 | Lernrate (Bigram: SGD; Transformer: Adam, Spitzenwert mit Warmup und Cosine-Abfall auf 10 %) |
| `batch` | 1 | Bigram: Tokens pro Update; Transformer: Sequenzen pro Update (Gradient wird gemittelt) |
| `steps` | 0 | 0 = jede Epoche geht durch **alle** Tokens; sonst nur so viele Token-Schritte (zufällige Fenster) pro Epoche, für große Dateien |
| `dim`, `vocab` | 64, 256 | Auch im `define model` setzbar |
| `val` | 0.1 | Anteil der Daten (vom **Dateiende**), der nicht trainiert, sondern zum Bewerten benutzt wird; 0 = aus. Höchstens 50 000 Token; bei zu wenig Daten (< 100 Token) wird übersprungen, mit Hinweis |
| `patience` | 0 | Early Stopping: Abbruch, wenn sich der Validierungs-Loss N Epochen nicht verbessert (0 = aus) |
| `context` | 1 | 1 = Bigram, > 1 = Transformer mit diesem Kontext (max. 8192) |
| `heads`, `layers` | 2, 1 | Nur Transformer. `dim` muss durch `heads` teilbar sein |
| `bpe`, `bpe_vocab` | false, 1000 | Byte-Pair-Encoding statt Bytes als Token. Der Tokenizer wird in der Gewichte-Datei mitgespeichert, `generate` benutzt ihn automatisch |
| `gpu`, `prefer_amd` | false | Zeigt das OpenCL-Gerät an. **Das Training läuft trotzdem auf der CPU** (Hinweis wird ausgegeben) |

### Validierung: lernt das Modell wirklich?
Der Trainings-Loss sinkt fast immer, auch wenn das Modell nur auswendig lernt. Deshalb hält Skull
standardmäßig die letzten 10 % der Daten zurück und zeigt pro Epoche den **Val-Loss** und die
**Perplexität** (`exp(Loss)`, grob: unter wie vielen gleich wahrscheinlichen Token das Modell
schwankt; kleiner ist besser):

```
Epoche 10/80  |  Loss: 2.13  |  Val: 2.34 (Perplexitaet 10.4) *  |  Zeit: 15s
```
`*` markiert eine neue Bestleistung. Steigt der Val-Loss, während der Trainings-Loss weiter sinkt,
ist das **Überanpassung**; Skull speichert dann die Gewichte mit dem **besten Val-Loss**
(nicht die der letzten Epoche) und sagt, aus welcher Epoche sie stammen. Mit `patience = 5` bricht
das Training nach 5 Epochen ohne Verbesserung ab.

Im Skript lassen sich die Ergebnisse abfragen, z. B. für einen Vergleich mehrerer Läufe:
`last_loss()` (Trainings-Loss der letzten Epoche), `last_val_loss()` (Val-Loss des gespeicherten
Modells) und `last_best_epoch()`.

Unbekannte Felder (z. B. Tippfehler) und `heads`/`layers` ohne `context > 1` werden mit einer
Warnung gemeldet statt still ignoriert.

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
