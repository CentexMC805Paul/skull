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

> **Ohne Compiler:** Fertige Pakete für Linux, macOS (Apple Silicon) und Windows gibt es auf der
> Seite **Releases** des Repositories. Entpacken, fertig. Die Schritt-für-Schritt-Anleitung vom Download bis
> zum ersten selbst trainierten Modell steht in [`docs/TUTORIAL.md`](docs/TUTORIAL.md).
> Der Rest dieses Abschnitts beschreibt das Bauen aus dem Quelltext.

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

### 3a. Listen und Text
```skull
define xs = [10, 20, 30]
push(xs, 40)                  // anhängen
xs[1] = 99                    // Element ändern
print(xs, len(xs), xs[0], xs[-1])   // [10, 99, 30, 40] 4 10 40   (-1 = letztes Element)
define last = pop(xs)         // letztes Element entfernen und zurückgeben

for x in xs { print(x) }      // über eine Liste laufen (oder über die Zeichen eines Texts)

define m = [[1, 2], [3, 4]]   // verschachtelt
m[1][0] = 7

define t = "häll€"
print(len(t), t[1], substr(t, 1, 3))   // 5 ä äll   -> ein Zeichen ist ein UTF-8-Zeichen, nicht ein Byte
define n = num("12.5") + 1            // Text -> Zahl (Fehler, wenn es keine Zahl ist)
```
- **Listen sind Verweise:** `b = a` zeigt auf *dieselbe* Liste (wie in Python); Funktionen, die eine Liste
  bekommen, können sie ändern. Eine Kopie bekommst du mit `b = a + []`.
- `liste + liste` hängt zusammen und legt eine neue Liste an; `"Text" + liste` schreibt die Liste als Text.
- `==` vergleicht Listen Element für Element. Eine leere Liste ist `false`, sonst `true`.
- Indizes sind ganze Zahlen ab 0; negative zählen vom Ende. Außerhalb gibt es einen Fehler mit Zeilennummer.
- Text lässt sich nicht ändern (`t[0] = "x"` ist ein Fehler); baue stattdessen einen neuen Text mit `substr` und `+`.
  `text[i]` muss die Zeichen davor zählen — für lange Texte ist `for c in text { ... }` schneller als eine
  Schleife mit Index.
- `for x in liste` läuft so oft, wie die Liste beim Start lang war; `push` in der Schleife macht sie nicht endlos.
- Eine Liste kann sich nicht selbst enthalten (`push(a, a)` ist ein Fehler).
- Grenzen: höchstens 10 000 000 Elemente pro Liste, 256 MB pro Text; `print` kürzt sehr lange Listen mit `...`.

| Funktion | Bedeutung |
|---|---|
| `len(x)` | Länge einer Liste oder eines Texts (in Zeichen) |
| `push(liste, wert)` | hängt an |
| `pop(liste)` | entfernt das letzte Element und gibt es zurück |
| `substr(text, start, anzahl)` | Teiltext ab `start` (0-basiert); eine zu große `anzahl` wird gekürzt |
| `num(text)` | Text → Zahl (`"12"`, `"-3.5"`; alles andere ist ein Fehler) |
| `str(x)` | Wert als Text |

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
Weitere Felder von `generate`:

| Feld | Standard | Bedeutung |
|------|----------|-----------|
| `seed` | zufällig | Mit Seed ist die Ausgabe reproduzierbar: dasselbe Programm mit denselben Gewichten liefert denselben Text. Zwischen verschiedenen Builds (anderer Compiler, AVX2 an/aus) können Rundungsunterschiede selten eine andere Zeichenwahl auslösen |
| `top_k` | 0 | Nur die k wahrscheinlichsten Token zulassen (0 = aus) |
| `top_p` | 1 | Nur die kleinste Menge wahrscheinlichster Token zulassen, deren Wahrscheinlichkeiten sich zu `top_p` summieren (1 = aus) |

`top_k`/`top_p` wirken nur bei `temperature > 0`. Sie halten das Modell davon ab, seltene, meist
unsinnige Token zu würfeln, ohne es wie `temperature = 0` komplett festzunageln (`top_k = 1` ist
identisch zu greedy).

Beim Transformer gibt es `shift`: Standardmäßig (`false`) sieht das Modell immer exakt die letzten
`context` Token. Das Generieren nutzt dabei einen KV-Cache, solange der Kontext noch nicht voll ist;
danach muss jedes Token neu gerechnet werden. Mit `shift = true` wird bei vollem Kontext nur die
jüngere Hälfte behalten und neu aufgebaut. Jedes Token bleibt dann billig (gemessen: 300 Token mit
Kontext 64 in 0,04 s statt 0,83 s, also 20× schneller), das Modell sieht dafür zeitweise nur
`context/2` bis `context` Token.

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
| `val_skip` | 0 | Nur Transformer: so viele Positionen am Anfang **jedes Validierungsfensters** werden nicht mitgezählt. Der Val-Loss wird auf Fenstern der Länge `context` berechnet, und die ersten Zeichen eines Fensters haben kaum Vorwissen; bei kurzem `context` fällt das stärker ins Gewicht. Mit `val_skip = context / 2` zählen nur Zeichen, die mindestens halb so viel Vorwissen haben. Nützlich beim **Vergleich verschiedener `context`-Werte**; muss kleiner als `context` sein. Werte mit und ohne `val_skip` sind nicht vergleichbar |
| `threads` | 0 | Nur Transformer: Anzahl Rechen-Threads (0 = alle Kerne). Parallel laufen die Sequenzen eines Batches (`batch > 1`) und die Validierung. **Das Ergebnis ist bitgleich, egal wie viele Threads rechnen.** |
| `seed` | 42 | Zufallsstart für Startgewichte und die Reihenfolge der Trainingsfenster. Gleicher Seed + gleiche Daten = **gleiche Gewichte** (bitgleich mit demselben Programm; zwischen verschiedenen Builds, also anderer Compiler oder AVX2 an/aus, unterscheiden sie sich nur in den letzten Stellen: gemessen gleicher Loss auf 6 Stellen). Die Startgewichte sind auf jedem System identisch |
| `checkpoint` | 0 | Alle N Epochen den kompletten Trainingszustand nach `<out>.ckpt` schreiben (0 = aus) |
| `stop_after` | 0 | Nach dieser Epoche **pausieren** und einen Checkpoint schreiben (0 = bis `epochs`) |
| `resume` | – | Pfad eines Checkpoints: Training dort **fortsetzen** |
| `patience` | 0 | Early Stopping: Abbruch, wenn sich der Validierungs-Loss N Epochen nicht verbessert (0 = aus) |
| `context` | 1 | 1 = Bigram, > 1 = Transformer mit diesem Kontext (max. 8192) |
| `heads`, `layers` | 2, 1 | Nur Transformer. `dim` muss durch `heads` teilbar sein |
| `bpe`, `bpe_vocab` | false, 1000 | Byte-Pair-Encoding statt Bytes als Token. Der Tokenizer wird in der Gewichte-Datei mitgespeichert, `generate` benutzt ihn automatisch |
| `gpu`, `prefer_amd` | false | Zeigt das OpenCL-Gerät an. **Das Training läuft trotzdem auf der CPU** (Hinweis wird ausgegeben) |

### Pausieren und Fortsetzen
```skull
// 1. Lauf: nach Epoche 20 anhalten (z. B. um zu schauen, ob das Training gut aussieht)
train Mini { data = "text.txt"  out = "mini.weights"  epochs = 100  stop_after = 20 }

// später: dort weitermachen, bis Epoche 100
train Mini { data = "text.txt"  out = "mini.weights"  epochs = 100  resume = "mini.weights.ckpt" }
```
Ein Checkpoint enthält den **ganzen** Trainingszustand (Gewichte, Adam-Momente, Zufallsgenerator,
Stelle im Lernraten-Plan, bisher bestes Modell). Pausieren und Fortsetzen ergibt deshalb **exakt
dieselben Gewichte** wie ein durchgehender Lauf. Skull prüft beim Fortsetzen, ob Modellart, Größe
und Daten (inkl. `val`/`bpe`) zum Checkpoint passen, und lehnt sonst mit einer Meldung ab. Wer
`epochs` gegenüber dem ersten Lauf ändert, ändert damit auch den Lernraten-Plan.

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
Modells), `last_best_epoch()`, `last_seconds()` (Rechenzeit der Epochen) und `last_params()` (Anzahl der
Parameter). Beispiele für ganze Messreihen: [`../skull-research/experiments/`](../skull-research/experiments/README.md).

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
(AddressSanitizer/UBSan für Tests), `-DSKULL_TSAN=ON` (ThreadSanitizer), `-DSKULL_USE_OPENCL=ON`,
`-DSKULL_STATIC_RUNTIME=ON` (Laufzeitbibliotheken statisch einbinden; so werden die fertigen Pakete gebaut).
Ein mit AVX2 gebautes Programm prüft beim Start, ob die CPU AVX2 kann, und nennt sonst die Lösung.

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
