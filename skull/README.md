# Skull v1.0.0 - Einfache KI-Trainings-Sprache

**Skull macht KI-Training für jeden zugänglich - ohne komplizierte Setups!**

---

## 🚀 Schnellstart (3 Schritte)

### 1️⃣ Installieren

#### **Linux/macOS:**
```bash
# Abhängigkeiten installieren
sudo apt install build-essential cmake git g++  # Ubuntu/Debian
brew install cmake gcc                              # macOS

# Repository klonen
git clone https://github.com/CentexMC805Paul/skull.git
cd skull/skull

# Einfach bauen
chmod +x build.sh
./build.sh
```

#### **Windows:**
1. **Visual Studio 2022** mit C++-Tools installieren: [Download](https://visualstudio.microsoft.com/downloads/)
2. **Repository klonen:**
   ```cmd
git clone https://github.com/CentexMC805Paul/skull.git
cd skull\skull
```
3. **Einfach bauen:**
   ```cmd
build.bat
```

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

---

## 📚 Einfache Beispiele

| Beispiel | Beschreibung | Befehl |
|----------|-------------|--------|
| `hello.skull` | Grundlagen (Variablen, Funktionen, Schleifen) | `./build/skull examples/hello.skull` |
| `train.skull` | Einfaches KI-Training | `./build/skull examples/train.skull` |
| `generate.skull` | Text generieren | `./build/skull examples/generate.skull` |

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

### 3. Schleifen & Bedingungen
```skull
// Schleife
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

### 4. KI-Modell trainieren
```skull
// Modell definieren
define model MeinModell {
    dim   = 64    // Größe des Modells
    vocab = 256   // Vokabular-Größe
}

// Training starten
train MeinModell {
    data   = "mein_text.txt"  // Deine Trainingsdaten
    epochs = 50              // Anzahl Durchläufe
    rate   = 0.01            // Lernrate
}
```

### 5. Text generieren
```skull
// Modell definieren (muss zum Training passen)
define model MeinModell {
    dim   = 64
    vocab = 256
}

// Text generieren
generate MeinModell {
    weights     = "mein_text.txt.weights"  // Trainierte Gewichte
    prompt      = "Hallo"                   // Starttext
    tokens      = 50                       // Anzahl Wörter
    temperature = 0.8                      // Kreativität (0.0-1.0)
}
```

---

## 🎯 Optionen für Fortgeschrittene

### GPU-Unterstützung
```skull
train MeinModell {
    data   = "mein_text.txt"
    epochs = 50
    rate   = 0.01
    gpu    = true          // GPU aktivieren
}
```

### Mehrere Layer
```skull
define model GroßesModell {
    dim      = 128
    vocab    = 50000
    layers   = 6          // Mehr Layer = bessere Ergebnisse
    heads    = 8          // Attention-Heads (nur für Transformer)
}
```

---

## 🔧 Build-Optionen

| Befehl | Beschreibung |
|--------|-------------|
| `./build.sh` | Standard: CPU mit AVX2 |
| `./build.sh --gpu` | Mit OpenCL GPU-Unterstützung |
| `./build.sh --cuda` | Mit CUDA GPU-Unterstützung (NVIDIA) |
| `./build.sh --debug` | Debug-Modus (mehr Infos) |
| `./build.sh --clean` | Build-Verzeichnis bereinigen |

---

## 📦 Abhängigkeiten

### Linux (Ubuntu/Debian)
```bash
sudo apt install build-essential cmake git g++ opencl-headers ocl-icd-opencl-dev
```

### macOS
```bash
brew install cmake gcc opencl
```

### Windows
- Visual Studio 2022 mit C++-Tools
- Optional: CUDA Toolkit für NVIDIA GPUs

---

## 🤝 Mitmachen

- **Bugs melden:** [GitHub Issues](https://github.com/CentexMC805Paul/skull/issues)
- **Fragen stellen:** [GitHub Discussions](https://github.com/CentexMC805Paul/skull/discussions)
- **Code verbessern:** Pull Requests sind willkommen!

---

## 📜 Lizenz

MIT License - frei für alles, für immer.

---

**Skull - KI-Training für jeden!** 🚀
