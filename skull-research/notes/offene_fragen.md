# Offene Fragen — Skull Research

Gestartet: 2025

---

## 1. Attention-Komplexitat

**Problem:** Standard Attention ist O(seq_len^2) in Speicher und Rechenzeit.
Bei seq_len=1024 sind das 1 Million Attention-Gewichte pro Layer.

**Was wir wissen:**
- FlashAttention loest das durch clevere Speicherverwaltung (IO-aware)
- Linear Attention versucht O(n) aber verliert Ausdrueckskraft
- Sparse Attention (wie in Longformer) begrenzt den Aufmerksamkeitsbereich

**Unsere Frage:**
Gibt es eine Naeherung die auf kleiner Hardware (CPU, Consumer-GPU)
schneller ist als Standard-Attention, ohne signifikante Qualitaetseinbussen?

**Status:** Offen — noch keine Experimente

---

## 2. Gradient-Akkumulation ohne vollstaendigen Forward-Pass

**Problem:** Backprop braucht alle Zwischenwerte aus dem Forward-Pass.
Bei grossen Modellen bedeutet das enormen Speicherbedarf.

**Was wir wissen:**
- Gradient Checkpointing speichert nur Teile, berechnet den Rest neu
- Das kostet ~33% mehr Rechenzeit, spart aber 80% Speicher

**Unsere Frage:**
Gibt es eine mathematisch exakte Methode die WENIGER Speicher braucht
als normales Backprop, ohne mehr neu zu berechnen?

**Status:** Theoretisch interessant — Literatur noch nicht vollstaendig gelesen

---

## 3. Bessere Initialisierung

**Problem:** Xavier/Kaiming Initialisierung ist Standard aber nicht optimal
fuer alle Architekturen.

**Was wir wissen:**
- Schlechte Initialisierung fuehrt zu explodierenden/verschwindenden Gradienten
- Gute Initialisierung kann Trainingszeit halbieren

**Unsere Frage:**
Kann Skull die Initialisierung automatisch an die Modell-Groesse anpassen?

**Status:** Einfach zu testen — erstes Experiment moeglich

---

## 4. Loss-Funktion Alternativen

**Problem:** Cross-Entropy ist standard fuer Language Models. Ist es optimal?

**Was wir wissen:**
- Focal Loss (aus Computer Vision) gibt seltenen Tokens mehr Gewicht
- Label Smoothing stabilisiert Training aber verwascht Confidences

**Unsere Frage:**
Gibt es eine Loss-Funktion die auf kleinen Datenmengen schneller konvergiert?

**Status:** Offen — interessant fuer Skull's Use-Case (kleine Trainingsdaten)
