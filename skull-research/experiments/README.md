# Experimente

Skripte, mit denen wir Fragen aus [`../notes/offene_fragen.md`](../notes/offene_fragen.md) **messen**, statt
sie zu vermuten. Alles läuft mit dem normalen `skull`-Programm, es ist nichts weiter zu installieren.

## Vorbereitung (einmal)

Im Ordner `skull/` (dort liegt `build/skull`) den Trainingstext holen:

```bash
curl -L -o shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Starten, ebenfalls aus `skull/`:

```bash
./build/skull ../skull-research/experiments/01_lernrate.skull
```

Beim Training läuft viel Text durch. Das **Ergebnis steht am Ende** unter `ERGEBNIS`. Wer nur die Tabelle will:

```bash
./build/skull ../skull-research/experiments/01_lernrate.skull | sed -n '/ERGEBNIS/,$p'
```

## Die Spielregeln

Damit Zahlen etwas bedeuten, halten sich alle Experimente an dieselben Regeln:

1. **Eine Sache ändern.** Alles andere (Daten, Modellgröße, Seed, Budget) bleibt gleich.
2. **Immer dieselbe Messgröße:** der **Val-Loss** (Fehler auf Text, den das Modell nie gesehen hat), dazu
   Rechenzeit und Parameterzahl. Der Trainings-Loss allein täuscht, er sinkt auch beim Auswendiglernen.
3. **Das Rauschen kennen.** Zwei Läufe mit verschiedenem Seed liefern nie dasselbe Ergebnis (Experiment 02).
   Unterschiede unterhalb der Streuung sind keine Befunde.
4. **Budget festhalten.** Zwei Größen, die man vergleicht, brauchen gleich viel Training (gleiche Epochen und
   `steps`) oder gleich viel Rechenzeit. Wer bei *gleichen Parametern* schneller ist, ist ein anderer Befund als
   wer bei *gleicher Zeit* besser ist. Beides ist interessant, aber getrennt zu benennen.
5. **Ergebnis weitergeben heißt: Skript und Zahlen.** Wer ein Ergebnis nicht mit dem Skript nachlaufen lassen
   kann, hat nichts bewiesen.

## Messwerte im Skript

Nach jedem `train` liefern diese Funktionen die Ergebnisse (siehe auch `skull/README.md`):

| Funktion | Bedeutung |
|---|---|
| `last_val_loss()` | Val-Loss des gespeicherten (besten) Modells |
| `last_loss()` | Trainings-Loss der letzten Epoche |
| `last_best_epoch()` | Epoche des besten Modells |
| `last_seconds()` | Rechenzeit der Epochen in Sekunden |
| `last_params()` | Anzahl trainierbarer Parameter |

Daraus baut ein Skript Listen und Tabellen; ein Beispiel dafür ist jedes Skript hier. Ein neues Experiment
beginnt am einfachsten als Kopie von `01_lernrate.skull`: Liste der Werte austauschen, in der `train`-Zeile das
Feld ändern, das variiert werden soll.

## Die Messlatte

Alle Zahlen: Modell `dim 64, context 64, heads 4, layers 2` (136 448 Parameter), Tiny Shakespeare,
`batch 8`, `rate 0.003`.

| Budget | Einstellung | Val-Loss | Hinweis |
|---|---|---|---|
| kurz | 10 Epochen × 50 000 Token, `seed 1` | **2.1936** | etwa 20 s auf 4 Kernen |
| lang | 40 Epochen × 100 000 Token, `seed 42` (Standard) | **1.7798** | 63 s auf 16 Kernen, 153 s auf 4 Kernen |

Die Zahlen sind mit demselben Programm (gcc, AVX2) auf zwei verschiedenen Rechnern **bitgleich** herausgekommen.
Wer andere Zahlen bekommt, hat vermutlich einen anderen Compiler oder ein anderes Programm; Abweichungen in den
letzten Stellen sind dann normal, größere nicht.

## Bisherige Ergebnisse

### 01 Lernrate (kurzes Budget, `seed 1`)

| rate | Val-Loss |
|---|---|
| 0.0003 | 2.5759 |
| 0.001 | 2.3645 |
| **0.003** | **2.1936** |
| 0.01 | 2.3718 |
| 0.03 | 2.6224 |

Eine klare U-Kurve mit dem Optimum bei 0.003. Der Abstand zu den Nachbarn (0.17 und 0.18) ist mehr als das
Sechsfache der Streuung aus Experiment 02: der Befund ist belastbar, **für dieses Modell und dieses kurze
Budget**. Ob das Optimum bei längerem Training woanders liegt, ist offen (Vermutung: eher niedriger); das wäre ein
eigener Versuch. Alle fünf Läufe waren in der letzten Epoche noch am Verbessern (`beste Epoche = 10`), die Modelle
sind in dieser Kurzmessung also noch nicht ausgelernt.

### 02 Seed-Rauschen (rate 0.003, Seeds 1 bis 5)

Val-Loss 2.1936, 2.1793, 2.1966, 2.1356, 2.1994. Mittelwert **2.181**, Standardabweichung **0.027**,
Spannweite **0.064**. Faustregel: Unterschiede unter etwa **0.05** sind bei diesem Aufbau Rauschen.
Mit nur fünf Seeds ist selbst die Streuung ungenau; für knappe Entscheidungen mehr Seeds verwenden.

## Nächste Schritte

Ideen, die mit dieser Vorlage direkt messbar sind: Modellgröße (`dim`, `layers`) bei gleicher Zeit, `context`,
`batch`, die Länge des Trainings. Die Fragen zu Attention, Loss-Funktionen und Präzision aus den Notizen
brauchen erst Änderungen am Modell selbst; die Messlatte oben ist dann der Vergleichswert.
