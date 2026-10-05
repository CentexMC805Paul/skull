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

### 03 Modellgröße bei gleicher Rechenmenge (`seed 1`, `rate 0.003`, 50 000 Token pro Epoche)

Rechenmenge = Token × Parameter, für alle gleich (so viel wie die Messlatte `mittel` mit 24 Epochen, das sind
rund 1,2 Millionen gesehene Token). Kleinere Modelle bekommen entsprechend mehr Epochen.

| Modell | dim / Schichten | Parameter | Epochen | Val-Loss | Sekunden (4 Kerne) |
|---|---|---|---|---|---|
| winzig | 32 / 1 | 31 072 | 105 | 1.9584 | 67 |
| **klein** | 32 / 2 | 43 648 | 75 | **1.8874** | 72 |
| mittel | 64 / 2 | 136 448 | 24 | 1.9212 | 54 |
| tief | 64 / 4 | 235 904 | 14 | 2.0065 | 62 |
| breit | 128 / 2 | 469 504 | 7 | 2.2178 | 46 |
| groß | 128 / 4 | 865 024 | 4 | 2.4001 | 55 |

**Was belastbar ist** (Unterschied über der Streuung von etwa 0.03 bis 0.05): Bei diesem kleinen Budget sind
Modelle ab etwa 235 000 Parametern deutlich schlechter, und zwar umso mehr, je größer sie sind (+0.12 bis
+0.51 gegenüber `klein`). Sie sehen zu wenige Token pro Parameter: `groß` hat in 4 Epochen nur etwa 0,2 Token
je Parameter gesehen, `mittel` rund 9. Auch `winzig` ist schlechter als `klein` (+0.07).

**Was nicht belastbar ist:** `klein` gegen `mittel` (0.034 Unterschied, im Rauschen). Wer „klein ist besser als
mittel“ behauptet, hat es nicht gemessen.

**Was dieser Versuch nicht sagt:**
- Dass es bei mehr Rechenmenge so bleibt. Die Vermutung war, dass sich das Optimum mit mehr Rechenmenge zu
  größeren Modellen verschiebt (in der Literatur oft als „Chinchilla-Regel“ beschrieben); Experiment 04 hat das
  danach gemessen (siehe unten).
- Dass `rate 0.003` für alle Größen passt. Große Modelle brauchen oft eine kleinere Lernrate; ein Teil ihres
  Rückstands kann daran liegen.
- Dass „gleiche Rechenmenge“ gleich viel Zeit heißt: die Spalte „Sekunden“ streut von 46 bis 72, weil kleine
  Matrizen schlechter ausgelastet werden und die Validierung pro Epoche mitläuft. Bei *gleicher Zeit* bekämen
  die großen Modelle etwas mehr Training; die große Lücke (`groß`, `breit`) würde das nicht schließen.
- Nur ein Seed pro Modell.

### 04 Dieselbe Frage bei vierfacher Rechenmenge (`basis_epochs = 96`, sonst wie 03)

Gemessen auf Pauls Rechner (16 Kerne, 8 Threads). Bestätigung auf einem zweiten Rechner: siehe unten.

| Modell | Parameter | Epochen | Val-Loss 1× (03) | Val-Loss 4× (04) | Änderung |
|---|---|---|---|---|---|
| winzig | 31 072 | 422 | 1.9584 | 1.9463 | −0.012 |
| klein | 43 648 | 300 | **1.8874** | 1.8374 | −0.050 |
| mittel | 136 448 | 96 | 1.9212 | **1.7620** | −0.159 |
| tief | 235 904 | 56 | 2.0065 | **1.7426** | −0.264 |
| breit | 469 504 | 28 | 2.2178 | 1.8452 | −0.373 |
| groß | 865 024 | 15 | 2.4001 | 2.0363 | −0.364 |

**Was belastbar ist:** Das beste Modell hat sich verschoben. Bei 1× war es `klein` (44 000 Parameter), bei 4× sind
es `mittel` und `tief` (136 000 bis 236 000); `klein` liegt jetzt 0.075 bzw. 0.095 dahinter, mehr als die
Rauschgrenze von etwa 0.05. Alle Modelle werden mit mehr Rechenmenge besser, die größeren aber viel stärker
(bis −0.37) als die kleinen (−0.01 bis −0.05). Damit ist die Vermutung aus 03 in diesem Ausschnitt bestätigt: mit mehr
Rechenmenge lohnt sich ein größeres Modell.

**Was nicht belastbar ist:** `mittel` gegen `tief` (0.019, Rauschen). Aus zwei Punkten (1× und 4×) lässt sich
auch nicht ablesen, *wie schnell* das beste Modell mit der Rechenmenge wächst; dafür bräuchte es mehr Stufen.

**Nebenbefunde:** `winzig` verbessert sich kaum (−0.012): mit 31 000 Parametern ist seine Kapazität erreicht, mehr Training
hilft nicht mehr. `groß` liegt auch bei 4× noch 0.29 hinter dem Besten; vermutlich braucht es dafür noch deutlich mehr
Rechenmenge (Vermutung, nicht gemessen).

**Grenzen:**
- Der Trainingstext ist nur 1,1 MB groß. Die kleinen Modelle sehen ihn bei 4× etwa 20-mal (422 Epochen × 50 000
  Token), die großen nur wenige Male. Das vermischt „mehr Parameter“ mit „wie oft derselbe Text gelesen wird“. Mit sehr
  viel mehr Text ließe sich die Frage sauberer beantworten.
- Die Lernrate war für alle gleich (0.003), es gab nur einen Seed, und das Ergebnis gilt für dieses Modellschema.

## Nächste Schritte

Ideen, die mit dieser Vorlage direkt messbar sind: die Lernrate pro Modellgröße einstellen (bisher fest 0.003, bei
großen Modellen vermutlich zu hoch); mehr und größere Texte; weitere Stufen der Rechenmenge, um zu sehen, wie das
beste Modell mit ihr wächst; `context`, `batch`. Die Fragen zu Attention, Loss-Funktionen und Präzision aus den Notizen
brauchen erst Änderungen am Modell selbst; die Messlatte oben ist dann der Vergleichswert.
