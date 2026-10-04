# Skull-Tutorial: vom Download zum eigenen kleinen Sprachmodell

Diese Anleitung führt dich in etwa 30 Minuten von der heruntergeladenen Datei zu einem Sprachmodell, das
du selbst trainiert hast. Du brauchst nichts zu installieren und keinen Compiler.

**Was du dabei erwartest, solltest du vorher wissen:** Skull rechnet auf der CPU und trainiert *kleine*
Modelle (einige zehntausend bis wenige Millionen Parameter). Auf einem normalen Rechner entsteht so in
Minuten ein Modell, das Wörter und Satzteile im Stil des Trainingstexts erzeugt. Ein ChatGPT-Ersatz wird
daraus nicht, aber du siehst jeden Schritt des Trainings und kannst alles selbst ausprobieren. Genau dafür
ist Skull gedacht.

---

## 1. Herunterladen und starten

Lade auf der Seite **Releases** des Projekts das Paket für dein System herunter:

| System | Paket | Hinweis |
|---|---|---|
| Linux (x86-64) | `skull-<Version>-linux-x86_64.tar.gz` | läuft auf jeder Distribution (statisch gelinkt) |
| macOS (Apple Silicon: M1 bis M4) | `skull-<Version>-macos-arm64.tar.gz` | Intel-Macs: aus dem Quelltext bauen (`./build.sh`) |
| Windows (x86-64) | `skull-<Version>-windows-x86_64.zip` | keine zusätzliche Installation nötig |

**Welche Variante?** Die normalen Pakete nutzen *AVX2* (eine CPU-Erweiterung, die praktisch jede CPU seit
2013 hat) und sind dadurch deutlich schneller. Meldet Skull beim Start, dass deine CPU AVX2 nicht kann, nimm
das Paket mit `-compat` im Namen. Es läuft auf jeder x86-64-CPU, rechnet aber langsamer.

Entpacken und ausprobieren:

```bash
# Linux / macOS
tar xzf skull-*-linux-x86_64.tar.gz        # macOS: skull-*-macos-arm64.tar.gz
cd skull-*/
./skull examples/hello.skull
```

```powershell
# Windows (PowerShell): Rechtsklick auf die Zip-Datei → "Alle extrahieren", dann
cd skull-1.0.0-windows-x86_64
.\skull.exe examples\hello.skull
```

Du solltest `=== Skull v1.0.0 ===` und danach ein paar Zeilen Ausgabe sehen.

**Warnungen des Betriebssystems:** Die Programme sind nicht signiert (das kostet Geld).
- *macOS* meldet evtl. „kann nicht geöffnet werden". Einmalig im Paketordner: `xattr -d com.apple.quarantine skull`
- *Windows* zeigt evtl. SmartScreen: „Weitere Informationen" → „Trotzdem ausführen".
- Wenn du lieber prüfen willst, was du startest: Die Datei `SHA256SUMS.txt` auf der Release-Seite enthält die
  Prüfsummen (`sha256sum skull-*.tar.gz`), und der gesamte Quelltext liegt im Repository.

Alle Befehle in dieser Anleitung führst du **im entpackten Ordner** aus. Skull-Skripte (`*.skull`) sind
einfache Textdateien; öffne sie mit einem beliebigen Editor.

---

## 2. Die Sprache in fünf Minuten

Lege eine Datei `erste_schritte.skull` an:

```skull
define name = "Skull"
print("Hallo,", name)

define func quadrat(x) { return x * x }

define zahlen = []
for i in 1..5 { push(zahlen, quadrat(i)) }
print(zahlen)

define summe = 0
for z in zahlen { summe = summe + z }
print("Summe:", summe)

if summe > 50 { print("groß") } else { print("klein") }
```

und starte sie mit `./skull erste_schritte.skull`. Die Ausgabe:

```
Hallo, Skull
[1, 4, 9, 16, 25]
Summe: 55
groß
```

Das ist fast die ganze Sprache: Variablen (`define`), Funktionen (`define func`), `if`/`else if`/`else`,
`for` (über Zahlenbereiche, Listen und Text), `while`, Listen (`[1, 2]`, `liste[0]`, `push`, `pop`, `len`)
und Text (`text[0]`, `substr`, `num`). Die vollständige Referenz steht in der `README.md` im Paket
(Abschnitt „Sprach-Referenz"). Wichtiger für dich ist, was jetzt kommt: Skull kann **Modelle trainieren**,
und zwar mit eingebauten Befehlen statt mit Bibliotheken.

---

## 3. Dein erstes Sprachmodell

### Trainingstext besorgen

Ein Sprachmodell lernt aus Text. Für den Anfang eignet sich „Tiny Shakespeare" (1,1 MB, gemeinfrei):

```bash
# Linux / macOS
curl -L -o shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```
```powershell
# Windows
curl.exe -L -o shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Jeder andere Text geht auch (`.txt`, `.md`, `.json`, `.jsonl`, `.csv`), je mehr, desto besser; mit einem
Roman oder deinen eigenen Texten macht es mehr Spaß. Weniger als etwa 100 000 Zeichen reichen nur für sehr
einfache Muster.

### Trainieren und Text erzeugen

Datei `mini.skull`:

```skull
define model Mini {
    dim     = 64     // Breite des Modells
    context = 64     // so viele Zeichen sieht es gleichzeitig (mehr als 1 = Transformer)
    heads   = 4      // Attention-Köpfe (dim muss durch heads teilbar sein)
    layers  = 2      // Anzahl Schichten
}

train Mini {
    data   = "shakespeare.txt"
    out    = "mini.weights"
    epochs = 5
    steps  = 40000   // nur 40 000 Zeichen pro Epoche: schnell für den ersten Versuch
    rate   = 0.003
    batch  = 8
}

generate Mini {
    weights     = "mini.weights"
    prompt      = "ROMEO:"
    tokens      = 300
    temperature = 0.8
    seed        = 1
}
```

`./skull mini.skull` — auf einem 4-Kern-Rechner dauert das etwa 10 Sekunden. Du siehst, wie der Verlust
pro Epoche sinkt:

```
Epoche 1/5  |  Loss: 3.56273  |  Val: 2.73805 (Perplexitaet 15.4567) *  |  Zeit: 1.8319s
Epoche 5/5  |  Loss: 2.47467  |  Val: 2.46061 (Perplexitaet 11.7119) *  |  Zeit: 9.04457s
```

und danach erzeugt das Modell Text. Nach so kurzem Training sieht der noch aus wie Wörter-Brei mit
englischem Klang: Das Modell kennt Buchstabenhäufigkeiten und einige Silben, aber noch keine Wörter. Das ist
normal. Im nächsten Schritt lernst du, woran du erkennst, ob ein Modell besser wird, und wie du es besser
machst.

---

## 4. Wie gut ist das Modell? (Loss, Val, Perplexität)

Der **Loss** misst, wie überrascht das Modell vom nächsten Zeichen ist. Kleiner ist besser. Das Training
senkt ihn fast immer, auch wenn das Modell den Text nur **auswendig lernt**. Deshalb hält Skull die letzten
10 % der Daten zurück, mit denen nie trainiert wird, und zeigt dafür den **Val-Loss**. Nur der zählt:

- **Perplexität** = `exp(Val-Loss)`: grob, unter wie vielen gleich wahrscheinlichen Zeichen das Modell
  schwankt. Ein Modell, das nur rät, hat bei 256 möglichen Zeichen Perplexität 256 (Loss 5,5); Skull-Läufe
  auf Shakespeare enden bei etwa 6 (Loss 1,8), große Modelle erreichen deutlich weniger.
- Das `*` hinter einer Epoche heißt: neue Bestleistung. Skull speichert immer die Gewichte der **besten**
  Epoche, nicht die der letzten.
- Sinkt der Trainings-Loss weiter, aber der Val-Loss steigt, ist das **Überanpassung**. Dann helfen mehr Daten,
  ein kleineres Modell, weniger Epochen oder `patience = 5` (bricht nach 5 Epochen ohne Verbesserung ab).

Aus der Praxis (gemessen, 4 Kerne): Dasselbe Modell (136 448 Parameter) erreicht mit dem Skript aus dem
nächsten Abschnitt nach 40 Epochen (rund 2,5 Minuten) Val-Loss 1,78, Perplexität 5,9. Dann sind echte Wörter,
Zeilenstrukturen und Sprecherwechsel („ROMEO:", „DUKE VINCENTIO:") zu erkennen.

---

## 5. Besser trainieren

Starte dasselbe Skript noch einmal, diesmal mit mehr Zeit (rund 2,5 Minuten). Ändere in `train`:

```skull
train Mini {
    data     = "shakespeare.txt"
    out      = "mini.weights"
    epochs   = 40
    steps    = 100000
    rate     = 0.003
    batch    = 8
    patience = 6      // abbrechen, wenn der Val-Loss 6 Epochen nicht besser wird
}
```

Die wichtigsten Stellschrauben:

| Feld | Wirkung | Faustregel |
|---|---|---|
| `dim`, `layers` | Größe des Modells | größer = lernt mehr, braucht mehr Daten und Zeit |
| `context` | wie viele Zeichen das Modell auf einmal sieht | 64 bis 128 für den Anfang |
| `epochs`, `steps` | wie lange trainiert wird | beobachte den Val-Loss, bis er nicht mehr sinkt |
| `rate` | Lernrate | 0.003 ist ein guter Start; Loss springt wild → kleiner; sinkt kaum → größer |
| `batch` | Sequenzen pro Schritt | größer = ruhigeres Training, mehr Threads nutzbar |
| `threads` | Rechenkerne | 0 = alle (Standard). Das Ergebnis ist **bitgleich**, egal wie viele Threads rechnen |
| `seed` | Zufallsstart | gleicher Seed + gleiche Daten = gleiche Gewichte (bitgleich mit demselben Programm) |

Alle Felder stehen in der README im Abschnitt „Weitere Optionen".

**Pausieren und weitermachen:** Training dauert manchmal länger, als man am Stück Zeit hat.

```skull
// Heute: nach Epoche 20 anhalten
train Mini { data = "shakespeare.txt"  out = "mini.weights"  epochs = 100  stop_after = 20 }

// Morgen: genau dort weitermachen
train Mini { data = "shakespeare.txt"  out = "mini.weights"  epochs = 100  resume = "mini.weights.ckpt" }
```

Skull speichert dabei den *kompletten* Zustand. Das Ergebnis ist **exakt dasselbe**, als hättest du
durchgetrainiert. Mit `checkpoint = 5` entsteht außerdem alle 5 Epochen automatisch ein Sicherungspunkt, falls
der Rechner mittendrin ausgeht.

---

## 6. Text erzeugen und steuern

```skull
generate Mini {
    weights     = "mini.weights"
    prompt      = "ROMEO:"
    tokens      = 500
    temperature = 0.7
    top_k       = 20
    seed        = 7
}
```

- `temperature`: niedrig (0.3) = vorsichtig und oft wiederholend; hoch (1.2) = kreativ und fehlerhaft;
  `0` = immer das wahrscheinlichste Zeichen.
- `top_k = 20` lässt nur die 20 wahrscheinlichsten Zeichen zu, `top_p = 0.9` nur die kleinste Menge, die zusammen
  90 % Wahrscheinlichkeit hat. Beides verhindert, dass seltene Zeichen Unsinn erzeugen.
- Mit `seed` kommt bei jedem Lauf **derselbe** Text heraus (mit demselben Programm und denselben Gewichten),
  ohne `seed` jedes Mal ein anderer.
- `shift = true` macht lange Texte schneller (das Modell vergisst dafür zeitweise die ältere Hälfte seines
  Kontexts).

So sieht ein Text nach dem 40-Epochen-Training aus (`temperature = 0.7`, `top_k = 20`, `seed = 7`; dein Text
kann sich in Einzelheiten unterscheiden, siehe Hinweis zu `seed` in der README):

```
ROMEO:
This lady, hath longing and me in the fine!
For the macking. Angerous and man in proudders;
Which seek to tell some for the birth to my course than
A more out apperous death a land seconding enemy
```

Kein Shakespeare, aber Wörter, Zeilen und Satzmelodie stammen vom Modell, das **du** gerade selbst trainiert hast.

Das Skript kann auch mehrere Texte hintereinander erzeugen oder per Schleife verschiedene Einstellungen
vergleichen:

```skull
for t in [0.3, 0.7, 1.1] {
    print("--- temperature", t)
    generate Mini { weights = "mini.weights"  prompt = "ROMEO:"  tokens = 120  temperature = t  seed = 1 }
}
```

---

## 7. Mit eigenen Daten

- **Format:** `.txt`, `.md`, `.json`, `.jsonl` und `.csv` werden erkannt, der Text wird herausgezogen.
- **Menge:** Je größer das Modell, desto mehr Text braucht es. Als Richtwert für die Größen in dieser
  Anleitung: mindestens 1 MB.
- **Tokenizer:** Standard ist ein Token pro Byte (`vocab = 256`). Mit `bpe = true` lernt Skull häufige
  Zeichenfolgen als Tokens (`bpe_vocab = 1000`). Der Tokenizer steckt in der Gewichte-Datei; `generate`
  benutzt ihn automatisch.
- **Umlaute und Sonderzeichen** funktionieren; Skull arbeitet mit UTF-8-Bytes.

---

## 8. Häufige Meldungen

| Meldung | Bedeutung und Lösung |
|---|---|
| `Datei nicht gefunden: shakespeare.txt` | Skull sucht die Datei **relativ zu dem Ordner, in dem du startest**. Mit `cd` in den Paketordner wechseln oder den Pfad im Skript anpassen. |
| `FEHLER: Diese Skull-Version nutzt AVX2-Befehle ...` (oder `Illegal instruction`) | Deine CPU kann AVX2 nicht: das Paket mit `-compat` im Namen nehmen. |
| `[WARNUNG] Unbekanntes Feld ...` | Tippfehler in einem Feldnamen (z. B. `epoch` statt `epochs`). Das Feld wird ignoriert. |
| `dim (30) muss durch heads (4) teilbar sein` | `dim` muss ein Vielfaches von `heads` sein, z. B. `dim = 64`, `heads = 4`. |
| `Validierung uebersprungen: 0 von 4 Token waeren zu wenig` | Zu wenig Daten (unter 100 Token) für einen Val-Anteil: mehr Text verwenden. |
| `Checkpoint passt nicht zu den aktuellen Daten` | Beim Fortsetzen haben sich Datei, `val` oder `bpe` geändert. Mit den ursprünglichen Einstellungen fortsetzen. |
| Loss steigt auf Hunderte, Perplexität wird astronomisch (z. B. `2e+256`) | Lernrate viel zu groß. `rate = 0.003` ist ein guter Start; Raten bei 256 Zeichen entspricht einem Loss von 5,5. |

Fehler beenden Skull immer mit `[FEHLER] Zeile N: ...` und Exit-Code 1, damit sie in Skripten und CI auffallen.

---

## 9. Wie geht es weiter?

- `examples/` im Paket: kleine, lauffähige Skripte zu Tokenizer, Formaten, Transformer und Generierung.
- `README.md`: vollständige Sprach-Referenz und alle Felder von `train` und `generate`.
- Der Quelltext ist klein und lesbar: `src/transformer.h` enthält das ganze Modell samt Backward-Pass von
  Hand, `src/trainer.h` die Trainingsschleife. Wer wissen will, wie ein Sprachmodell wirklich funktioniert,
  findet hier jede Zeile.
- Fehler, Ideen und Fragen gern als Issue auf GitHub.
