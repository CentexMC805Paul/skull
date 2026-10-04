#pragma once
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <functional>
#include <cstring>
#include <cctype>
#include <cstdint>
#include <array>
#include <stdexcept>

// ============================================================
//  SKULL TOKENIZER  v0.7.0
//
//  Unterstuetzte Eingabeformate:
//    .txt   — Roher Text
//    .md    — Markdown (Formatierung wird entfernt)
//    .json  — JSON, extrahiert alle String-Werte
//    .jsonl — JSON Lines (ein JSON-Objekt pro Zeile)
//    .csv   — CSV, extrahiert Text-Spalten
//
//  Tokenisierungsmethoden:
//    1. CHAR   — Ein Zeichen = ein Token (wie bisher)
//    2. BPE    — Byte-Pair Encoding (wie GPT)
//               Lernt haeufige Zeichenpaare und fasst sie zusammen
//               Beispiel: "hallo" -> ["hall", "o"] statt ["h","a","l","l","o"]
//
//  BPE-Vorteile:
//    - Vokabular viel kleiner als Wort-basiert
//    - Funktioniert sprachunabhaengig
//    - Unbekannte Woerter werden trotzdem tokenisiert
// ============================================================

// ---- Dateiformat-Erkennung ----
enum class FileFormat { TXT, MD, JSON, JSONL, CSV, UNKNOWN };

inline FileFormat detect_format(const std::string& path) {
    // Endung nur im letzten Pfadteil suchen ("dir.v2/daten" hat keine Endung)
    size_t slash = path.find_last_of("/\\");
    size_t dot   = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash))
        return FileFormat::TXT;
    std::string ext = path.substr(dot);
    for (auto& ch : ext) ch = (char)std::tolower((unsigned char)ch);
    if (ext == ".txt")   return FileFormat::TXT;
    if (ext == ".md")    return FileFormat::MD;
    if (ext == ".json")  return FileFormat::JSON;
    if (ext == ".jsonl") return FileFormat::JSONL;
    if (ext == ".csv")   return FileFormat::CSV;
    return FileFormat::TXT; // Default: als Text behandeln
}

// ---- Text-Extraktion nach Format ----

// Markdown: Formatierungs-Symbole entfernen
inline std::string extract_markdown(const std::string& raw) {
    std::string result;
    result.reserve(raw.size());
    size_t i = 0;
    while (i < raw.size()) {
        // Ueberschriften: # ## ### -> weglassen
        if (raw[i] == '#') {
            while (i < raw.size() && raw[i] == '#') i++;
            while (i < raw.size() && raw[i] == ' ') i++;
            continue;
        }
        // Code-Bloecke: ``` ... ``` -> weglassen
        if (i + 2 < raw.size() && raw[i]=='`' && raw[i+1]=='`' && raw[i+2]=='`') {
            i += 3;
            while (i + 2 < raw.size() &&
                   !(raw[i]=='`' && raw[i+1]=='`' && raw[i+2]=='`'))
                i++;
            i += 3;
            continue;
        }
        // Inline-Code: `...` -> Inhalt behalten
        if (raw[i] == '`') {
            i++;
            while (i < raw.size() && raw[i] != '`')
                result += raw[i++];
            if (i < raw.size()) i++;
            continue;
        }
        // Bold/Italic: ** * -> weglassen
        if (raw[i] == '*') {
            while (i < raw.size() && raw[i] == '*') i++;
            continue;
        }
        // Links: [text](url) -> nur text behalten
        if (raw[i] == '[') {
            i++;
            while (i < raw.size() && raw[i] != ']')
                result += raw[i++];
            if (i < raw.size()) i++; // ]
            // (url) ueberspringen
            if (i < raw.size() && raw[i] == '(') {
                i++;
                while (i < raw.size() && raw[i] != ')') i++;
                if (i < raw.size()) i++;
            }
            continue;
        }
        result += raw[i++];
    }
    return result;
}

// Einfacher JSON-String-Extraktor (ohne externe Library)
// Findet alle "value"-Strings in JSON
inline std::string extract_json_strings(const std::string& raw) {
    std::string result;
    size_t i = 0;
    while (i < raw.size()) {
        if (raw[i] == '"') {
            i++; // oeffnendes "
            std::string s;
            while (i < raw.size() && raw[i] != '"') {
                if (raw[i] == '\\' && i + 1 < raw.size()) {
                    i++; // escape-Zeichen
                    if (raw[i] == 'n') s += '\n';
                    else if (raw[i] == 't') s += '\t';
                    else s += raw[i];
                } else {
                    s += raw[i];
                }
                i++;
            }
            if (i < raw.size()) i++; // schliessendes "
            // Nur wenn der String laenger als 2 Zeichen ist (kein Key-Name)
            if (s.size() > 2) {
                result += s + " ";
            }
        } else {
            i++;
        }
    }
    return result;
}

// JSONL: Jede Zeile ist ein JSON-Objekt
// Extrahiert "text", "content", "prompt", "completion" Felder
inline std::string extract_jsonl(const std::string& raw) {
    std::string result;
    std::istringstream stream(raw);
    std::string line;

    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        // Suche nach bekannten Feldern
        for (const auto& field : {"\"text\":", "\"content\":",
                                   "\"prompt\":", "\"completion\":",
                                   "\"message\":", "\"body\":"}) {
            size_t pos = line.find(field);
            if (pos == std::string::npos) continue;
            pos += strlen(field);
            while (pos < line.size() && (line[pos] == ' ' || line[pos] == '"')) pos++;
            size_t end = line.find('"', pos);
            if (end == std::string::npos) end = line.size();
            result += line.substr(pos, end - pos) + " ";
        }
    }
    return result;
}

// CSV: Erste Spalte oder alle Text-Spalten extrahieren
inline std::string extract_csv(const std::string& raw) {
    std::string result;
    std::istringstream stream(raw);
    std::string line;
    bool first_line = true;

    while (std::getline(stream, line)) {
        if (first_line) { first_line = false; continue; } // Header ueberspringen
        if (line.empty()) continue;
        // Komma-getrennte Felder
        std::istringstream row(line);
        std::string cell;
        while (std::getline(row, cell, ',')) {
            // Anfuehrungszeichen entfernen
            if (cell.size() > 1 && cell.front() == '"' && cell.back() == '"')
                cell = cell.substr(1, cell.size() - 2);
            // Nur Textzellen (keine reinen Zahlen)
            bool is_number = !cell.empty() &&
                std::all_of(cell.begin(), cell.end(),
                    [](char c){ return isdigit(c) || c == '.' || c == '-'; });
            if (!is_number && !cell.empty())
                result += cell + " ";
        }
    }
    return result;
}

// ---- Text aus Datei laden und extrahieren ----
inline std::string load_and_extract(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Datei nicht gefunden: " + path);
    std::stringstream buf;
    buf << file.rdbuf();
    std::string raw = buf.str();

    FileFormat fmt = detect_format(path);
    switch (fmt) {
        case FileFormat::MD:    return extract_markdown(raw);
        case FileFormat::JSON:  return extract_json_strings(raw);
        case FileFormat::JSONL: return extract_jsonl(raw);
        case FileFormat::CSV:   return extract_csv(raw);
        default:                return raw;
    }
}

// ============================================================
//  BYTE-PAIR ENCODING (BPE) TOKENIZER
//
//  Wie BPE funktioniert:
//  1. Starte mit Zeichen-Level-Tokens (jedes Byte = 1 Token, ID 0..255)
//  2. Zaehle alle Paare von benachbarten Tokens
//  3. Fasse das haeufigste Paar zu einem neuen Token zusammen
//  4. Wiederhole bis Vokabular-Groesse erreicht
//
//  Ergebnis: Haeufige Woerter/Silben werden zu eigenen Tokens.
//  Vokabular-Groesse ist kontrollierbar.
//
//  Festgelegte Semantik (die schnelle Umsetzung unten liefert exakt das
//  Ergebnis dieser einfachen Beschreibung, tests/bpecheck.cpp prueft das
//  gegen eine absichtlich naive Referenz):
//
//  Training
//    - Haeufigkeit von (a,b) = Anzahl der Positionen i der AKTUELLEN Folge mit
//      tokens[i]==a und tokens[i+1]==b (Ueberlappungen zaehlen: "aaa" hat
//      zweimal (a,a)).
//    - Gewaehlt wird das haeufigste Paar, bei Gleichstand das lexikographisch
//      kleinste (ID_a, ID_b). Ende, wenn die beste Haeufigkeit < 2 ist oder
//      das Vokabular die Zielgroesse erreicht hat. Es gibt KEINE Obergrenze
//      fuer die Zahl der Merges.
//    - Ein Merge ersetzt die Folge von links nach rechts, ohne Ueberlappung
//      ("aaa" mit (a,a) -> [aa, a]).
//    - Das neue Token ist add_token(vocab[a] + vocab[b]). Zwei verschiedene
//      Merges koennen denselben String ergeben (("a","bc") und ("ab","c"));
//      dann liefert add_token die vorhandene ID und das Vokabular waechst nicht.
//
//  Encoding
//    - Entspricht dem nacheinander Anwenden ALLER Merges in ihrer Reihenfolge
//      (jeder Merge: Text von links nach rechts, ohne Ueberlappung ersetzen).
//
//  Umsetzung (damit 1 MB Text in Sekundenbruchteilen statt Stunden geht):
//    - Tokens sind ints, der Text liegt in einer doppelt verketteten Liste.
//    - Training: Paar -> Haeufigkeit + Positionsliste, die Haeufigkeiten werden
//      bei jedem Merge nur an den Nachbarn der ersetzten Stellen angepasst.
//      Ein Max-Heap mit "lazy deletion" liefert das beste Paar; veraltete
//      Eintraege werden beim Herausnehmen geprueft und verworfen oder
//      korrigiert.
//    - Encode: Min-Heap nach Merge-Rang (bei Gleichstand die linkeste Stelle),
//      O(n log n) statt O(Merges x n).
// ============================================================

namespace bpe_detail {

static_assert(sizeof(int) >= 4, "BPE: int muss mindestens 32 Bit haben");

inline constexpr int32_t  DEAD    = -2;                 // Position ist in ihren linken Nachbarn verschmolzen
inline constexpr uint32_t NO_RANK = 0xFFFFFFFFu;        // "kein Merge"
inline constexpr size_t   MAX_TEXT = 0x7FFFFFF0u;       // Positionen und Zaehler passen in int32_t

// Zwei Token-IDs (beide >= 0) zu einem 64-Bit-Schluessel packen. Die numerische
// Ordnung der Schluessel ist die lexikographische Ordnung der Paare (a, b).
inline uint64_t pair_key(int32_t a, int32_t b) {
    return ((uint64_t)(uint32_t)a << 32) | (uint64_t)(uint32_t)b;
}

// std::hash<uint64_t> ist je nach Standardbibliothek die Identitaet (libstdc++,
// libc++) oder etwas anderes (MSVC). Der eigene Mischer (splitmix64) verteilt
// ueberall gleich gut, auch bei Zweierpotenz-Tabellen.
struct PairHash {
    size_t operator()(uint64_t x) const {
        x += 0x9E3779B97F4A7C15ull;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        return (size_t)(x ^ (x >> 31));
    }
};

inline void check_text_size(size_t n) {
    if (n > MAX_TEXT)
        throw std::runtime_error("BPE: Text zu lang (mehr als 2 GiB)");
}

// Ein Merge in ID-Form: a + b -> x. a < 0 bedeutet "tote Regel": ein Bestandteil
// ist nicht im Vokabular, die Regel kann nie greifen.
struct MergeRule {
    int32_t a = -1, b = -1, x = -1;
};

// Aus vocab/merges abgeleitete ID-Tabelle fuer encode(). Rang = Index in merges.
// Ein Paar kann mehrfach als Merge vorkommen (z.B. wenn ein spaeterer Merge
// dasselbe Token noch einmal erzeugt); first_rank hat den kleinsten Rang,
// more_ranks die weiteren (aufsteigend). Fast immer ist more_ranks leer.
struct MergeTable {
    std::array<int32_t, 256> byte_id;                   // ID des Ein-Byte-Tokens, sonst -1
    std::vector<MergeRule>   rules;
    std::unordered_map<uint64_t, uint32_t, PairHash>              first_rank;
    std::unordered_map<uint64_t, std::vector<uint32_t>, PairHash> more_ranks;
    size_t n_merges = 0;   // Stand von merges / vocab, zu dem die Tabelle passt
    size_t n_vocab  = 0;

    MergeTable() { byte_id.fill(-1); }

    void add_rule(int32_t a, int32_t b, int32_t x) {
        const uint32_t rank = (uint32_t)rules.size();
        if (a < 0 || b < 0 || x < 0) { rules.push_back(MergeRule()); return; }
        rules.push_back(MergeRule{a, b, x});
        const uint64_t key = pair_key(a, b);
        if (!first_rank.emplace(key, rank).second) more_ranks[key].push_back(rank);
    }

    // Kleinster Rang > after, mit dem das Paar verschmolzen wird (NO_RANK, wenn
    // keiner mehr kommt). Ein Merge wirkt nur auf Paare, die nach ihm entstehen
    // oder schon da waren - nie rueckwirkend auf Merges mit kleinerem Rang.
    uint32_t rank_after(uint64_t key, int64_t after) const {
        const auto it = first_rank.find(key);
        if (it == first_rank.end()) return NO_RANK;
        if ((int64_t)it->second > after) return it->second;
        if (more_ranks.empty()) return NO_RANK;
        const auto m = more_ranks.find(key);
        if (m == more_ranks.end()) return NO_RANK;
        for (uint32_t r : m->second)
            if ((int64_t)r > after) return r;
        return NO_RANK;
    }
};

// Text -> Token-IDs. Verkettete Liste + Min-Heap nach (Rang, Position):
// immer der Merge mit dem kleinsten Rang, bei Gleichstand die linkeste Stelle.
// Das entspricht dem nacheinander Anwenden aller Merges (siehe oben); neu
// entstehende Paare bekommen ihren Rang NACH dem gerade angewendeten.
inline std::vector<int> encode_ids(const MergeTable& t, const std::string& text) {
    const size_t n = text.size();
    check_text_size(n);
    std::vector<int> tok(n);
    for (size_t i = 0; i < n; ++i) tok[i] = t.byte_id[(unsigned char)text[i]];
    if (n < 2 || t.first_rank.empty()) return tok;

    std::vector<int32_t> prv(n), nxt(n);
    for (size_t i = 0; i < n; ++i) {
        prv[i] = (int32_t)i - 1;
        nxt[i] = (i + 1 < n) ? (int32_t)(i + 1) : -1;
    }

    // Eintrag = Rang (obere 32 Bit) und Position des linken Tokens (untere 32 Bit)
    std::vector<uint64_t> heap;
    for (size_t i = 0; i + 1 < n; ++i) {
        if (tok[i] < 0 || tok[i + 1] < 0) continue;     // unbekanntes Byte: nie Teil eines Merges
        const uint32_t r = t.rank_after(pair_key(tok[i], tok[i + 1]), -1);
        if (r != NO_RANK) heap.push_back(((uint64_t)r << 32) | (uint64_t)i);
    }
    std::make_heap(heap.begin(), heap.end(), std::greater<uint64_t>());

    while (!heap.empty()) {
        std::pop_heap(heap.begin(), heap.end(), std::greater<uint64_t>());
        const uint64_t e = heap.back();
        heap.pop_back();
        const uint32_t r = (uint32_t)(e >> 32);
        const int32_t  i = (int32_t)(e & 0xFFFFFFFFu);
        const MergeRule& rule = t.rules[r];

        // Veraltete Eintraege (Position verschmolzen oder Nachbar hat sich geaendert) ueberspringen
        const int32_t j = nxt[(size_t)i];
        if (j < 0 || tok[(size_t)i] != rule.a || tok[(size_t)j] != rule.b) continue;

        const int32_t q = nxt[(size_t)j];
        tok[(size_t)i] = rule.x;
        tok[(size_t)j] = DEAD;
        nxt[(size_t)i] = q;
        if (q >= 0) prv[(size_t)q] = i;

        // Die zwei neuen Nachbarschaften bekommen ihren Rang (nur Merges nach r)
        const int32_t p = prv[(size_t)i];
        if (p >= 0 && tok[(size_t)p] >= 0) {
            const uint32_t rr = t.rank_after(pair_key(tok[(size_t)p], rule.x), (int64_t)r);
            if (rr != NO_RANK) {
                heap.push_back(((uint64_t)rr << 32) | (uint64_t)(uint32_t)p);
                std::push_heap(heap.begin(), heap.end(), std::greater<uint64_t>());
            }
        }
        if (q >= 0 && tok[(size_t)q] >= 0) {
            const uint32_t rr = t.rank_after(pair_key(rule.x, tok[(size_t)q]), (int64_t)r);
            if (rr != NO_RANK) {
                heap.push_back(((uint64_t)rr << 32) | (uint64_t)(uint32_t)i);
                std::push_heap(heap.begin(), heap.end(), std::greater<uint64_t>());
            }
        }
    }

    // Verschmolzene Positionen entfernen (Reihenfolge bleibt erhalten)
    size_t w = 0;
    for (size_t i = 0; i < n; ++i)
        if (tok[i] != DEAD) tok[w++] = tok[i];
    tok.resize(w);
    tok.shrink_to_fit();
    return tok;
}

}  // namespace bpe_detail

struct BPETokenizer {
    // Vokabular: Token-ID -> String
    std::vector<std::string> vocab;
    // Umgekehrt: String -> Token-ID
    std::unordered_map<std::string, int> vocab_map;
    // BPE-Merge-Regeln: (a, b) -> merged. Das ist auch das Speicherformat
    // (TOKB-Abschnitt der Gewichte-Datei); alles andere wird daraus abgeleitet.
    std::vector<std::pair<std::string, std::string>> merges;

    // Abgeleitete ID-Tabelle fuer encode() (nur intern). add_token() und
    // add_merge() halten sie aktuell; wer vocab/merges direkt aendert, ruft
    // rebuild_table() auf. Passt sie nicht mehr zu vocab/merges, baut encode()
    // sie fuer den Aufruf neu auf (korrekt, aber langsamer).
    bpe_detail::MergeTable table;

    int vocab_size() const { return (int)vocab.size(); }

    // Token-ID fuer einen String (oder -1 wenn nicht gefunden)
    int token_id(const std::string& s) const {
        auto it = vocab_map.find(s);
        return (it != vocab_map.end()) ? it->second : -1;
    }

    // Neues Token hinzufuegen (oder die vorhandene ID liefern)
    int add_token(const std::string& s) {
        const auto it = vocab_map.find(s);
        if (it != vocab_map.end()) return it->second;
        const bool in_sync = table_current();
        const int id = insert_token(s);
        // Die Basis-Tokens (vor dem ersten Merge) fuehrt die Tabelle mit. Ein Token,
        // das nach Merges dazukommt, macht sie ungueltig, bis rebuild_table() laeuft.
        if (in_sync && merges.empty()) {
            table.n_vocab = vocab.size();
            if (s.size() == 1) table.byte_id[(unsigned char)s[0]] = id;
        }
        return id;
    }

    // Merge-Regel (a, b) anhaengen: legt das Token a+b an (oder nimmt die
    // vorhandene ID, falls ein anderer Merge denselben String erzeugt hat) und
    // gibt dessen ID zurueck. So baut auch rebuild_bpe() den Tokenizer auf.
    int add_merge(const std::string& a, const std::string& b) {
        const bool in_sync = table_current();
        const int  x = insert_token(a + b);
        merges.emplace_back(a, b);
        if (in_sync) {
            table.add_rule(token_id(a), token_id(b), x);
            table.n_merges = merges.size();
            table.n_vocab  = vocab.size();
        }
        return x;
    }

    // ID-Tabelle komplett aus vocab/merges neu aufbauen
    void rebuild_table() { table = build_table(); }

    // Text in Token-IDs umwandeln
    std::vector<int> encode(const std::string& text) const {
        if (vocab.empty()) {
            // Fallback: Zeichen-Level
            std::vector<int> ids;
            ids.reserve(text.size());
            for (unsigned char c : text) ids.push_back((int)c);
            return ids;
        }
        if (table_current()) return bpe_detail::encode_ids(table, text);
        return bpe_detail::encode_ids(build_table(), text);
    }

    // Token-IDs zurueck in Text umwandeln
    std::string decode(const std::vector<int>& ids) const {
        std::string result;
        for (int id : ids) {
            if (id >= 0 && id < (int)vocab.size())
                result += vocab[(size_t)id];
        }
        return result;
    }

private:
    bool table_current() const {
        return table.n_merges == merges.size() && table.n_vocab == vocab.size();
    }

    int insert_token(const std::string& s) {
        const auto it = vocab_map.find(s);
        if (it != vocab_map.end()) return it->second;
        const int id = (int)vocab.size();
        vocab.push_back(s);
        vocab_map[s] = id;
        return id;
    }

    // Tabelle aus dem endgueltigen Vokabular ableiten. Ein Bestandteil, der erst
    // durch einen spaeteren Merge entsteht, greift trotzdem nicht rueckwirkend
    // (siehe MergeTable::rank_after), das Ergebnis ist also dasselbe wie beim
    // schrittweisen Aufbau ueber add_merge().
    bpe_detail::MergeTable build_table() const {
        bpe_detail::MergeTable t;
        for (int c = 0; c < 256; ++c) t.byte_id[(size_t)c] = token_id(std::string(1, (char)c));
        t.rules.reserve(merges.size());
        for (const auto& m : merges)
            t.add_rule(token_id(m.first), token_id(m.second), token_id(m.first + m.second));
        t.n_merges = merges.size();
        t.n_vocab  = vocab.size();
        return t;
    }
};

// ---- BPE Training ----
// Lernt BPE-Merges aus einem Text (Semantik siehe oben). Das Training hoert auf,
// sobald das Vokabular target_vocab_size erreicht hat oder kein Paar mehr
// mindestens zweimal vorkommt. Es gibt keine feste Obergrenze fuer Merges.
//
// out_tokens (optional): bekommt die kodierte Trainingsfolge, also dasselbe wie
// encode(text), ohne dass noch einmal kodiert werden muss.
// verbose: Fortschritt auf std::cout ausgeben.
inline BPETokenizer train_bpe(const std::string& text,
                               int target_vocab_size = 1000,
                               std::vector<int>* out_tokens = nullptr,
                               bool verbose = true)
{
    using namespace bpe_detail;

    BPETokenizer tok;
    if (verbose) {
        std::cout << "[BPE] Starte Training...\n";
        std::cout << "[BPE] Ziel-Vokabular: " << target_vocab_size << " Tokens\n";
    }

    // 1. Basis-Vokabular: alle 256 moeglichen Bytes (ID = Bytewert)
    for (int i = 0; i < 256; ++i)
        tok.add_token(std::string(1, (char)i));

    if (verbose) std::cout << "[BPE] Basis-Vokabular: 256 Zeichen\n";

    // 2. Text als Folge von Byte-IDs in einer doppelt verketteten Liste
    //    (Position = Index; verschmolzene Positionen werden DEAD)
    const size_t n = text.size();
    check_text_size(n);
    std::vector<int32_t> tokens(n), prv(n), nxt(n);
    for (size_t i = 0; i < n; ++i) {
        tokens[i] = (int32_t)(unsigned char)text[i];
        prv[i]    = (int32_t)i - 1;
        nxt[i]    = (i + 1 < n) ? (int32_t)(i + 1) : -1;
    }

    // 3. Paar -> aktuelle Haeufigkeit (exakt mitgefuehrt) + Positionen, an denen
    //    das Paar entstanden ist. Die Positionsliste darf veraltete Eintraege
    //    enthalten; beim Merge wird jede Position noch einmal geprueft.
    struct PairInfo {
        int32_t               count = 0;
        std::vector<uint32_t> pos;
    };
    std::unordered_map<uint64_t, PairInfo, PairHash> pairs;
    pairs.reserve(1u << 16);
    for (size_t i = 0; i + 1 < n; ++i) {
        PairInfo& pi = pairs[pair_key(tokens[i], tokens[i + 1])];
        pi.count++;
        pi.pos.push_back((uint32_t)i);
    }

    // Max-Heap mit lazy deletion: Eintraege koennen veraltet sein. Es gilt immer:
    // fuer jedes Paar mit Haeufigkeit >= 2 gibt es einen Eintrag, dessen Zahl
    // mindestens so gross ist wie die wahre Haeufigkeit. Sinkt eine Haeufigkeit,
    // wird nichts getan (der Eintrag ist dann zu hoch und wird beim Herausnehmen
    // korrigiert); steigt sie, kommt ein neuer Eintrag hinein.
    struct HeapEntry {
        int32_t  count;
        uint64_t key;
    };
    struct HeapLess {   // oben liegt: groesste Haeufigkeit, bei Gleichstand kleinster Schluessel
        bool operator()(const HeapEntry& x, const HeapEntry& y) const {
            if (x.count != y.count) return x.count < y.count;
            return x.key > y.key;
        }
    };
    std::vector<HeapEntry> heap;
    for (const auto& kv : pairs)
        if (kv.second.count >= 2) heap.push_back({kv.second.count, kv.first});
    std::make_heap(heap.begin(), heap.end(), HeapLess());

    // 4. BPE-Merge-Schritte
    int merges_done = 0;
    std::vector<uint64_t> touched;      // Paare, deren Haeufigkeit im aktuellen Merge gestiegen ist
    std::vector<uint32_t> positions;
    while (tok.vocab_size() < target_vocab_size) {
        // Bestes Paar suchen: veraltete Eintraege verwerfen oder korrigieren
        uint64_t best_key = 0;
        bool     found    = false;
        while (!heap.empty()) {
            std::pop_heap(heap.begin(), heap.end(), HeapLess());
            const HeapEntry e = heap.back();
            heap.pop_back();
            const auto it = pairs.find(e.key);
            const int32_t cur = (it == pairs.end()) ? 0 : it->second.count;
            if (cur < 2) continue;                      // kommt nicht mehr oft genug vor
            if (cur == e.count) { best_key = e.key; found = true; break; }
            if (cur < e.count) {                        // Eintrag zu hoch: mit wahrem Wert zurueck
                heap.push_back({cur, e.key});
                std::push_heap(heap.begin(), heap.end(), HeapLess());
            }
            // cur > e.count: ein neuerer Eintrag mit dem aktuellen Wert liegt im Heap
        }
        if (!found) break;                              // Kein Paar kommt mehr als einmal vor

        const int32_t a = (int32_t)(best_key >> 32);
        const int32_t b = (int32_t)(best_key & 0xFFFFFFFFu);
        // Kopien, weil add_merge das Vokabular vergroessern kann
        const std::string str_a = tok.vocab[(size_t)a];
        const std::string str_b = tok.vocab[(size_t)b];
        const int32_t     x     = tok.add_merge(str_a, str_b);   // evtl. vorhandene ID

        // Positionen des Paars von links nach rechts abarbeiten. Jede Position wird
        // gegen den aktuellen Stand geprueft: bei a == b ist die zweite von zwei
        // ueberlappenden Positionen dann schon verbraucht ("aaa" -> [aa, a]).
        PairInfo& best = pairs[best_key];
        positions.swap(best.pos);
        best.pos.clear();
        std::sort(positions.begin(), positions.end());
        positions.erase(std::unique(positions.begin(), positions.end()), positions.end());

        touched.clear();
        auto dec = [&](uint64_t key) { pairs[key].count--; };
        auto inc = [&](uint64_t key, int32_t at) {
            PairInfo& pi = pairs[key];
            pi.count++;
            pi.pos.push_back((uint32_t)at);
            touched.push_back(key);
        };
        for (uint32_t pos : positions) {
            const int32_t i = (int32_t)pos;
            if (tokens[(size_t)i] != a) continue;
            const int32_t j = nxt[(size_t)i];
            if (j < 0 || tokens[(size_t)j] != b) continue;
            const int32_t p = prv[(size_t)i];
            const int32_t q = nxt[(size_t)j];

            // (tok[p], a) -> (tok[p], x)   und   (b, tok[q]) -> (x, tok[q])
            if (p >= 0) {
                dec(pair_key(tokens[(size_t)p], a));
                inc(pair_key(tokens[(size_t)p], x), p);
            }
            if (q >= 0) {
                dec(pair_key(b, tokens[(size_t)q]));
                inc(pair_key(x, tokens[(size_t)q]), i);
            }
            dec(best_key);

            tokens[(size_t)i] = x;
            tokens[(size_t)j] = DEAD;
            nxt[(size_t)i]    = q;
            if (q >= 0) prv[(size_t)q] = i;
        }

        // Gestiegene Haeufigkeiten (einmal je Paar) in den Heap
        std::sort(touched.begin(), touched.end());
        touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
        for (uint64_t key : touched) {
            const int32_t c = pairs[key].count;
            if (c >= 2) {
                heap.push_back({c, key});
                std::push_heap(heap.begin(), heap.end(), HeapLess());
            }
        }

        merges_done++;
        if (verbose && merges_done % 100 == 0)
            std::cout << "[BPE] " << merges_done << " Merges | Vokabular: "
                      << tok.vocab_size() << "\n";
    }

    if (out_tokens) {
        out_tokens->clear();
        for (size_t i = 0; i < n; ++i)
            if (tokens[i] != DEAD) out_tokens->push_back((int)tokens[i]);
    }

    if (verbose) {
        std::cout << "[BPE] Fertig! " << merges_done << " Merges\n";
        if (tok.vocab_size() < target_vocab_size)
            std::cout << "[BPE] Hinweis: Ziel " << target_vocab_size
                      << " nicht erreicht, kein Paar kommt mehr als einmal vor\n";
        std::cout << "[BPE] Endgroesse Vokabular: " << tok.vocab_size() << " Tokens\n";
    }
    return tok;
}

// ---- Einfacher Zeichen-Tokenizer (fuer schnelle Tests) ----
inline std::vector<int> tokenize_chars(const std::string& text) {
    std::vector<int> tokens;
    tokens.reserve(text.size());
    for (unsigned char c : text)
        tokens.push_back((int)c);
    return tokens;
}

// ---- Haupt-Tokenisierungs-Funktion ----
// Laedt Datei, erkennt Format, tokenisiert
struct TokenizerResult {
    std::vector<int>  tokens;
    BPETokenizer      bpe;       // nur wenn use_bpe=true
    int               vocab_size;
    std::string       text;      // extrahierter Rohtext
    size_t            file_size;
    std::string       format;
};

inline TokenizerResult skull_tokenize_file(
    const std::string& path,
    bool use_bpe = false,
    int bpe_vocab_size = 1000)
{
    TokenizerResult result;

    // Format erkennen und anzeigen
    FileFormat fmt = detect_format(path);
    switch (fmt) {
        case FileFormat::TXT:   result.format = "TXT";   break;
        case FileFormat::MD:    result.format = "MD";    break;
        case FileFormat::JSON:  result.format = "JSON";  break;
        case FileFormat::JSONL: result.format = "JSONL"; break;
        case FileFormat::CSV:   result.format = "CSV";   break;
        default:                result.format = "TXT";   break;
    }

    std::cout << "[Tokenizer] Format erkannt: " << result.format << "\n";

    // Text laden und extrahieren
    result.text = load_and_extract(path);
    result.file_size = result.text.size();

    if (result.text.empty())
        throw std::runtime_error(
            "Kein Text extrahiert aus '" + path +
            "' (leere Datei oder Format passt nicht zur Endung)");

    std::cout << "[Tokenizer] Text extrahiert: "
              << result.text.size() << " Zeichen\n";

    if (use_bpe) {
        // BPE-Tokenisierung (das Training liefert die kodierte Folge gleich mit)
        result.bpe = train_bpe(result.text, bpe_vocab_size, &result.tokens);
        result.vocab_size = result.bpe.vocab_size();
        std::cout << "[Tokenizer] BPE-Tokens: " << result.tokens.size() << "\n";
        std::cout << "[Tokenizer] Kompressionsrate: "
                  << (double)result.text.size() / result.tokens.size()
                  << "x\n";
    } else {
        // Zeichen-Level Tokenisierung (schnell, einfach)
        result.tokens = tokenize_chars(result.text);
        result.vocab_size = 256;
        std::cout << "[Tokenizer] Char-Tokens: " << result.tokens.size() << "\n";
    }

    return result;
}
