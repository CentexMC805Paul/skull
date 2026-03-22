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
    if (path.size() < 4) return FileFormat::TXT;
    std::string ext = path.substr(path.find_last_of('.'));
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
    if (!file.is_open()) {
        std::cerr << "[FEHLER] Datei nicht gefunden: " << path << "\n";
        return "";
    }
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
//  1. Starte mit Zeichen-Level-Tokens (jedes Byte = 1 Token)
//  2. Zaehle alle Paare von benachbarten Tokens
//  3. Fasse das haeufigste Paar zu einem neuen Token zusammen
//  4. Wiederhole bis Vokabular-Groesse erreicht
//
//  Ergebnis: Haeufige Woerter/Silben werden zu eigenen Tokens.
//  Vokabular-Groesse ist kontrollierbar.
// ============================================================

struct BPETokenizer {
    // Vokabular: Token-ID -> String
    std::vector<std::string> vocab;
    // Umgekehrt: String -> Token-ID
    std::unordered_map<std::string, int> vocab_map;
    // BPE-Merge-Regeln: (a, b) -> merged
    std::vector<std::pair<std::string, std::string>> merges;

    int vocab_size() const { return (int)vocab.size(); }

    // Token-ID fuer einen String (oder -1 wenn nicht gefunden)
    int token_id(const std::string& s) const {
        auto it = vocab_map.find(s);
        return (it != vocab_map.end()) ? it->second : -1;
    }

    // Neues Token hinzufuegen
    int add_token(const std::string& s) {
        auto it = vocab_map.find(s);
        if (it != vocab_map.end()) return it->second;
        int id = (int)vocab.size();
        vocab.push_back(s);
        vocab_map[s] = id;
        return id;
    }

    // Text in Token-IDs umwandeln
    std::vector<int> encode(const std::string& text) const {
        if (vocab.empty()) {
            // Fallback: Zeichen-Level
            std::vector<int> ids;
            for (unsigned char c : text) ids.push_back((int)c);
            return ids;
        }

        // Starte mit Zeichen-Level
        std::vector<std::string> tokens;
        for (unsigned char c : text)
            tokens.push_back(std::string(1, (char)c));

        // BPE-Merges anwenden
        for (const auto& [a, b] : merges) {
            std::string merged = a + b;
            std::vector<std::string> new_tokens;
            size_t i = 0;
            while (i < tokens.size()) {
                if (i + 1 < tokens.size() && tokens[i] == a && tokens[i+1] == b) {
                    new_tokens.push_back(merged);
                    i += 2;
                } else {
                    new_tokens.push_back(tokens[i]);
                    i++;
                }
            }
            tokens = new_tokens;
        }

        // Tokens in IDs umwandeln
        std::vector<int> ids;
        for (const auto& t : tokens) {
            auto it = vocab_map.find(t);
            if (it != vocab_map.end()) {
                ids.push_back(it->second);
            } else {
                // Unbekanntes Token -> Zeichen-Level Fallback
                for (unsigned char c : t)
                    ids.push_back(token_id(std::string(1,(char)c)));
            }
        }
        return ids;
    }

    // Token-IDs zurueck in Text umwandeln
    std::string decode(const std::vector<int>& ids) const {
        std::string result;
        for (int id : ids) {
            if (id >= 0 && id < (int)vocab.size())
                result += vocab[id];
        }
        return result;
    }
};

// ---- BPE Training ----
// Lernt BPE-Merges aus einem Text
inline BPETokenizer train_bpe(const std::string& text,
                               int target_vocab_size = 1000,
                               int max_merges = 500)
{
    BPETokenizer tok;
    std::cout << "[BPE] Starte Training...\n";
    std::cout << "[BPE] Ziel-Vokabular: " << target_vocab_size << " Tokens\n";

    // 1. Basis-Vokabular: alle 256 moeglichen Bytes
    for (int i = 0; i < 256; ++i)
        tok.add_token(std::string(1, (char)i));

    std::cout << "[BPE] Basis-Vokabular: 256 Zeichen\n";

    // 2. Text in Zeichen-Tokens aufteilen
    std::vector<std::string> tokens;
    tokens.reserve(text.size());
    for (unsigned char c : text)
        tokens.push_back(std::string(1, (char)c));

    // 3. BPE-Merge-Schritte
    int merges_done = 0;
    while (tok.vocab_size() < target_vocab_size && merges_done < max_merges) {
        // Haeufigkeiten aller Paare zaehlen
        std::map<std::pair<std::string,std::string>, int> pair_freq;
        for (size_t i = 0; i + 1 < tokens.size(); ++i)
            pair_freq[{tokens[i], tokens[i+1]}]++;

        if (pair_freq.empty()) break;

        // Haeufigste Pair finden
        auto best = std::max_element(pair_freq.begin(), pair_freq.end(),
            [](const auto& a, const auto& b){ return a.second < b.second; });

        if (best->second < 2) break; // Kein Paar kommt mehr als einmal vor

        const auto& [a, b] = best->first;
        std::string merged = a + b;

        // Neues Token hinzufuegen
        tok.add_token(merged);
        tok.merges.push_back({a, b});

        // Tokens mergen
        std::vector<std::string> new_tokens;
        size_t i = 0;
        while (i < tokens.size()) {
            if (i+1 < tokens.size() && tokens[i]==a && tokens[i+1]==b) {
                new_tokens.push_back(merged);
                i += 2;
            } else {
                new_tokens.push_back(tokens[i]);
                i++;
            }
        }
        tokens = new_tokens;
        merges_done++;

        if (merges_done % 100 == 0)
            std::cout << "[BPE] " << merges_done << " Merges | Vokabular: "
                      << tok.vocab_size() << "\n";
    }

    std::cout << "[BPE] Fertig! " << merges_done << " Merges\n";
    std::cout << "[BPE] Endgroesse Vokabular: " << tok.vocab_size() << " Tokens\n";
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

    if (result.text.empty()) {
        std::cerr << "[FEHLER] Kein Text extrahiert!\n";
        return result;
    }

    std::cout << "[Tokenizer] Text extrahiert: "
              << result.text.size() << " Zeichen\n";

    if (use_bpe) {
        // BPE-Tokenisierung
        result.bpe = train_bpe(result.text, bpe_vocab_size);
        result.tokens = result.bpe.encode(result.text);
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
