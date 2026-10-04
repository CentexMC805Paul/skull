// ============================================================
//  Test fuer src/tokenizer.h (BPE-Tokenizer)
//
//  Die schnelle Umsetzung (inkrementelle Paarzaehlung mit Heap, verkettete
//  Liste, Encode mit Min-Heap) wird gegen absichtlich naive, langsame
//  Referenzen mit der festgelegten Semantik geprueft:
//
//   Training: Haeufigkeit = Anzahl der Positionen i mit (t[i], t[i+1]) == (a, b)
//     in der aktuellen Folge (Ueberlappungen zaehlen), gewaehlt wird die
//     hoechste Haeufigkeit, bei Gleichstand das kleinste (id_a, id_b); Ende bei
//     Haeufigkeit < 2 oder Vokabular >= Ziel; ein Merge ersetzt von links nach
//     rechts ohne Ueberlappung ("aaa" -> [aa, a]); neues Token = add_token(a+b).
//   Encode: alle Merges nacheinander anwenden, jeder von links nach rechts.
//
//  Geprueft wird mit tausenden zufaelliger kleiner Korpora (feste Seeds, kleine
//  Alphabete, stark repetitive Texte), zufaelligen Merge-Listen mit doppelten
//  Strings/Paaren, dem frueheren 500-Merge-Limit, der Kompatibilitaet zu
//  rebuild_bpe() und zum TOKB-Dateiformat, und es wird die Geschwindigkeit
//  gemessen.
//
//  Exit-Code 0 = alles in Ordnung.
// ============================================================
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "rng.h"
#include "weights.h"

// Unter AddressSanitizer/UBSan laeuft alles um ein Vielfaches langsamer:
// dort wird nur der grosse Leistungstest verkleinert.
#if defined(__SANITIZE_ADDRESS__)
#  define BPECHECK_SANITIZED 1
#elif defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define BPECHECK_SANITIZED 1
#  endif
#endif
#ifndef BPECHECK_SANITIZED
#  define BPECHECK_SANITIZED 0
#endif

static int failures = 0;

static void expect(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "ok" : "FEHLER", what.c_str());
    std::fflush(stdout);
    if (!ok) ++failures;
}

typedef std::vector<std::pair<std::string, std::string> > Merges;

// ------------------------------------------------------------
//  Naive Referenz
// ------------------------------------------------------------

// Vokabular wie in rebuild_bpe(): 256 Byte-Tokens, danach je Merge a+b (falls neu)
struct RefVocab {
    std::vector<std::string>   vocab;
    std::map<std::string, int> id;
    RefVocab() { for (int i = 0; i < 256; ++i) add(std::string(1, (char)i)); }
    int add(const std::string& s) {
        std::map<std::string, int>::const_iterator it = id.find(s);
        if (it != id.end()) return it->second;
        const int k = (int)vocab.size();
        vocab.push_back(s);
        id[s] = k;
        return k;
    }
    int find(const std::string& s) const {
        std::map<std::string, int>::const_iterator it = id.find(s);
        return it == id.end() ? -1 : it->second;
    }
};

struct RefTrained {
    Merges                   merges;
    std::vector<std::string> vocab;
    std::vector<int>         ids;
};

// Training nach der Beschreibung oben: jede Runde komplett neu zaehlen
static RefTrained ref_train(const std::string& text, int target) {
    RefVocab v;
    RefTrained res;
    std::vector<int> t;
    for (size_t i = 0; i < text.size(); ++i) t.push_back((int)(unsigned char)text[i]);

    while ((int)v.vocab.size() < target) {
        std::map<std::pair<int, int>, int> freq;
        for (size_t i = 0; i + 1 < t.size(); ++i) freq[std::make_pair(t[i], t[i + 1])]++;

        // hoechste Haeufigkeit; die Map laeuft nach (a, b) aufsteigend, ">" behaelt bei
        // Gleichstand also das lexikographisch kleinste Paar
        bool have = false;
        std::pair<int, int> best(0, 0);
        int best_count = 0;
        for (std::map<std::pair<int, int>, int>::const_iterator it = freq.begin(); it != freq.end(); ++it) {
            if (it->second > best_count) { best_count = it->second; best = it->first; have = true; }
        }
        if (!have || best_count < 2) break;

        const std::string sa = v.vocab[(size_t)best.first];
        const std::string sb = v.vocab[(size_t)best.second];
        const int x = v.add(sa + sb);
        res.merges.push_back(std::make_pair(sa, sb));

        std::vector<int> nt;
        size_t i = 0;
        while (i < t.size()) {
            if (i + 1 < t.size() && t[i] == best.first && t[i + 1] == best.second) {
                nt.push_back(x);
                i += 2;
            } else {
                nt.push_back(t[i]);
                i++;
            }
        }
        t.swap(nt);
    }
    res.vocab = v.vocab;
    res.ids   = t;
    return res;
}

// Encode-Referenz auf Strings (wie die fruehere Umsetzung): alle Merges nacheinander
struct RefEncoder {
    Merges   merges;
    RefVocab v;
    explicit RefEncoder(const Merges& m) : merges(m) {
        for (size_t k = 0; k < m.size(); ++k) v.add(m[k].first + m[k].second);
    }
    std::vector<int> encode(const std::string& text) const {
        std::vector<std::string> toks;
        for (size_t i = 0; i < text.size(); ++i) toks.push_back(std::string(1, text[i]));
        for (size_t k = 0; k < merges.size(); ++k) {
            const std::string& a = merges[k].first;
            const std::string& b = merges[k].second;
            const std::string merged = a + b;
            std::vector<std::string> nt;
            size_t i = 0;
            while (i < toks.size()) {
                if (i + 1 < toks.size() && toks[i] == a && toks[i + 1] == b) {
                    nt.push_back(merged);
                    i += 2;
                } else {
                    nt.push_back(toks[i]);
                    i++;
                }
            }
            toks.swap(nt);
        }
        std::vector<int> ids;
        for (size_t i = 0; i < toks.size(); ++i) ids.push_back(v.find(toks[i]));
        return ids;
    }
};

// Dasselbe auf IDs (fuer lange Texte, etwas schneller); setzt voraus, dass jeder
// Bestandteil eines Merges schon im Vokabular steht (gilt fuer trainierte Merges)
static std::vector<int> ref_encode_ids(const Merges& merges, const std::string& text) {
    RefVocab v;
    std::vector<int> t;
    for (size_t i = 0; i < text.size(); ++i) t.push_back((int)(unsigned char)text[i]);
    for (size_t k = 0; k < merges.size(); ++k) {
        const int a = v.find(merges[k].first);
        const int b = v.find(merges[k].second);
        const int x = v.add(merges[k].first + merges[k].second);
        std::vector<int> nt;
        nt.reserve(t.size());
        size_t i = 0;
        while (i < t.size()) {
            if (i + 1 < t.size() && t[i] == a && t[i + 1] == b) { nt.push_back(x); i += 2; }
            else                                                  { nt.push_back(t[i]); i++; }
        }
        t.swap(nt);
    }
    return t;
}

// ------------------------------------------------------------
//  Zufallstexte
// ------------------------------------------------------------

static size_t rnd(std::mt19937& g, size_t n) { return rng_below(g, n); }

// Text der Laenge len ueber dem Alphabet; mehrere Stile, darunter stark repetitive
static std::string gen_text(std::mt19937& g, const std::string& alphabet, size_t len) {
    std::string s;
    const size_t k = alphabet.size();
    switch (rnd(g, 6)) {
        case 0:   // gleichverteilt
            while (s.size() < len) s += alphabet[rnd(g, k)];
            break;
        case 1: { // kurzes Muster, wiederholt, gelegentlich gestoert ("abababab")
            std::string pat;
            const size_t pl = 1 + rnd(g, 4);
            for (size_t i = 0; i < pl; ++i) pat += alphabet[rnd(g, k)];
            while (s.size() < len) {
                s += pat;
                if (rnd(g, 30) == 0) s += alphabet[rnd(g, k)];
            }
            break;
        }
        case 2:   // Laeufe gleicher Zeichen ("aaabbbbaa")
            while (s.size() < len) s.append(1 + rnd(g, 9), alphabet[rnd(g, k)]);
            break;
        case 3:   // ein Zeichen dominiert
            while (s.size() < len) s += (rnd(g, 10) < 7) ? alphabet[0] : alphabet[rnd(g, k)];
            break;
        case 4:   // nur ein Zeichen ("aaaa...")
            s.assign(len, alphabet[rnd(g, k)]);
            break;
        default: { // Muster aus zwei Teilen, abwechselnd lang und kurz ("aabaab")
            std::string p1(1 + rnd(g, 3), alphabet[rnd(g, k)]);
            std::string p2(1 + rnd(g, 2), alphabet[rnd(g, k)]);
            while (s.size() < len) { s += p1; s += p2; }
            break;
        }
    }
    s.resize(len);
    return s;
}

static size_t gen_len(std::mt19937& g, size_t max_len) {
    return (rnd(g, 10) == 0) ? rnd(g, 6) : rnd(g, max_len + 1);
}

// 2..6 verschiedene Zeichen; meist Buchstaben, manchmal beliebige Bytes (auch >= 128 und '\0')
static std::string gen_alphabet(std::mt19937& g) {
    const size_t k = 2 + rnd(g, 5);
    std::string a;
    if (rnd(g, 4) != 0) {
        std::string pool = "abcdef";
        for (size_t i = 0; i < k; ++i) {
            const size_t j = i + rnd(g, pool.size() - i);
            std::swap(pool[i], pool[j]);
            a += pool[i];
        }
    } else {
        while (a.size() < k) {
            const char c = (char)rnd(g, 256);
            if (a.find(c) == std::string::npos) a += c;
        }
    }
    return a;
}

static std::string show(const std::string& s) {
    std::string r;
    for (size_t i = 0; i < s.size() && i < 60; ++i) {
        const unsigned char c = (unsigned char)s[i];
        if (c >= 32 && c < 127) r += (char)c;
        else { char buf[8]; std::snprintf(buf, sizeof buf, "\\x%02x", c); r += buf; }
    }
    if (s.size() > 60) r += "...";
    return r;
}

// Fester Wortschatz aus Silben, danach Woerter-Folgen (haeufige Woerter oefter)
static std::vector<std::string> make_words(std::mt19937& g, size_t n_words) {
    static const char* const syl[] = {"ka", "lo", "mi", "ne", "ru", "sta", "ber", "gen", "li", "ton",
                                      "ar", "el", "sch", "ung", "ei", "da", "ho", "fe", "zu", "wi"};
    std::vector<std::string> words;
    for (size_t i = 0; i < n_words; ++i) {
        std::string w;
        const size_t ns = 2 + rnd(g, 3);
        for (size_t j = 0; j < ns; ++j) w += syl[rnd(g, 20)];
        words.push_back(w);
    }
    return words;
}

static std::string make_text(std::mt19937& g, const std::vector<std::string>& words, size_t n_chars) {
    std::string s;
    while (s.size() < n_chars) {
        const double u = rng_uniform01(g);
        s += words[(size_t)(u * u * (double)words.size())];
        const size_t sep = rnd(g, 20);
        s += (sep == 0) ? ".\n" : (sep == 1) ? ", " : " ";
    }
    s.resize(n_chars);
    return s;
}

static double seconds_since(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// ------------------------------------------------------------
//  Hilfen fuer Fehlerausgabe (nur die ersten Abweichungen zeigen)
// ------------------------------------------------------------
static int shown = 0;
static void note(const std::string& s) {
    if (shown < 8) { std::printf("      %s\n", s.c_str()); ++shown; }
}

// ============================================================
int main() {
    // ---- 1) Feste Spezialfaelle ----
    std::printf("Spezialfaelle\n");
    {
        std::vector<int> ids;
        BPETokenizer t = train_bpe("aaaa", 300, &ids, false);
        expect(t.merges.size() == 1 && t.merges[0].first == "a" && t.merges[0].second == "a",
               "\"aaaa\": ein Merge (a,a)");
        expect(ids == std::vector<int>({256, 256}), "\"aaaa\" -> [aa, aa]");
        expect(t.encode("aaa") == std::vector<int>({256, 97}), "encode(\"aaa\") = [aa, a] (nicht ueberlappend)");
        expect(t.encode("aaaaa") == std::vector<int>({256, 256, 97}), "encode(\"aaaaa\") = [aa, aa, a]");
        expect(t.encode("") .empty(), "encode(\"\") ist leer");
        expect(t.encode("a") == std::vector<int>({97}), "encode(\"a\") = [a]");
        expect(t.encode("b") == std::vector<int>({98}), "encode(\"b\") = [b]");
        expect(t.decode(t.encode("aaaaaab")) == "aaaaaab", "decode(encode(x)) == x");
    }
    {
        // Leerer Text, ein Byte, Ziel kleiner als die Basis, kein Paar >= 2
        std::vector<int> ids;
        BPETokenizer t = train_bpe("", 1000, &ids, false);
        expect(t.vocab_size() == 256 && t.merges.empty() && ids.empty(), "Training mit leerem Text");
        t = train_bpe("x", 1000, &ids, false);
        expect(t.vocab_size() == 256 && ids == std::vector<int>({120}), "Training mit einem Byte");
        t = train_bpe("abababab", 100, &ids, false);
        expect(t.vocab_size() == 256 && t.merges.empty() && ids.size() == 8, "Ziel < 256: keine Merges");
        t = train_bpe("abcdefg", 1000, &ids, false);
        expect(t.vocab_size() == 256 && ids.size() == 7, "kein Paar kommt 2x vor: keine Merges");
        BPETokenizer none;   // nie trainiert
        expect(none.encode("hi") == std::vector<int>({104, 105}), "leeres Vokabular: Byte-Ebene wie frueher");
        expect(none.encode("").empty(), "leeres Vokabular, leerer Text");
    }
    {
        // Doppelte Strings: ("a","bc") und ("ab","c") ergeben beide "abc" (dieselbe ID)
        Merges m;
        m.push_back(std::make_pair(std::string("a"), std::string("b")));
        m.push_back(std::make_pair(std::string("b"), std::string("c")));
        m.push_back(std::make_pair(std::string("a"), std::string("bc")));
        m.push_back(std::make_pair(std::string("ab"), std::string("c")));
        BPETokenizer t = rebuild_bpe(m);
        expect(t.vocab_size() == 256 + 4 - 1 && t.token_id("abc") == 258, "doppelter String teilt sich eine ID");
        RefEncoder r(m);
        const char* const probe[] = {"abc", "abcabc", "ababc", "bcabc", "aabcc", "cba", ""};
        bool same = true;
        for (size_t i = 0; i < sizeof probe / sizeof probe[0]; ++i)
            same = same && (t.encode(probe[i]) == r.encode(probe[i]));
        expect(same, "encode mit doppeltem Ergebnis-String == Referenz");
        expect(t.encode("abc") == std::vector<int>({(int)t.token_id("abc")}), "\"abc\" -> ein Token");
    }

    // ---- 2) Property-Test: Training + Encode gegen die Referenz ----
    std::printf("Property-Test (Training + Encode, zufaellige kleine Korpora)\n");
    {
        std::mt19937 g(20240607);
        const int targets[] = {256, 257, 258, 259, 262, 270, 300, 400, 1000};
        const int N = 3500;
        int bad_train = 0, bad_enc = 0, bad_rt = 0, bad_rebuild = 0;
        int with_same_pair = 0, ended_early = 0, dup_string = 0, long_runs = 0, total_merges = 0;
        shown = 0;
        for (int it = 0; it < N; ++it) {
            const std::string alpha = gen_alphabet(g);
            const std::string text  = gen_text(g, alpha, gen_len(g, 300));
            const int target = targets[rnd(g, sizeof targets / sizeof targets[0])];

            std::vector<int> ids;
            BPETokenizer tok = train_bpe(text, target, &ids, false);
            const RefTrained ref = ref_train(text, target);

            if (!(tok.merges == ref.merges && tok.vocab == ref.vocab && ids == ref.ids)) {
                ++bad_train;
                note("Training weicht ab: alphabet=" + show(alpha) + " ziel=" + std::to_string(target) +
                     " text=" + show(text));
            }
            total_merges += (int)ref.merges.size();
            for (size_t k = 0; k < ref.merges.size(); ++k)
                if (ref.merges[k].first == ref.merges[k].second) { ++with_same_pair; break; }
            if (ref.vocab.size() < (size_t)target && target > 256) ++ended_early;
            if (ref.vocab.size() < 256 + ref.merges.size()) ++dup_string;
            if (text.size() >= 8 && text.find(std::string(8, text[0])) != std::string::npos) ++long_runs;

            // Encode: Trainingstext und fremde Texte (anderes Alphabet-Ende, andere Laenge)
            const RefEncoder renc(ref.merges);
            bool enc_ok = (tok.encode(text) == ref.ids) && (renc.encode(text) == ref.ids);
            std::string alpha2 = alpha;
            alpha2 += (char)(rnd(g, 3) == 0 ? rnd(g, 256) : 'x');   // ein Zeichen, das beim Training fehlt
            bool rt_ok = (tok.decode(ids) == text);
            for (int f = 0; f < 2; ++f) {
                const std::string other = gen_text(g, alpha2, gen_len(g, 150));
                const std::vector<int> a = tok.encode(other);
                enc_ok = enc_ok && (a == renc.encode(other));
                rt_ok  = rt_ok && (tok.decode(a) == other);
            }
            if (!enc_ok) { ++bad_enc; note("encode weicht ab: alphabet=" + show(alpha) + " text=" + show(text)); }
            if (!rt_ok)  { ++bad_rt;  note("decode(encode(x)) != x: alphabet=" + show(alpha)); }

            // Kompatibilitaet: aus den gespeicherten String-Merges neu aufgebaut
            const BPETokenizer re = rebuild_bpe(tok.merges);
            if (!(re.vocab == tok.vocab && re.merges == tok.merges && re.encode(text) == ids &&
                  re.encode("") == tok.encode("")))
                ++bad_rebuild;
        }
        expect(bad_train == 0, std::to_string(N) + " Korpora: Merge-Folge, Vokabular und kodierte Folge == Referenz");
        expect(bad_enc == 0,   "encode (Trainingstext und fremde Texte) == Referenz");
        expect(bad_rt == 0,    "decode(encode(x)) == x");
        expect(bad_rebuild == 0, "rebuild_bpe(merges) == Original");
        std::printf("      Abdeckung: %d Merges gesamt, %d Korpora mit (a,a)-Merge, %d mit langen Laeufen, "
                    "%d vorzeitig beendet (kein Paar >= 2), %d mit doppeltem Ergebnis-String\n",
                    total_merges, with_same_pair, long_runs, ended_early, dup_string);
        expect(with_same_pair > 100 && ended_early > 100 && long_runs > 100,
               "die Zufallskorpora decken (a,a)-Merges, Laeufe und vorzeitiges Ende ab");
    }

    // ---- 3) Zufaellige Merge-Listen (auch doppelte Strings/Paare, tote Regeln) nur fuer encode ----
    std::printf("Property-Test (encode mit beliebigen Merge-Listen)\n");
    {
        std::mt19937 g(777);
        const int N = 4000;
        int bad = 0, bad_legacy = 0, lists_dup_string = 0, lists_dup_pair = 0, lists_dead = 0;
        shown = 0;
        for (int it = 0; it < N; ++it) {
            const std::string alpha = gen_alphabet(g).substr(0, 3);   // klein: viele Zusammentreffen
            std::vector<std::string> pool;
            for (size_t i = 0; i < alpha.size(); ++i) pool.push_back(std::string(1, alpha[i]));
            Merges m;
            const size_t count = 1 + rnd(g, 40);
            for (size_t k = 0; k < count; ++k) {
                const size_t r = rnd(g, 100);
                if (r < 8 && !m.empty()) {
                    m.push_back(m[rnd(g, m.size())]);                 // dasselbe Paar noch einmal
                } else if (r < 14) {
                    const char* const junk[] = {"zz", "", "q"};      // Bestandteil, der nie als Token vorkommt
                    const std::string a = (rnd(g, 2) == 0) ? std::string(junk[rnd(g, 3)]) : pool[rnd(g, pool.size())];
                    const std::string b = (rnd(g, 2) == 0) ? std::string(junk[rnd(g, 3)]) : pool[rnd(g, pool.size())];
                    m.push_back(std::make_pair(a, b));
                } else {
                    m.push_back(std::make_pair(pool[rnd(g, pool.size())], pool[rnd(g, pool.size())]));
                }
                pool.push_back(m.back().first + m.back().second);
            }
            // Reihenfolge teilweise durcheinander: Merges, die Tokens aus spaeteren Merges brauchen
            if (rnd(g, 3) == 0) {
                const size_t swaps = 1 + rnd(g, 4);
                for (size_t s = 0; s < swaps; ++s) std::swap(m[rnd(g, m.size())], m[rnd(g, m.size())]);
            }

            // Statistik zur Abdeckung
            {
                std::map<std::string, std::vector<std::pair<std::string, std::string> > > by_str;
                RefVocab known;
                bool dead = false, dup_pair = false;
                for (size_t k = 0; k < m.size(); ++k) {
                    if (known.find(m[k].first) < 0 || known.find(m[k].second) < 0) dead = true;
                    known.add(m[k].first + m[k].second);
                    std::vector<std::pair<std::string, std::string> >& vec = by_str[m[k].first + m[k].second];
                    for (size_t q = 0; q < vec.size(); ++q) if (vec[q] == m[k]) dup_pair = true;
                    if (std::find(vec.begin(), vec.end(), m[k]) == vec.end()) vec.push_back(m[k]);
                }
                bool dup_str = false;
                for (std::map<std::string, std::vector<std::pair<std::string, std::string> > >::const_iterator
                         p = by_str.begin(); p != by_str.end(); ++p)
                    if (p->second.size() > 1) dup_str = true;
                lists_dead += dead; lists_dup_pair += dup_pair; lists_dup_string += dup_str;
            }

            const BPETokenizer tok = rebuild_bpe(m);           // Weg beim Laden der Gewichte
            BPETokenizer legacy;                               // alter Weg: merges direkt befuellt
            for (int c = 0; c < 256; ++c) legacy.add_token(std::string(1, (char)c));
            for (size_t k = 0; k < m.size(); ++k) { legacy.add_token(m[k].first + m[k].second); legacy.merges.push_back(m[k]); }
            const RefEncoder renc(m);

            std::string alpha2 = alpha;
            if (rnd(g, 2) == 0) alpha2 += 'z';
            for (int f = 0; f < 4; ++f) {
                const std::string text = gen_text(g, alpha2, gen_len(g, 80));
                const std::vector<int> want = renc.encode(text);
                if (tok.encode(text) != want) {
                    ++bad;
                    note("Merge-Liste: encode weicht ab, text=" + show(text) + " merges=" + std::to_string(m.size()));
                }
                if (legacy.encode(text) != want) ++bad_legacy;
                if (tok.decode(tok.encode(text)) != text) ++bad;
            }
        }
        expect(bad == 0, std::to_string(N) + " zufaellige Merge-Listen: encode == nacheinander angewendete Merges");
        expect(bad_legacy == 0, "Tokenizer mit direkt befuellten merges (alte Bauweise) kodiert ebenfalls richtig");
        std::printf("      Abdeckung: %d Listen mit doppeltem Ergebnis-String, %d mit wiederholtem Paar, "
                    "%d mit Regeln, deren Bestandteile erst spaeter entstehen\n",
                    lists_dup_string, lists_dup_pair, lists_dead);
        expect(lists_dup_string > 200 && lists_dup_pair > 200 && lists_dead > 200,
               "die Merge-Listen decken doppelte Strings, doppelte Paare und tote Regeln ab");
    }

    // ---- 4) Kein 500er-Limit; mittelgrosse Korpora gegen die Referenz ----
    std::printf("Kein Merge-Limit\n");
    {
        std::mt19937 g(4242);
        const std::vector<std::string> words = make_words(g, 400);
        const std::string text = make_text(g, words, 60000);
        std::vector<int> ids;
        BPETokenizer tok = train_bpe(text, 1000, &ids, false);
        std::printf("      60000 Zeichen, Ziel 1000: %d Tokens, %zu Merges\n", tok.vocab_size(), tok.merges.size());
        expect(tok.vocab_size() > 756, "Ziel 1000 ergibt mehr als die frueheren 756 Tokens");
        expect(tok.vocab_size() == 1000, "Ziel 1000 wird erreicht");
        expect((size_t)(tok.vocab_size() - 256) <= tok.merges.size(), "Anzahl Merges passt zum Vokabular");
        expect(tok.encode(text) == ids, "encode(Trainingstext) == bei Training erzeugte Folge");
        expect(tok.decode(ids) == text, "decode(Trainingsfolge) == Trainingstext");
        expect(ids.size() < text.size() / 2, "Text wird deutlich verkuerzt");

        // fremder Text aus demselben Wortschatz, gegen die langsame Referenz
        const std::string other = make_text(g, words, 20000);
        expect(tok.encode(other) == ref_encode_ids(tok.merges, other), "fremder Text (20000 Zeichen) == Referenz");

        // mittelgrosses Training komplett gegen die Referenz
        const std::string mid = make_text(g, words, 4000);
        std::vector<int> mid_ids;
        BPETokenizer mid_tok = train_bpe(mid, 1000, &mid_ids, false);
        const RefTrained mid_ref = ref_train(mid, 1000);
        expect(mid_tok.merges == mid_ref.merges && mid_tok.vocab == mid_ref.vocab && mid_ids == mid_ref.ids,
               "4000 Zeichen, Ziel 1000: Training == Referenz (" + std::to_string(mid_ref.merges.size()) + " Merges)");

        // sehr repetitiv
        std::string rep;
        for (int i = 0; i < 3000; ++i) rep += "ab";
        rep += std::string(500, 'c');
        std::vector<int> rep_ids;
        BPETokenizer rep_tok = train_bpe(rep, 1000, &rep_ids, false);
        const RefTrained rep_ref = ref_train(rep, 1000);
        expect(rep_tok.merges == rep_ref.merges && rep_ids == rep_ref.ids,
               "stark repetitiver Text (abab..., cccc...): Training == Referenz");
    }

    // ---- 5) Rueckwaertskompatibilitaet: rebuild_bpe und das TOKB-Dateiformat ----
    std::printf("Rueckwaertskompatibilitaet\n");
    {
        std::mt19937 g(99);
        const std::vector<std::string> words = make_words(g, 60);
        const std::string text = make_text(g, words, 8000);
        std::vector<int> ids;
        BPETokenizer orig = train_bpe(text, 400, &ids, false);
        const BPETokenizer re = rebuild_bpe(orig.merges);
        const std::string other = make_text(g, words, 3000) + " ueberraschung: \xC3\xA4\xC3\xB6\xC3\xBC\n";
        expect(re.vocab == orig.vocab && re.merges == orig.merges, "rebuild_bpe: gleiches Vokabular und gleiche Merges");
        expect(re.encode(text) == orig.encode(text) && re.encode(text) == ids,
               "rebuild_bpe(merges) kodiert den Trainingstext identisch");
        expect(re.encode(other) == orig.encode(other), "rebuild_bpe(merges) kodiert fremden Text identisch (auch UTF-8-Bytes)");
        expect(re.decode(re.encode(other)) == other, "decode(encode(x)) == x fuer UTF-8-Bytes");

        // Kopie/Zuweisung (der Trainer kopiert den Tokenizer in die Gewichte)
        SkullWeights w;
        w.has_bpe = true;
        w.bpe = orig;
        BPETokenizer copy = w.bpe;
        expect(copy.encode(text) == ids, "kopierter Tokenizer kodiert gleich");

        // Dateiformat: von Hand nach der dokumentierten Struktur geschrieben, mit load_weights gelesen
        const std::string path = "bpecheck_tmp.weights";
        {
            std::ofstream f(path.c_str(), std::ios::binary);
            auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) f.put((char)((v >> (8 * i)) & 0xFF)); };
            auto u64 = [&](uint64_t v) { for (int i = 0; i < 8; ++i) f.put((char)((v >> (8 * i)) & 0xFF)); };
            const uint64_t dim = 2, vocab = (uint64_t)orig.vocab_size();
            f.write("SKULL", 5);
            u64(dim); u64(vocab);
            const double zero = 0.0;
            for (uint64_t i = 0; i < vocab * dim + dim * dim + dim * vocab; ++i)
                f.write(reinterpret_cast<const char*>(&zero), sizeof zero);
            f.write("TOKB", 4);
            u32((uint32_t)orig.merges.size());
            for (size_t k = 0; k < orig.merges.size(); ++k) {
                u32((uint32_t)orig.merges[k].first.size());  f.write(orig.merges[k].first.data(),  (std::streamsize)orig.merges[k].first.size());
                u32((uint32_t)orig.merges[k].second.size()); f.write(orig.merges[k].second.data(), (std::streamsize)orig.merges[k].second.size());
            }
        }
        bool loaded = false, same = false;
        try {
            const SkullWeights lw = load_weights(path);
            loaded = lw.has_bpe && (size_t)lw.vocab == (size_t)orig.vocab_size();
            same = lw.bpe.merges == orig.merges && lw.bpe.vocab == orig.vocab &&
                   lw.bpe.encode(text) == ids && lw.bpe.encode(other) == orig.encode(other);
        } catch (const std::exception& e) {
            note(std::string("load_weights: ") + e.what());
        }
        expect(loaded, "Gewichte-Datei im dokumentierten TOKB-Format laedt");
        expect(same, "geladener Tokenizer kodiert wie das Original");

        // und der Weg ueber save_weights
        bool saved_same = false;
        try {
            SkullWeights sw;
            sw.dim = 2; sw.vocab = (size_t)orig.vocab_size();
            sw.W_embed.assign(sw.vocab * 2, 0.0); sw.W_hidden.assign(4, 0.0); sw.W_out.assign(2 * sw.vocab, 0.0);
            sw.has_bpe = true; sw.bpe = orig;
            save_weights(path, sw);
            const SkullWeights lw = load_weights(path);
            saved_same = lw.has_bpe && lw.bpe.merges == orig.merges && lw.bpe.encode(other) == orig.encode(other);
        } catch (const std::exception& e) {
            note(std::string("save/load_weights: ") + e.what());
        }
        std::remove(path.c_str());
        expect(saved_same, "save_weights -> load_weights: Tokenizer unveraendert");
    }

    // ---- 6) Geschwindigkeit ----
    std::printf("Geschwindigkeit\n");
    {
        const size_t size = BPECHECK_SANITIZED ? 200000u : 1000000u;
        std::mt19937 g(31337);
        const std::vector<std::string> words = make_words(g, 3000);
        const std::string text = make_text(g, words, size);

        std::vector<int> ids;
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        BPETokenizer tok = train_bpe(text, 1000, &ids, false);
        const double t_train = seconds_since(t0);

        t0 = std::chrono::steady_clock::now();
        const std::vector<int> enc = tok.encode(text);
        const double t_enc = seconds_since(t0);

        const BPETokenizer re = rebuild_bpe(tok.merges);
        t0 = std::chrono::steady_clock::now();
        const std::vector<int> enc2 = re.encode(text);
        const double t_enc2 = seconds_since(t0);

        std::printf("      %zu Zeichen, Ziel 1000 -> %d Tokens, %zu Tokens-Folge (%.2fx kuerzer)\n",
                    text.size(), tok.vocab_size(), ids.size(), (double)text.size() / (double)ids.size());
        std::printf("      Zeit: Training %.3f s, encode %.3f s (rebuild_bpe: %.3f s), zusammen %.3f s\n",
                    t_train, t_enc, t_enc2, t_train + t_enc);
        expect(enc == ids, "encode(Text) == bei Training erzeugte Folge");
        expect(enc2 == ids, "rebuild_bpe-Tokenizer kodiert gleich");
        expect(tok.decode(ids) == text, "decode(Folge) == Text");
        expect(BPECHECK_SANITIZED || (t_train + t_enc) < 120.0, "Training + encode unter 120 s (grosszuegige Grenze)");

        // Lange Texte duerfen auch bei einem langen Lauf gleicher Zeichen nicht quadratisch werden
        const std::string run(BPECHECK_SANITIZED ? 100000u : 1000000u, 'a');
        std::vector<int> run_ids;
        t0 = std::chrono::steady_clock::now();
        BPETokenizer run_tok = train_bpe(run, 300, &run_ids, false);
        const std::vector<int> run_enc = run_tok.encode(run);
        const double t_run = seconds_since(t0);
        std::printf("      %zu mal 'a': Training + encode %.3f s (%d Tokens)\n", run.size(), t_run, run_tok.vocab_size());
        expect(run_enc == run_ids && run_tok.decode(run_enc) == run, "langer Lauf gleicher Zeichen: Training und encode stimmen ueberein");
        expect(BPECHECK_SANITIZED || t_run < 120.0, "langer Lauf unter 120 s");
    }

    std::printf(failures == 0 ? "\nBPE-Test bestanden.\n" : "\nBPE-Test: %d Fehler.\n", failures);
    return failures == 0 ? 0 : 1;
}
