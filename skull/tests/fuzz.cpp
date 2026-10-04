// ============================================================
//  Robustheits- und Modelltests fuer Lexer, Parser und Interpreter
//
//  A) Mutations-Fuzzing: Beispiele und Testskripte werden zufaellig veraendert (Bytes loeschen,
//     verdoppeln, Schluesselwoerter einfuegen, Klammern vervielfachen ...). Lexer und Parser
//     duerfen jede Eingabe nur mit std::runtime_error (saubere Fehlermeldung) ablehnen -
//     nie abstuerzen, haengen oder eine andere Ausnahme werfen.
//  B) Generierte Programme: aus einer kleinen Grammatik entstehen zufaellige, meist gueltige
//     Programme (Schleifen, Funktionen, Listen, Text). Der Interpreter muss sie ausfuehren
//     oder sauber ablehnen, und zweimal dasselbe Ergebnis liefern.
//  C) Modelltest: Listen-/Text-Operationen laufen parallel in Skull und in einem einfachen
//     C++-Modell; Ausgabe und Fehlermeldungen (inkl. Zeilennummer) muessen uebereinstimmen.
//  D) Grenzen: tief verschachtelte Listen, Zyklen, zu grosse Listen/Texte, aufwendige Vergleiche.
//
//  Alles ist deterministisch (feste Seeds). Zusammen mit -DSKULL_SANITIZE=ON / -DSKULL_TSAN=ON
//  fangen diese Tests auch Speicherfehler und Datenwettlaeufe.
//  Arbeitsverzeichnis: build/testenv (dort liegen examples/ und cases/).
// ============================================================
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "stack.h"
#include "lexer.h"
#include "ast.h"
#include "parser.h"
#include "interpreter.h"

static int failures = 0;

static void expect(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "ok" : "FEHLER", what.c_str());
    if (!ok) ++failures;
}

// ---- Hilfen ----
struct Rng {
    std::mt19937 g;
    explicit Rng(unsigned seed) : g(seed) {}
    int below(int n) { return (int)(g() % (unsigned)n); }
    bool chance(int percent) { return below(100) < percent; }
    bool chance_per_mille(int pm) { return below(1000) < pm; }
};

// std::cout fuer die Dauer eines Laufs in einen Puffer umleiten (wird auch bei Ausnahmen zurueckgesetzt)
struct CoutCapture {
    std::ostringstream buf;
    std::streambuf* old;
    CoutCapture() : old(std::cout.rdbuf(buf.rdbuf())) {}
    ~CoutCapture() { std::cout.rdbuf(old); }
};

struct Outcome {
    bool ok = true;
    std::string out;
    std::string error;
};

// Fuehrt Quelltext aus. Nur std::runtime_error gilt als "saubere Ablehnung"; alles andere wird weitergereicht.
static Outcome run_script(const std::string& src, size_t max_list = 64, size_t max_string = 4096) {
    Outcome o;
    CoutCapture cap;
    try {
        Lexer lexer(src);
        Parser parser(lexer.tokenize());
        auto program = parser.parse();
        Interpreter interp(max_list, max_string);
        interp.run(program.get());
    } catch (const std::runtime_error& e) {
        o.ok = false;
        o.error = e.what();
    }
    o.out = cap.buf.str();
    return o;
}

static std::string slurp(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Eingabe, die einen Fehler ausgeloest hat, zum Nachstellen speichern
static void save_repro(const std::string& tag, int n, const std::string& src) {
    const std::string name = "fuzz_repro_" + tag + "_" + std::to_string(n) + ".skull";
    std::ofstream(name, std::ios::binary) << src;
    std::printf("    Eingabe gespeichert: %s (%zu Byte)\n", name.c_str(), src.size());
}

// ============================================================
//  A) Mutations-Fuzzing
// ============================================================
static const char* kTokens[] = {
    "[", "]", "{", "}", "(", ")", ",", "=", "..", ".", ";", "\"", "\\", "/*", "*/", "//", "!", "-", "+", "*", "/", "%",
    "<", ">", "<=", "==", "!=", "for", "in", "while", "if", "else", "func", "define", "return", "train", "generate",
    "model", "not", "and", "or", "true", "false", "x", "xs[0] = ", "for x in ", "0", "1.5", "99999999999999999999999",
    "0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001",
    "\n", " ", "\t", "\r", "\xC3", "\xE2\x82", "\xF0\x9F", "\xFF", "\x80"};
static const int kNumTokens = (int)(sizeof(kTokens) / sizeof(kTokens[0]));

static std::string mutate(const std::string& base, const std::vector<std::string>& others, Rng& r) {
    std::string s = base;
    const int ops = 1 + r.below(4);
    for (int k = 0; k < ops; ++k) {
        const size_t n = s.size();
        const size_t pos = n ? (size_t)r.below((int)n) : 0;
        switch (r.below(8)) {
            case 0:  // Bereich loeschen
                if (n) s.erase(pos, (size_t)(1 + r.below(20)));
                break;
            case 1:  // Schluesselwort/Zeichen einfuegen
                s.insert(std::min(pos, s.size()), kTokens[r.below(kNumTokens)]);
                break;
            case 2:  // Bereich verdoppeln
                if (n) s.insert(pos, s.substr(pos, (size_t)(1 + r.below(40))));
                break;
            case 3:  // Byte ersetzen
                if (n) s[pos] = (char)r.below(256);
                break;
            case 4:  // abschneiden
                if (n) s.resize(pos);
                break;
            case 5: {  // Stueck aus einer anderen Datei einsetzen
                const std::string& o = others[(size_t)r.below((int)others.size())];
                if (!o.empty()) {
                    const size_t from = (size_t)r.below((int)o.size());
                    s.insert(std::min(pos, s.size()), o.substr(from, (size_t)(1 + r.below(120))));
                }
                break;
            }
            case 6: {  // Zeichen/Token sehr oft wiederholen: prueft die Verschachtelungs- und Kettengrenzen
                std::string tok = kTokens[r.below(kNumTokens)];
                std::string rep;
                const int times = 50 + r.below(3000);
                for (int i = 0; i < times; ++i) rep += tok;
                s.insert(std::min(pos, s.size()), rep);
                break;
            }
            default:  // Zeile loeschen
                if (n) {
                    size_t a = s.rfind('\n', pos);
                    a = (a == std::string::npos) ? 0 : a + 1;
                    size_t b = s.find('\n', pos);
                    b = (b == std::string::npos) ? s.size() : b + 1;
                    s.erase(a, b - a);
                }
        }
    }
    return s;
}

// 0 = angenommen, 1 = sauber abgelehnt; andere Ausnahmen laufen nach oben
static int parse_only(const std::string& src) {
    try {
        Lexer lexer(src);
        Parser parser(lexer.tokenize());
        auto program = parser.parse();
        (void)program;
        return 0;
    } catch (const std::runtime_error&) {
        return 1;
    }
}

static void check_mutation_fuzz() {
    std::printf("A) Mutations-Fuzzing von Lexer und Parser\n");
    std::vector<std::filesystem::path> files;
    for (const char* dir : {"examples", "cases"}) {
        std::error_code ec;
        for (auto it = std::filesystem::directory_iterator(dir, ec); !ec && it != std::filesystem::directory_iterator(); it.increment(ec))
            if (it->path().extension() == ".skull") files.push_back(it->path());
    }
    std::sort(files.begin(), files.end());
    expect(files.size() >= 20, "Ausgangsskripte gefunden (" + std::to_string(files.size()) + ")");
    std::vector<std::string> seeds;
    for (const auto& f : files) seeds.push_back(slurp(f));

    Rng r(20240607);
    int accepted = 0, rejected = 0, crashed = 0;
    const int per_seed = 120;
    for (size_t i = 0; i < seeds.size(); ++i) {
        for (int k = 0; k < per_seed; ++k) {
            const std::string m = mutate(seeds[i], seeds, r);
            try {
                (parse_only(m) == 0 ? accepted : rejected)++;
            } catch (const std::exception& e) {
                std::printf("    unerwartete Ausnahme: %s\n", e.what());
                save_repro("mut", ++crashed, m);
            } catch (...) {
                std::printf("    unerwartete Ausnahme (unbekannt)\n");
                save_repro("mut", ++crashed, m);
            }
        }
    }
    std::printf("    %d angenommen, %d abgelehnt\n", accepted, rejected);
    expect(crashed == 0, "keine unerwarteten Ausnahmen bei " + std::to_string(accepted + rejected) + " mutierten Skripten");
    expect(accepted > 500 && rejected > 500, "der Fuzzer trifft beide Pfade (angenommen und abgelehnt)");

    // Zufaellige Bytes: Lexer und Parser muessen auch Muell vertragen
    int junk_crash = 0;
    for (int k = 0; k < 3000; ++k) {
        std::string s;
        const int len = r.below(200);
        for (int i = 0; i < len; ++i) s += (char)(r.chance(60) ? "[](){}=,.\"+-*/ \n\tabxyz019"[r.below(26)] : r.below(256));
        try { parse_only(s); }
        catch (...) { save_repro("junk", ++junk_crash, s); }
    }
    expect(junk_crash == 0, "3000 zufaellige Byte-Folgen: nur saubere Ablehnung");
}

// ============================================================
//  B) Generierte Programme
// ============================================================
struct Gen {
    Rng r;
    std::vector<std::string> nums, strs, lists;   // lesbare Namen im aktuellen Gueltigkeitsbereich
    std::vector<std::string> anums = {"n0", "n1", "n2"};   // davon beschreibbar (nie Schleifenvariablen und -zaehler)
    int loop_depth = 0, counter = 0;
    int funcs = 0;                                // aufrufbare Funktionen f0..f(funcs-1)
    std::string out;
    int indent = 0;

    explicit Gen(unsigned seed) : r(seed) {}

    std::string pick(const std::vector<std::string>& v) { return v[(size_t)r.below((int)v.size())]; }
    std::string num_lit() {
        static const char* v[] = {"0", "1", "2", "3", "5", "10", "0.5", "-1", "-2", "7", "100", "2.5"};
        return v[r.below(12)];
    }
    std::string str_lit() {
        static const char* v[] = {"\"q\"", "\"a\"", "\"abc\"", "\"h\xC3\xA4llo\"", "\"12\"", "\"-3.5\"", "\"x y\"", "\"\xE2\x82\xAC\xF0\x9F\x98\x80\""};
        return v[r.below(8)];
    }
    std::string idx(const std::string& container, int d) {   // meist gueltiger Index
        if (r.chance(4)) return num_expr(d);
        return "(floor(abs(" + num_expr(d) + ")) % max(len(" + container + "), 1))";
    }

    std::string chaos() {   // Absichtlich falscher Typ: ergibt meist einen sauberen Fehler
        switch (r.below(4)) {
            case 0: return str_lit();
            case 1: return "[1, 2]";
            case 2: return "true";
            default: return "(1 / 0)";
        }
    }

    std::string num_expr(int d) {
        if (r.chance_per_mille(2)) return chaos();
        if (d <= 0) return r.chance(50) || nums.empty() ? num_lit() : pick(nums);
        switch (r.below(13)) {
            case 0: case 1: return num_lit();
            case 2: return nums.empty() ? num_lit() : pick(nums);
            case 3: case 4: {
                static const char* ops[] = {"+", "-", "*"};
                return "(" + num_expr(d - 1) + " " + ops[r.below(3)] + " " + num_expr(d - 1) + ")";
            }
            case 5: return "(" + num_expr(d - 1) + " / " + (r.chance(5) ? num_expr(d - 1) : "(abs(" + num_expr(d - 1) + ") + 1)") + ")";
            case 6: return "(" + num_expr(d - 1) + " % " + (r.chance(5) ? num_expr(d - 1) : "(floor(abs(" + num_expr(d - 1) + ")) + 1)") + ")";
            case 7: return "len(" + list_expr(d - 1) + ")";
            case 8: return "len(" + str_expr(d - 1) + ")";
            case 9: { std::string l = list_expr(d - 1); return l + "[" + idx(l, d - 1) + "]"; }
            case 10: return r.chance(5) ? "num(" + str_expr(d - 1) + ")" : "num(str(" + num_expr(d - 1) + "))";
            case 11: return std::string(r.chance(50) ? "min(" : "max(") + num_expr(d - 1) + ", " + num_expr(d - 1) + ")";
            default: return "abs(" + num_expr(d - 1) + ")";
        }
    }
    std::string str_expr(int d) {
        if (r.chance_per_mille(2)) return chaos();
        if (d <= 0) return r.chance(50) || strs.empty() ? str_lit() : pick(strs);
        switch (r.below(8)) {
            case 0: return str_lit();
            case 1: return strs.empty() ? str_lit() : pick(strs);
            case 2: return "(" + str_expr(d - 1) + " + " + str_expr(d - 1) + ")";
            case 3: return "(" + str_expr(d - 1) + " + " + num_expr(d - 1) + ")";
            case 4: { std::string t = str_expr(d - 1);
                      return "substr(" + t + ", " + (r.chance(5) ? num_expr(d - 1) : "(floor(abs(" + num_expr(d - 1) + ")) % (len(" + t + ") + 1))") + ", " +
                             (r.chance(5) ? num_expr(d - 1) : "floor(abs(" + num_expr(d - 1) + "))") + ")"; }
            case 5: return "str(" + list_expr(d - 1) + ")";
            case 6: { std::string s = str_expr(d - 1); return s + "[" + idx(s, d - 1) + "]"; }
            default: return "str(" + num_expr(d - 1) + ")";
        }
    }
    std::string list_expr(int d) {
        if (r.chance_per_mille(2)) return chaos();
        if (d <= 0 || r.chance(35)) return lists.empty() || r.chance(2) ? "[]" : pick(lists);
        switch (r.below(5)) {
            case 0: return "[" + num_expr(d - 1) + ", " + num_expr(d - 1) + "]";
            case 1: return "(" + list_expr(d - 1) + " + " + list_expr(d - 1) + ")";
            case 2: return r.chance(15) ? "[" + list_expr(d - 1) + ", " + list_expr(d - 1) + "]" : "[" + num_expr(d - 1) + "]";
            case 3: return lists.empty() ? "[]" : pick(lists);
            default: return "(" + list_expr(d - 1) + " + [" + num_expr(d - 1) + "])";
        }
    }
    std::string cond(int d) {
        switch (r.below(8)) {
            case 0: return "(" + num_expr(d) + " < " + num_expr(d) + ")";
            case 1: return "(" + num_expr(d) + " == " + num_expr(d) + ")";
            case 2: return "(" + str_expr(d) + " == " + str_expr(d) + ")";
            case 3: return "(" + list_expr(d) + " == " + list_expr(d) + ")";
            case 4: return "(" + list_expr(d) + " != " + list_expr(d) + ")";
            case 5: return "not " + std::string(r.chance(50) ? num_expr(d) : list_expr(d));
            case 6: return "(" + num_expr(d) + " >= " + num_expr(d) + " and " + str_expr(d) + " != \"\")";
            default: return list_expr(d);
        }
    }

    void line(const std::string& s) { out += std::string((size_t)indent * 2, ' ') + s + "\n"; }

    void block(int n, int depth) {
        ++indent;
        const size_t ns = nums.size(), ss = strs.size(), ls = lists.size();
        for (int i = 0; i < n; ++i) stmt(depth);
        nums.resize(ns); strs.resize(ss); lists.resize(ls);
        --indent;
    }

    void stmt(int depth) {
        const int choice = r.below(loop_depth >= 2 || depth <= 0 ? 12 : 17);
        switch (choice) {
            case 0: case 1: line(pick(anums) + " = " + num_expr(2)); break;
            case 2: if (!strs.empty()) line(pick(strs) + " = " + str_expr(2)); break;
            case 3: if (!lists.empty()) line(pick(lists) + " = " + list_expr(2)); break;
            case 4: case 5: if (!lists.empty()) line("push(" + pick(lists) + ", " + (r.chance(97) ? num_expr(2) : list_expr(1)) + ")"); break;
            case 6: if (!lists.empty()) { std::string l = pick(lists); line(l + "[" + idx(l, 1) + "] = " + (r.chance(97) ? num_expr(2) : list_expr(1))); } break;
            case 7: if (!lists.empty()) { std::string l = pick(lists); line(r.chance(95) ? "if len(" + l + ") > 2 { pop(" + l + ") }" : "pop(" + l + ")"); } break;
            case 8: case 9: line("print(" + (r.chance(35) ? list_expr(2) : r.chance(50) ? num_expr(2) : str_expr(2)) + ")"); break;
            case 10: if (funcs > 0) line("f" + std::to_string(r.below(funcs)) + "(" + num_expr(1) + ", " + list_expr(1) + ")"); break;
            case 11: line("print(" + cond(2) + ")"); break;
            case 12: {   // if / else
                line("if " + cond(2) + " {");
                block(1 + r.below(3), depth - 1);
                if (r.chance(50)) { line("} else {"); block(1 + r.below(3), depth - 1); }
                line("}");
                break;
            }
            case 13: case 14: {   // for i in a..b
                const std::string v = "i" + std::to_string(counter++);
                line("for " + v + " in " + std::to_string(r.below(3)) + ".." + std::to_string(r.below(4)) + " {");
                const size_t ns = nums.size();
                ++loop_depth; nums.push_back(v);
                block(1 + r.below(3), depth - 1);
                --loop_depth; nums.resize(ns);
                line("}");
                break;
            }
            case 15: {   // for x in liste / text
                const bool text = r.chance(30);
                const std::string v = (text ? "c" : "e") + std::to_string(counter++);
                line("for " + v + " in " + (text ? str_expr(1) : list_expr(1)) + " {");
                const size_t ns = nums.size(), ss = strs.size();
                ++loop_depth;
                (text ? strs : nums).push_back(v);
                block(1 + r.below(3), depth - 1);
                --loop_depth; nums.resize(ns); strs.resize(ss);
                line("}");
                break;
            }
            default: {   // while mit Zaehler
                const std::string v = "w" + std::to_string(counter++);
                line(v + " = 0");
                line("while " + v + " < " + std::to_string(1 + r.below(3)) + " {");
                const size_t ns = nums.size();
                ++loop_depth; nums.push_back(v);
                ++indent;
                line(v + " = " + v + " + 1");
                --indent;
                block(1 + r.below(2), depth - 1);
                --loop_depth; nums.resize(ns);
                line("}");
            }
        }
    }

    std::string program() {
        out.clear(); indent = 0;
        nums = {"n0", "n1", "n2"}; strs = {"s0", "s1"}; lists = {"l0", "l1", "l2"};
        line("n0 = 1"); line("n1 = 2.5"); line("n2 = 0");
        line("s0 = \"abc\""); line("s1 = \"xy\"");
        line("l0 = [1, 2, 3]"); line("l1 = [4, 5]"); line("l2 = [6, 7, 8, 9]");
        funcs = 0;
        const int nf = r.below(4);
        for (int k = 0; k < nf; ++k) {
            line("define func f" + std::to_string(k) + "(a, b) {");
            const auto sn = nums, sl = lists;
            nums.push_back("a"); lists.push_back("b");
            const int ld = loop_depth;
            funcs = k;                    // nur kleinere Funktionen: keine Rekursion
            block(2 + r.below(4), 2);
            ++indent; line("return " + num_expr(2)); --indent;
            loop_depth = ld;
            nums = sn; lists = sl;
            line("}");
        }
        funcs = nf;
        const int n = 8 + r.below(16);
        for (int i = 0; i < n; ++i) stmt(3);
        line("print(n0, s0, l0)");
        return out;
    }
};

static void check_generated_programs() {
    std::printf("B) Generierte Programme\n");
    int completed = 0, rejected = 0, unstable = 0, crashed = 0;
    double slowest = 0.0;
    int slowest_idx = -1;
    std::map<std::string, int> reasons;
    for (int k = 0; k < 600; ++k) {
        Gen g(1000 + (unsigned)k);
        const std::string src = g.program();
        try {
            const auto t0 = std::chrono::steady_clock::now();
            const Outcome a = run_script(src);
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (secs > slowest) { slowest = secs; slowest_idx = k; }
            const Outcome b = run_script(src);
            if (a.ok != b.ok || a.out != b.out || a.error != b.error) {
                std::printf("    nicht reproduzierbar (Programm %d)\n", k);
                save_repro("gen_unstable", ++unstable, src);
            }
            if (a.ok) ++completed;
            else { ++rejected; reasons[a.error.substr(a.error.find(':') + 1, 18)]++; }
        } catch (const std::exception& e) {
            std::printf("    unerwartete Ausnahme in Programm %d: %s\n", k, e.what());
            save_repro("gen", ++crashed, src);
        } catch (...) {
            std::printf("    unerwartete Ausnahme in Programm %d\n", k);
            save_repro("gen", ++crashed, src);
        }
    }
    std::printf("    %d vollstaendig gelaufen, %d mit Fehlermeldung beendet; langsamstes Programm: %d (%.2f s)\n",
                completed, rejected, slowest_idx, slowest);
    for (const auto& kv : reasons) std::printf("      %4d x %s\n", kv.second, kv.first.c_str());
    expect(crashed == 0, "keine unerwarteten Ausnahmen");
    expect(unstable == 0, "jedes Programm liefert bei zweitem Lauf dasselbe Ergebnis");
    expect(completed >= 200, "genug Programme laufen bis zum Ende (der Generator erzeugt meist gueltige Programme)");
}

// ============================================================
//  C) Modelltest: Listen und Text gegen ein C++-Modell
// ============================================================
using ModelList = std::shared_ptr<std::vector<long long>>;

static std::string fmt_list(const std::vector<long long>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) { if (i) s += ", "; s += std::to_string(v[i]); }
    return s + "]";
}

static void check_list_model() {
    std::printf("C1) Listen gegen ein C++-Modell (Referenz-Semantik, negative Indizes, Fehler mit Zeilennummer)\n");
    int mismatches = 0, programs = 0, errors_seen = 0;
    for (int seed = 0; seed < 300; ++seed) {
        Rng r((unsigned)(5000 + seed));
        const char* names[] = {"a", "b", "c"};
        std::vector<ModelList> model(3);
        std::string src, expected, expected_error;
        int lineno = 0;
        auto emit = [&](const std::string& s) { src += s + "\n"; ++lineno; };
        for (int i = 0; i < 3; ++i) {
            model[(size_t)i] = std::make_shared<std::vector<long long>>();
            emit(std::string(names[i]) + " = []");
        }
        const int nops = 10 + r.below(50);
        for (int op = 0; op < nops && expected_error.empty(); ++op) {
            const int x = r.below(3), y = r.below(3);
            auto& vx = *model[(size_t)x];
            const std::string X = names[x], Y = names[y];
            const long long k = r.below(21) - 10;
            // gueltiger Index mit gelegentlich ungueltigen Werten
            auto pick_index = [&](long long& i) {
                const long long n = (long long)vx.size();
                i = r.chance(4) ? (long long)r.below(7) - 3 + (r.chance(50) ? n : 0) : (n ? (long long)r.below((int)(2 * n)) - n : 0);
            };
            auto in_range = [&](long long i) { const long long n = (long long)vx.size(); return i >= -n && i < n; };
            auto range_error = [&](long long i) {
                const long long n = (long long)vx.size();
                std::string e = "Zeile " + std::to_string(lineno) + ": Index " + std::to_string(i) + " ausserhalb der Liste";
                (void)n;
                return e;
            };
            int which = r.below(11);
            // Auf leeren Listen meist erst etwas anhaengen, damit nicht jedes Programm gleich mit einem Fehler endet
            if (vx.empty() && (which == 2 || which == 3 || which == 4) && !r.chance(5)) which = 0;
            switch (which) {
                case 0: case 1: emit("push(" + X + ", " + std::to_string(k) + ")"); vx.push_back(k); break;
                case 2: {
                    emit("print(pop(" + X + "))");
                    if (vx.empty()) expected_error = "Zeile " + std::to_string(lineno) + ": pop(): die Liste ist leer";
                    else { expected += std::to_string(vx.back()) + "\n"; vx.pop_back(); }
                    break;
                }
                case 3: {
                    long long i; pick_index(i);
                    emit(X + "[" + std::to_string(i) + "] = " + std::to_string(k));
                    if (!in_range(i)) expected_error = range_error(i);
                    else vx[(size_t)(i < 0 ? i + (long long)vx.size() : i)] = k;
                    break;
                }
                case 4: {
                    long long i; pick_index(i);
                    emit("print(" + X + "[" + std::to_string(i) + "])");
                    if (!in_range(i)) expected_error = range_error(i);
                    else expected += std::to_string(vx[(size_t)(i < 0 ? i + (long long)vx.size() : i)]) + "\n";
                    break;
                }
                case 5: emit("print(len(" + X + "))"); expected += std::to_string(vx.size()) + "\n"; break;
                case 6: emit("print(" + X + ")"); expected += fmt_list(vx) + "\n"; break;
                case 7: emit(X + " = " + Y); model[(size_t)x] = model[(size_t)y]; break;   // Alias
                case 8: emit(X + " = " + Y + " + []"); model[(size_t)x] = std::make_shared<std::vector<long long>>(*model[(size_t)y]); break;
                case 9: {
                    const int z = r.below(3);
                    emit(X + " = " + Y + " + " + names[z]);
                    auto n = std::make_shared<std::vector<long long>>(*model[(size_t)y]);
                    n->insert(n->end(), model[(size_t)z]->begin(), model[(size_t)z]->end());
                    model[(size_t)x] = n;
                    break;
                }
                default: {
                    emit("print(" + X + " == " + Y + ")");
                    expected += (*model[(size_t)x] == *model[(size_t)y]) ? "true\n" : "false\n";
                    break;
                }
            }
        }
        // zum Schluss alles ausgeben: Wert, Laenge und for-each ueber jede Liste
        if (expected_error.empty()) {
            for (int i = 0; i < 3; ++i) {
                emit(std::string("for x in ") + names[i] + " { print(x) }");
                for (long long v : *model[(size_t)i]) expected += std::to_string(v) + "\n";
            }
        }
        const Outcome o = run_script(src, 100000, 1 << 20);
        ++programs;
        bool good = (o.out == expected);
        if (expected_error.empty()) good = good && o.ok;
        else { good = good && !o.ok && o.error.find(expected_error) != std::string::npos; ++errors_seen; }
        if (!good) {
            ++mismatches;
            if (mismatches <= 3) {
                std::printf("    Abweichung (seed %d)\n    erwartet Fehler: '%s'\n    bekommen: ok=%d error='%s'\n", seed,
                            expected_error.c_str(), (int)o.ok, o.error.c_str());
                save_repro("listmodel", mismatches, src);
            }
        }
    }
    std::printf("    %d Programme, davon %d mit erwartetem Fehler\n", programs, errors_seen);
    expect(mismatches == 0, "Skull stimmt in allen Programmen mit dem Modell ueberein");
    expect(errors_seen > 20 && errors_seen < programs - 100, "sowohl Fehlerfaelle als auch fehlerfreie Laeufe kommen vor");
}

static void check_text_model() {
    std::printf("C2) Text (UTF-8) gegen ein C++-Modell\n");
    static const char* alphabet[] = {"a", "b", " ", "z", "\xC3\xA4", "\xC3\xB6", "\xE2\x82\xAC", "\xF0\x9F\x98\x80"};
    int mismatches = 0, programs = 0;
    for (int seed = 0; seed < 400; ++seed) {
        Rng r((unsigned)(9000 + seed));
        std::vector<std::string> cps;
        const int n = r.below(9);
        for (int i = 0; i < n; ++i) cps.push_back(alphabet[r.below(8)]);
        std::string text;
        for (const auto& c : cps) text += c;
        std::string src = "t = \"" + text + "\"\n", expected;
        int lineno = 1;
        auto emit = [&](const std::string& s) { src += s + "\n"; ++lineno; };
        std::string expected_error;
        emit("print(len(t))");
        expected += std::to_string(cps.size()) + "\n";
        for (int op = 0; op < 8 && expected_error.empty(); ++op) {
            if (r.chance(50)) {
                const long long i = (long long)r.below(2 * (int)cps.size() + 4) - (long long)cps.size() - 2;
                emit("print(t[" + std::to_string(i) + "])");
                const long long len = (long long)cps.size();
                if (i < -len || i >= len) expected_error = "Zeile " + std::to_string(lineno) + ": Index " + std::to_string(i) + " ausserhalb des Textes";
                else expected += cps[(size_t)(i < 0 ? i + len : i)] + "\n";
            } else {
                const int start = r.below((int)cps.size() + 3) - 1;
                const int cnt = r.below((int)cps.size() + 3) - 1;
                emit("print(substr(t, " + std::to_string(start) + ", " + std::to_string(cnt) + "))");
                if (start < 0 || start > (int)cps.size()) expected_error = "Zeile " + std::to_string(lineno) + ": substr(): start " + std::to_string(start);
                else if (cnt < 0) expected_error = "Zeile " + std::to_string(lineno) + ": substr(): anzahl darf nicht negativ sein";
                else {
                    std::string s;
                    for (int i = start; i < std::min(start + cnt, (int)cps.size()); ++i) s += cps[(size_t)i];
                    expected += s + "\n";
                }
            }
        }
        if (expected_error.empty()) {
            emit("for c in t { print(c) }");
            for (const auto& c : cps) expected += c + "\n";
        }
        const Outcome o = run_script(src, 100, 1 << 20);
        ++programs;
        bool good = (o.out == expected);
        if (expected_error.empty()) good = good && o.ok;
        else good = good && !o.ok && o.error.find(expected_error) != std::string::npos;
        if (!good && ++mismatches <= 3) {
            std::printf("    Abweichung (seed %d): erwartet Fehler '%s', bekommen ok=%d '%s'\n", seed, expected_error.c_str(), (int)o.ok, o.error.c_str());
            save_repro("textmodel", mismatches, src);
        }
    }
    expect(mismatches == 0, std::to_string(programs) + " Text-Programme stimmen mit dem Modell ueberein");

    // Ungueltiges UTF-8: jedes unpassende Byte zaehlt als ein Zeichen, nichts geht verloren
    const std::string bad = std::string("a\xC3") + "b\xE2\x82" + "c\xFF\x80";
    size_t total = 0;
    for (size_t i = 0; i < bad.size(); i += utf8_char_len(bad, i)) total += utf8_char_len(bad, i);
    expect(total == bad.size() && utf8_count(bad) == 8, "ungueltiges UTF-8: jedes lose Byte ist ein Zeichen, kein Byte geht verloren");
    const auto off = utf8_offsets(bad);
    expect(off.size() == utf8_count(bad) + 1 && off.back() == bad.size(), "utf8_offsets: Anzahl und Ende stimmen");
}

// ============================================================
//  D) Grenzen und Robustheit
// ============================================================
static int deep_list_destruction() {
    // 300 000 ineinander verschachtelte Listen; auf diesem kleinen Stack wuerde rekursives Freigeben abstuerzen
    ListPtr root = make_list();
    ListPtr cur = root;
    for (int i = 0; i < 300000; ++i) {
        ListPtr n = make_list();
        cur->items.push_back(SkullValue(n));
        cur = n;
    }
    cur.reset();
    root.reset();
    return 0;
}

static void check_limits() {
    std::printf("D) Grenzen\n");
    expect(run_with_stack(1u << 20, deep_list_destruction) == 0,
           "300000 Ebenen tiefe Liste wird auf 1-MB-Stack freigegeben (nicht rekursiv)");

    Outcome o = run_script("a = [0]\nwhile true { a = a + a }", 1000);
    expect(!o.ok && o.error.find("Zeile 2: Liste zu gross (hoechstens 1000 Elemente)") != std::string::npos, "a = a + a endet mit 'Liste zu gross'");
    o = run_script("a = []\nwhile true { push(a, 1) }", 1000);
    expect(!o.ok && o.error.find("Liste zu gross") != std::string::npos, "push-Schleife endet mit 'Liste zu gross'");
    o = run_script("s = \"ab\"\nwhile true { s = s + s }", 1000, 1000);
    expect(!o.ok && o.error.find("Text zu lang (hoechstens 1000 Bytes)") != std::string::npos, "s = s + s endet mit 'Text zu lang'");
    o = run_script("a = [1, 2, 3]\nprint(a)", 2);
    expect(!o.ok && o.error.find("Liste zu gross") != std::string::npos, "Listen-Literal ueber der Grenze");

    // sehr tief verschachtelte Liste: Ausgabe gekuerzt, kein Stack-Problem
    o = run_script("a = []\nfor i in 1..2000 { a = [a] }\nprint(a)", 64);
    expect(o.ok && o.out == "[[[[[[[[[...]]]]]]]]]\n", "tiefe Liste wird ab Ebene 8 mit [...] gekuerzt");

    // Teillisten teilen: Ausgabe bleibt klein, Vergleich wird abgebrochen statt exponentiell zu laufen
    o = run_script("a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\nfor i in 1..9 { a = [a, a, a, a, a, a, a, a, a, a] }\nprint(a)", 64);
    expect(o.ok && o.out.size() < 2000000, "geteilte Teillisten: Ausgabe ist begrenzt (" + std::to_string(o.out.size()) + " Byte)");
    o = run_script("a = [1]\nb = [1]\nfor i in 1..40 { a = [a, a]\n b = [b, b] }\nprint(a == b)");
    expect(!o.ok && o.error.find("Listen-Vergleich zu aufwaendig") != std::string::npos, "exponentieller Vergleich wird abgebrochen");
    o = run_script("a = [1]\nb = [1]\nfor i in 1..300 { a = [a]\n b = [b] }\nprint(a == b)");
    expect(!o.ok && o.error.find("zu tief verschachtelt zum Vergleichen") != std::string::npos, "Vergleich tiefer Listen wird abgebrochen");
    o = run_script("a = [1]\nfor i in 1..40 { a = [a, a] }\nprint(a == a)");
    expect(o.ok && o.out == "true\n", "dieselbe Liste ist sofort gleich (kein Durchlaufen)");

    // Zyklen werden verhindert
    o = run_script("a = []\nb = [a]\nc = [b]\npush(a, c)");
    expect(!o.ok && o.error.find("Zeile 4: Eine Liste kann sich nicht selbst enthalten") != std::string::npos, "Zyklus ueber drei Listen wird abgelehnt");
    o = run_script("a = [1]\nb = [a, a]\npush(a, 2)\nprint(b)");
    expect(o.ok && o.out == "[[1, 2], [1, 2]]\n", "dieselbe Liste darf mehrfach vorkommen (kein Zyklus)");

    // Verschachtelung im Parser
    std::string deep(199, '[');
    deep += std::string(199, ']');
    o = run_script("x = " + deep);
    expect(o.ok, "199 Ebenen verschachtelte Liste im Quelltext laufen");
    o = run_script("x = " + std::string(250, '[') + std::string(250, ']'));
    expect(!o.ok && o.error.find("zu tief verschachtelt") != std::string::npos, "250 Ebenen werden mit Fehlermeldung abgelehnt");
    std::string many_if;
    for (int i = 0; i < 300; ++i) many_if += "if true {\n";
    for (int i = 0; i < 300; ++i) many_if += "}\n";
    o = run_script(many_if);
    expect(!o.ok && o.error.find("zu tief verschachtelt") != std::string::npos, "300 verschachtelte if-Bloecke werden abgelehnt");
    o = run_script("x = 1" + [] { std::string s; for (int i = 0; i < 15000; ++i) s += " + 1"; return s; }() + "\nprint(x)");
    expect(o.ok && o.out == "15001\n", "Kette mit 15000 Summanden laeuft");
}

template <typename F>
static void timed(F f) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("    (%.1f s)\n", s);
}

int main() {
    // Parser und Interpreter laufen wie im echten Programm auf dem grossen Stack
    return run_with_stack(512u * 1024u * 1024u, [] {
        timed(check_mutation_fuzz);
        timed(check_generated_programs);
        timed(check_list_model);
        timed(check_text_model);
        timed(check_limits);
        if (failures) { std::printf("\n%d Pruefung(en) fehlgeschlagen\n", failures); return 1; }
        std::printf("\nAlle Pruefungen bestanden\n");
        return 0;
    });
}
