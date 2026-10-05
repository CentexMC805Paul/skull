#pragma once
#include <string>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <climits>
#include <sstream>
#include <locale>
#include <unordered_set>
#include "ast.h"
#include "tensor.h"
#include "trainer.h"
#include "generator.h"
#include "version.h"

// ============================================================
//  SKULL INTERPRETER
//  Fuehrt den AST direkt aus (Tree-Walking).
// ============================================================

struct SkullList;
using ListPtr = std::shared_ptr<SkullList>;

// Obergrenzen: ein fehlerhaftes Skript soll mit einer Fehlermeldung enden, nicht den Rechner
// in den Speichermangel treiben.
constexpr size_t SKULL_MAX_LIST   = 10000000;        // Elemente pro Liste
constexpr size_t SKULL_MAX_STRING = 256u << 20;      // Bytes pro Text

struct SkullValue {
    enum class Kind { NUMBER, STRING, BOOL, TENSOR, LIST, NOTHING } kind;
    double number = 0.0;
    std::string text;
    bool flag = false;
    TensorPtr tensor;
    ListPtr list;      // Listen haben Referenz-Semantik: b = a zeigt auf dieselbe Liste

    SkullValue() : kind(Kind::NOTHING) {}
    SkullValue(double v) : kind(Kind::NUMBER), number(v) {}
    SkullValue(const std::string& s) : kind(Kind::STRING), text(s) {}
    SkullValue(bool b) : kind(Kind::BOOL), flag(b) {}
    SkullValue(TensorPtr t) : kind(Kind::TENSOR), tensor(t) {}
    SkullValue(ListPtr l) : kind(Kind::LIST), list(std::move(l)) {}

    bool is_truthy() const;
    double as_number(int line = 0) const {
        if (kind == Kind::NUMBER) return number;
        if (kind == Kind::BOOL)   return flag ? 1.0 : 0.0;
        if (kind == Kind::TENSOR && tensor && tensor->rows == 1 && tensor->cols == 1)
            return tensor->data[0];
        throw std::runtime_error("Zeile " + std::to_string(line) + ": Zahl erwartet");
    }
    const std::string& as_string(int line = 0) const {
        if (kind == Kind::STRING) return text;
        throw std::runtime_error("Zeile " + std::to_string(line) + ": Text (String in Anfuehrungszeichen) erwartet");
    }
    TensorPtr as_tensor(int line = 0) const {
        if (kind == Kind::TENSOR) return tensor;
        if (kind == Kind::NUMBER) {
            auto t = std::make_shared<Tensor>(1, 1);
            t->data[0] = number;
            return t;
        }
        throw std::runtime_error("Zeile " + std::to_string(line) + ": Tensor erwartet");
    }
    std::string to_string() const;
    void print() const {
        if (kind == Kind::TENSOR && tensor) tensor->print();
        else std::cout << to_string();
    }
    // Fuer Listen: schreibt "[1, "a", [2]]" in out; budget begrenzt die Gesamtzahl ausgegebener Elemente
    void format_list(std::string& out, int depth, size_t& budget) const;
};

struct SkullList {
    std::vector<SkullValue> items;
    ~SkullList();
};

// Teilbaeume nicht rekursiv freigeben: eine tief verschachtelte Liste (a = [a] in einer Schleife)
// wuerde sonst beim Aufraeumen den Stack sprengen.
inline SkullList::~SkullList() {
    std::vector<ListPtr> pending;
    auto take = [&pending](std::vector<SkullValue>& v) {
        for (auto& x : v)
            if (x.list && x.list.use_count() == 1) pending.push_back(std::move(x.list));
    };
    take(items);
    while (!pending.empty()) {
        ListPtr cur = std::move(pending.back());
        pending.pop_back();
        take(cur->items);
    }
}

inline ListPtr make_list() { return std::make_shared<SkullList>(); }

inline bool SkullValue::is_truthy() const {
    if (kind == Kind::NUMBER) return number != 0.0;
    if (kind == Kind::BOOL)   return flag;
    if (kind == Kind::STRING) return !text.empty();
    if (kind == Kind::TENSOR) return tensor != nullptr;
    if (kind == Kind::LIST)   return list && !list->items.empty();
    return false;
}

inline std::string SkullValue::to_string() const {
    if (kind == Kind::NUMBER) {
        if (number == std::floor(number) && std::abs(number) < 1e15)
            return std::to_string((long long)number);
        std::string s = std::to_string(number);
        s.erase(s.find_last_not_of('0') + 1);
        if (s.back() == '.') s += "0";
        return s;
    }
    if (kind == Kind::STRING) return text;
    if (kind == Kind::BOOL)   return flag ? "true" : "false";
    if (kind == Kind::TENSOR && tensor) return tensor->to_string();
    if (kind == Kind::LIST) {
        std::string out;
        size_t budget = 100000;   // sehr lange Listen werden mit "..." abgekuerzt
        format_list(out, 0, budget);
        return out;
    }
    return "";
}

inline void SkullValue::format_list(std::string& out, int depth, size_t& budget) const {
    if (depth >= 8) { out += "[...]"; return; }
    out += '[';
    bool first = true;
    for (const auto& x : list->items) {
        if (budget == 0) { out += first ? "..." : ", ..."; break; }
        --budget;
        if (!first) out += ", ";
        first = false;
        if (x.kind == Kind::LIST)         x.format_list(out, depth + 1, budget);
        else if (x.kind == Kind::STRING)  { out += '"'; out += x.text; out += '"'; }
        else if (x.kind == Kind::NOTHING) out += "nothing";
        else                              out += x.to_string();
    }
    out += ']';
}

// Gleichheit mit Typ: 1 == "1" ist false. Zahl und Bool vergleichen als Zahl.
// Listen vergleichen elementweise. Tiefe und Aufwand sind begrenzt (Listen koennen sich Teillisten teilen,
// der naive Vergleich wuerde dann exponentiell lang dauern).
inline bool skull_equal_impl(const SkullValue& a, const SkullValue& b, int depth, size_t& steps) {
    using K = SkullValue::Kind;
    auto numeric = [](const SkullValue& v) { return v.kind == K::NUMBER || v.kind == K::BOOL; };
    if (numeric(a) && numeric(b)) return a.as_number() == b.as_number();
    if (a.kind != b.kind) return false;
    switch (a.kind) {
        case K::STRING:  return a.text == b.text;
        case K::TENSOR:  return a.tensor == b.tensor;   // gleiche Tensor-Instanz
        case K::NOTHING: return true;
        case K::LIST: {
            if (a.list == b.list) return true;
            if (a.list->items.size() != b.list->items.size()) return false;
            if (depth >= 200)
                throw std::runtime_error("Listen zu tief verschachtelt zum Vergleichen (max. 200 Ebenen)");
            for (size_t i = 0; i < a.list->items.size(); ++i) {
                if (steps == 0) throw std::runtime_error("Listen-Vergleich zu aufwaendig");
                --steps;
                if (!skull_equal_impl(a.list->items[i], b.list->items[i], depth + 1, steps)) return false;
            }
            return true;
        }
        default:         return false;
    }
}
inline bool skull_values_equal(const SkullValue& a, const SkullValue& b) {
    size_t steps = 10000000;
    return skull_equal_impl(a, b, 0, steps);
}

// Enthaelt v (selbst oder in Teillisten) die Liste target? Verhindert Zyklen (a enthaelt a):
// die wuerden Ausgabe und Vergleich endlos laufen lassen und sich nie freigeben lassen.
inline bool list_contains(const SkullValue& v, const SkullList* target) {
    if (v.kind != SkullValue::Kind::LIST) return false;
    std::vector<const SkullList*> stack{v.list.get()};
    std::unordered_set<const SkullList*> seen;
    while (!stack.empty()) {
        const SkullList* l = stack.back();
        stack.pop_back();
        if (l == target) return true;
        if (!seen.insert(l).second) continue;
        for (const auto& x : l->items)
            if (x.kind == SkullValue::Kind::LIST) stack.push_back(x.list.get());
    }
    return false;
}

// ---- Text als Folge von Zeichen (UTF-8): ein Umlaut ist ein Zeichen, nicht zwei Bytes ----
inline size_t utf8_char_len(const std::string& s, size_t i) {
    const unsigned char c = (unsigned char)s[i];
    size_t n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
    if (n == 1 || i + n > s.size()) return 1;
    for (size_t k = 1; k < n; ++k)
        if (((unsigned char)s[i + k] & 0xC0) != 0x80) return 1;   // ungueltige Folge: ein Byte = ein Zeichen
    return n;
}
// Byte-Positionen aller Zeichen, plus s.size() am Ende (also Anzahl Zeichen + 1 Eintraege)
inline std::vector<size_t> utf8_offsets(const std::string& s) {
    std::vector<size_t> off;
    off.reserve(s.size() + 1);
    for (size_t i = 0; i < s.size(); i += utf8_char_len(s, i)) off.push_back(i);
    off.push_back(s.size());
    return off;
}
inline size_t utf8_count(const std::string& s) {
    size_t n = 0;
    for (size_t i = 0; i < s.size(); i += utf8_char_len(s, i)) ++n;
    return n;
}

struct ReturnSignal {
    SkullValue value;
    int line = 0;
};

struct Environment {
    std::map<std::string, SkullValue> vars;
    std::shared_ptr<Environment> parent;
    Environment() = default;
    explicit Environment(std::shared_ptr<Environment> p) : parent(p) {}
    void define(const std::string& n, SkullValue v) { vars[n] = std::move(v); }
    void assign(const std::string& n, SkullValue v) {
        if (vars.count(n)) { vars[n] = std::move(v); return; }
        if (parent && parent->has(n)) { parent->assign(n, std::move(v)); return; }
        vars[n] = std::move(v);
    }
    SkullValue get(const std::string& n, int line = 0) const {
        auto it = vars.find(n);
        if (it != vars.end()) return it->second;
        if (parent) return parent->get(n, line);
        throw std::runtime_error("Zeile " + std::to_string(line) + ": Unbekannte Variable '" + n + "'");
    }
    bool has(const std::string& n) const {
        if (vars.count(n)) return true;
        if (parent) return parent->has(n);
        return false;
    }
};

struct SkullFunction {
    std::string name;
    std::vector<std::string> params;
    std::vector<std::unique_ptr<StmtNode>>* body;
    std::shared_ptr<Environment> closure;
};

class Interpreter {
private:
    // Maximale Schachtelungstiefe von Skull-Funktionsaufrufen. Schuetzt vor
    // Stack-Overflow bei unendlicher Rekursion (Windows-Build hat 16 MB Stack).
    static constexpr int MAX_CALL_DEPTH = 1000;

    std::shared_ptr<Environment> global_env;
    std::map<std::string, SkullFunction> functions;
    std::map<std::string, const ModelStmt*> models;
    int call_depth = 0;
    size_t max_list_   = SKULL_MAX_LIST;
    size_t max_string_ = SKULL_MAX_STRING;
    TrainResult last_train_;
    bool has_train_ = false;

    struct DepthGuard {
        int& depth;
        DepthGuard(int& d, int line) : depth(d) {
            if (depth >= MAX_CALL_DEPTH)
                throw std::runtime_error(
                    "Zeile " + std::to_string(line) + ": Rekursionstiefe ueberschritten (max. " +
                    std::to_string(MAX_CALL_DEPTH) + " verschachtelte Funktionsaufrufe)");
            ++depth;
        }
        ~DepthGuard() { --depth; }
    };

    // ---- Hilfen fuer Feld-Bloecke (model / train / generate) ----
    static void warn_fields(const char* block, const std::string& model_name,
                            const std::vector<std::pair<std::string, int>>& names,
                            const std::set<std::string>& known,
                            const std::set<std::string>& unsupported) {
        for (const auto& f : names) {
            if (known.count(f.first)) continue;
            if (unsupported.count(f.first))
                std::cout << "[WARNUNG] Zeile " << f.second << ": '" << f.first << "' in " << block
                          << " " << model_name << " wird noch nicht unterstuetzt und ignoriert"
                          << " (das Modell hat eine feste Struktur)\n";
            else {
                std::cout << "[WARNUNG] Zeile " << f.second << ": Unbekanntes Feld '" << f.first
                          << "' in " << block << " " << model_name << " (ignoriert). Gueltig: ";
                bool first = true;
                for (const auto& k : known) { std::cout << (first ? "" : ", ") << k; first = false; }
                std::cout << "\n";
            }
        }
    }

    static int to_int(const SkullValue& v, int line, const char* field) {
        double d = v.as_number(line);
        if (!std::isfinite(d) || d < (double)INT_MIN || d > (double)INT_MAX)
            throw std::runtime_error("Zeile " + std::to_string(line) + ": '" + field +
                                     "' hat einen ungueltigen Wert");
        return (int)d;
    }
    static size_t to_size(const SkullValue& v, int line, const char* field) {
        double d = v.as_number(line);
        if (!std::isfinite(d) || d < 0.0 || d > 1e12)
            throw std::runtime_error("Zeile " + std::to_string(line) + ": '" + field +
                                     "' muss eine nichtnegative Zahl sein");
        return (size_t)d;
    }

    static void need_args(const CallExpr* n, const std::vector<SkullValue>& args, size_t count) {
        if (args.size() < count)
            throw std::runtime_error("Zeile " + std::to_string(n->line) + ": " + n->name + "() braucht " +
                                     std::to_string(count) + (count == 1 ? " Argument" : " Argumente"));
    }

    static std::string err(int line, const std::string& msg) {
        return "Zeile " + std::to_string(line) + ": " + msg;
    }

    // Index pruefen: ganze Zahl; negativ zaehlt vom Ende (-1 = letztes Element)
    static size_t resolve_index(const SkullValue& iv, size_t len, int line, const char* what) {
        if (iv.kind != SkullValue::Kind::NUMBER)
            throw std::runtime_error(err(line, "Index muss eine Zahl sein"));
        double d = iv.number;
        if (!std::isfinite(d) || d != std::floor(d))
            throw std::runtime_error(err(line, "Index muss eine ganze Zahl sein (bekommen: " + iv.to_string() + ")"));
        if (d < 0.0) d += (double)len;
        if (d < 0.0 || d >= (double)len) {
            std::string msg = "Index " + iv.to_string() + " ausserhalb " + what;
            if (len == 0) msg += " (leer)";
            else msg += " (Laenge " + std::to_string(len) + ", erlaubt 0.." + std::to_string(len - 1) + " oder -1.." + std::to_string(-(long long)len) + ")";
            throw std::runtime_error(err(line, msg));
        }
        return (size_t)d;
    }

    static double whole_number(const SkullValue& v, int line, const char* fn, const char* what) {
        if (v.kind != SkullValue::Kind::NUMBER || !std::isfinite(v.number) || v.number != std::floor(v.number))
            throw std::runtime_error(err(line, std::string(fn) + "(): " + what + " muss eine ganze Zahl sein"));
        return v.number;
    }

    // Strikte Umwandlung Text -> Zahl (unabhaengig von der eingestellten Sprache, ohne "nan"/"inf"/Hex)
    static bool parse_number(const std::string& text, double& out) {
        size_t a = text.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) return false;
        size_t b = text.find_last_not_of(" \t\r\n");
        std::istringstream is(text.substr(a, b - a + 1));
        is.imbue(std::locale::classic());
        double d = 0.0;
        is >> d;
        if (is.fail() || !is.eof() || !std::isfinite(d)) return false;
        const unsigned char first = (unsigned char)text[a];
        if (!(std::isdigit(first) || first == '-' || first == '+' || first == '.')) return false;
        out = d;
        return true;
    }

    void too_big_list(int line) const {
        throw std::runtime_error(err(line, "Liste zu gross (hoechstens " + std::to_string(max_list_) + " Elemente)"));
    }
    void list_add(SkullList& l, SkullValue v, int line) {
        if (l.items.size() >= max_list_) too_big_list(line);
        if (v.kind == SkullValue::Kind::LIST && list_contains(v, &l))
            throw std::runtime_error(err(line, "Eine Liste kann sich nicht selbst enthalten"));
        l.items.push_back(std::move(v));
    }

    SkullValue eval_expr(const ExprNode* node, std::shared_ptr<Environment> env) {
        if (!node) throw std::runtime_error("Leerer Ausdruck");
        if (auto* n = dynamic_cast<const NumberExpr*>(node)) return SkullValue(n->value);
        if (auto* n = dynamic_cast<const StringExpr*>(node)) return SkullValue(n->value);
        if (auto* n = dynamic_cast<const BoolExpr*>(node))   return SkullValue(n->value);
        if (auto* n = dynamic_cast<const IdentExpr*>(node))  return env->get(n->name, n->line);
        if (auto* n = dynamic_cast<const UnaryExpr*>(node))  return SkullValue(!eval_expr(n->operand.get(), env).is_truthy());
        if (auto* n = dynamic_cast<const BinaryExpr*>(node)) return eval_binary(n, env);
        if (auto* n = dynamic_cast<const CallExpr*>(node))   return eval_call(n, env);
        if (auto* n = dynamic_cast<const ListExpr*>(node)) {
            auto l = make_list();
            if (n->items.size() > max_list_) too_big_list(n->line);
            l->items.reserve(n->items.size());
            for (const auto& it : n->items) l->items.push_back(eval_expr(it.get(), env));
            return SkullValue(l);
        }
        if (auto* n = dynamic_cast<const IndexExpr*>(node)) return eval_index(n, env);
        throw std::runtime_error("Unbekannter Ausdruck");
    }

    SkullValue eval_index(const IndexExpr* n, std::shared_ptr<Environment> env) {
        SkullValue obj = eval_expr(n->object.get(), env);
        SkullValue idx = eval_expr(n->index.get(), env);
        if (obj.kind == SkullValue::Kind::LIST)
            return obj.list->items[resolve_index(idx, obj.list->items.size(), n->line, "der Liste")];
        if (obj.kind == SkullValue::Kind::STRING) {
            const auto off = utf8_offsets(obj.text);
            const size_t i = resolve_index(idx, off.size() - 1, n->line, "des Textes");
            return SkullValue(obj.text.substr(off[i], off[i + 1] - off[i]));
        }
        throw std::runtime_error(err(n->line, "[ ] geht nur bei Listen und Text"));
    }

    SkullValue eval_binary(const BinaryExpr* n, std::shared_ptr<Environment> env) {
        // and / or: Kurzschluss-Auswertung (rechte Seite nur wenn noetig); Ergebnis ist true/false
        if (n->op == "and" || n->op == "or") {
            bool l = eval_expr(n->left.get(), env).is_truthy();
            if (n->op == "and" && !l) return SkullValue(false);
            if (n->op == "or"  && l)  return SkullValue(true);
            return SkullValue(eval_expr(n->right.get(), env).is_truthy());
        }
        SkullValue lv = eval_expr(n->left.get(), env);
        SkullValue rv = eval_expr(n->right.get(), env);
        const std::string& op = n->op;
        bool lT = (lv.kind == SkullValue::Kind::TENSOR);
        bool rT = (rv.kind == SkullValue::Kind::TENSOR);

        if (op == "*" && (lT || rT)) {
            if (lv.kind == SkullValue::Kind::NUMBER) return SkullValue(tensor_scale(rv.as_tensor(n->line), lv.number));
            if (rv.kind == SkullValue::Kind::NUMBER) return SkullValue(tensor_scale(lv.as_tensor(n->line), rv.number));
            return SkullValue(tensor_matmul(lv.as_tensor(n->line), rv.as_tensor(n->line)));
        }
        if (op == "+" && (lT || rT)) return SkullValue(tensor_add(lv.as_tensor(n->line), rv.as_tensor(n->line)));
        if (op == "-" && (lT || rT)) return SkullValue(tensor_sub(lv.as_tensor(n->line), rv.as_tensor(n->line)));
        if (op == "+" && (lv.kind == SkullValue::Kind::STRING || rv.kind == SkullValue::Kind::STRING)) {
            std::string a = lv.to_string(), b = rv.to_string();
            if (a.size() + b.size() > max_string_)
                throw std::runtime_error(err(n->line, "Text zu lang (hoechstens " + std::to_string(max_string_) + " Bytes)"));
            return SkullValue(a + b);
        }
        if (op == "+" && (lv.kind == SkullValue::Kind::LIST || rv.kind == SkullValue::Kind::LIST)) {
            if (lv.kind != SkullValue::Kind::LIST || rv.kind != SkullValue::Kind::LIST)
                throw std::runtime_error(err(n->line, "Eine Liste laesst sich nur mit einer Liste oder einem Text addieren "
                                                      "(Element anhaengen: push(liste, wert) oder liste + [wert])"));
            if (lv.list->items.size() + rv.list->items.size() > max_list_) too_big_list(n->line);
            auto l = make_list();
            l->items.reserve(lv.list->items.size() + rv.list->items.size());
            l->items.insert(l->items.end(), lv.list->items.begin(), lv.list->items.end());
            l->items.insert(l->items.end(), rv.list->items.begin(), rv.list->items.end());
            return SkullValue(l);
        }
        if (op == "==" || op == "!=") {
            bool eq;
            try { eq = skull_values_equal(lv, rv); }
            catch (const std::runtime_error& e) { throw std::runtime_error(err(n->line, e.what())); }
            return SkullValue(op == "==" ? eq : !eq);
        }

        double l = lv.as_number(n->line), r = rv.as_number(n->line);
        if (op == "+") return SkullValue(l + r);
        if (op == "-") return SkullValue(l - r);
        if (op == "*") return SkullValue(l * r);
        if (op == "/") {
            if (r == 0.0) throw std::runtime_error("Zeile " + std::to_string(n->line) + ": Division durch 0");
            return SkullValue(l / r);
        }
        if (op == "%") {
            // wie in Python: Ergebnis hat das Vorzeichen des Divisors (-7 % 3 == 2)
            if (r == 0.0) throw std::runtime_error("Zeile " + std::to_string(n->line) + ": Modulo durch 0");
            double m = std::fmod(l, r);
            if (m != 0.0 && ((m < 0.0) != (r < 0.0))) m += r;
            return SkullValue(m);
        }
        if (op == "<")  return SkullValue(l < r);
        if (op == ">")  return SkullValue(l > r);
        if (op == "<=") return SkullValue(l <= r);
        if (op == ">=") return SkullValue(l >= r);
        throw std::runtime_error("Unbekannter Operator '" + op + "'");
    }

    SkullValue eval_call(const CallExpr* n, std::shared_ptr<Environment> env) {
        std::vector<SkullValue> args;
        for (const auto& arg : n->args) args.push_back(eval_expr(arg.get(), env));
        const std::string& name = n->name;
        const int line = n->line;

        if (name == "print") {
            for (size_t i = 0; i < args.size(); ++i) { if (i > 0) std::cout << " "; args[i].print(); }
            std::cout << "\n";
            return SkullValue();
        }
        if (name == "skull_info") { print_skull_info(); return SkullValue(); }
        if (name == "gpu_info")   { skull_print_gpu_status(); return SkullValue(); }
        // Anzahl lebender Tensoren (fuer Speicherleck-Tests)
        if (name == "live_tensors") return SkullValue((double)TensorLiveCounter::count);
        // Ergebnis des letzten train-Blocks (fuer Vergleiche und Parameter-Suchen im Skript)
        if (name == "last_loss") {
            if (!has_train_) throw std::runtime_error("Zeile " + std::to_string(line) + ": last_loss(): es lief noch kein train");
            return SkullValue(last_train_.train_loss);
        }
        if (name == "last_val_loss") {
            if (!has_train_) throw std::runtime_error("Zeile " + std::to_string(line) + ": last_val_loss(): es lief noch kein train");
            if (!last_train_.has_val)
                throw std::runtime_error("Zeile " + std::to_string(line) +
                    ": last_val_loss(): das letzte train hatte keine Validierung (val > 0 und genug Daten noetig)");
            return SkullValue(last_train_.val_loss);
        }
        if (name == "last_seconds") {
            if (!has_train_) throw std::runtime_error("Zeile " + std::to_string(line) + ": last_seconds(): es lief noch kein train");
            return SkullValue(last_train_.seconds);
        }
        if (name == "last_params") {
            if (!has_train_) throw std::runtime_error("Zeile " + std::to_string(line) + ": last_params(): es lief noch kein train");
            return SkullValue((double)last_train_.params);
        }
        if (name == "last_best_epoch") {
            if (!has_train_) throw std::runtime_error("Zeile " + std::to_string(line) + ": last_best_epoch(): es lief noch kein train");
            return SkullValue((double)last_train_.best_epoch);
        }

        if (name == "rand_tensor" || name == "zeros" || name == "ones") {
            need_args(n, args, 2);
            size_t r = to_size(args[0], line, "rows"), c = to_size(args[1], line, "cols");
            if (r == 0 || c == 0 || (double)r * (double)c > 2e9)
                throw std::runtime_error("Zeile " + std::to_string(line) + ": " + name + "(): ungueltige Groesse");
            if (name == "rand_tensor") return SkullValue(tensor_rand(r, c));
            if (name == "zeros")       return SkullValue(tensor_zeros(r, c));
            return SkullValue(tensor_ones(r, c));
        }
        if (name == "relu") {
            need_args(n, args, 1);
            if (args[0].kind == SkullValue::Kind::TENSOR) return SkullValue(tensor_relu(args[0].tensor));
            double x = args[0].as_number(line);
            return SkullValue(x > 0.0 ? x : 0.0);
        }
        if (name == "sigmoid") {
            need_args(n, args, 1);
            if (args[0].kind == SkullValue::Kind::TENSOR) return SkullValue(tensor_sigmoid(args[0].tensor));
            return SkullValue(1.0 / (1.0 + std::exp(-args[0].as_number(line))));
        }
        if (name == "tanh_act") {
            need_args(n, args, 1);
            if (args[0].kind == SkullValue::Kind::TENSOR) return SkullValue(tensor_tanh(args[0].tensor));
            return SkullValue(std::tanh(args[0].as_number(line)));
        }
        if (name == "mse_loss") {
            need_args(n, args, 2);
            return SkullValue(tensor_mse_loss(args[0].as_tensor(line), args[1].as_tensor(line)));
        }
        if (name == "backward")  { need_args(n, args, 1); args[0].as_tensor(line)->backward(); return SkullValue(); }
        if (name == "update")    { need_args(n, args, 2); tensor_update(args[0].as_tensor(line), args[1].as_number(line)); return SkullValue(); }
        if (name == "zero_grad") { need_args(n, args, 1); args[0].as_tensor(line)->zero_grad(); return SkullValue(); }
        if (name == "shape") {
            need_args(n, args, 1);
            auto t = args[0].as_tensor(line);
            std::cout << t->rows << "x" << t->cols;
            return SkullValue();
        }
        if (name == "get_loss") {
            need_args(n, args, 1);
            auto t = args[0].as_tensor(line);
            if (t->rows == 1 && t->cols == 1) return SkullValue(t->data[0]);
            throw std::runtime_error("Zeile " + std::to_string(line) + ": get_loss(): Tensor ist nicht 1x1");
        }
        if (name == "sqrt")  { need_args(n, args, 1); return SkullValue(std::sqrt(args[0].as_number(line))); }
        if (name == "abs")   { need_args(n, args, 1); return SkullValue(std::abs(args[0].as_number(line))); }
        if (name == "floor") { need_args(n, args, 1); return SkullValue(std::floor(args[0].as_number(line))); }
        if (name == "ceil")  { need_args(n, args, 1); return SkullValue(std::ceil(args[0].as_number(line))); }
        if (name == "round") { need_args(n, args, 1); return SkullValue(std::round(args[0].as_number(line))); }
        if (name == "pow")   { need_args(n, args, 2); return SkullValue(std::pow(args[0].as_number(line), args[1].as_number(line))); }
        if (name == "min")   { need_args(n, args, 2); return SkullValue(std::min(args[0].as_number(line), args[1].as_number(line))); }
        if (name == "max")   { need_args(n, args, 2); return SkullValue(std::max(args[0].as_number(line), args[1].as_number(line))); }
        if (name == "str")   { need_args(n, args, 1); return SkullValue(args[0].to_string()); }

        // ---- Listen und Text ----
        if (name == "len") {
            need_args(n, args, 1);
            if (args[0].kind == SkullValue::Kind::LIST)   return SkullValue((double)args[0].list->items.size());
            if (args[0].kind == SkullValue::Kind::STRING) return SkullValue((double)utf8_count(args[0].text));
            throw std::runtime_error(err(line, "len() braucht eine Liste oder einen Text"));
        }
        if (name == "push") {
            need_args(n, args, 2);
            if (args[0].kind != SkullValue::Kind::LIST)
                throw std::runtime_error(err(line, "push(): das erste Argument muss eine Liste sein"));
            list_add(*args[0].list, args[1], line);
            return SkullValue();
        }
        if (name == "pop") {
            need_args(n, args, 1);
            if (args[0].kind != SkullValue::Kind::LIST)
                throw std::runtime_error(err(line, "pop(): das Argument muss eine Liste sein"));
            auto& items = args[0].list->items;
            if (items.empty()) throw std::runtime_error(err(line, "pop(): die Liste ist leer"));
            SkullValue last = std::move(items.back());
            items.pop_back();
            return last;
        }
        if (name == "substr") {   // substr(text, start, anzahl): Zeichen ab start (0-basiert); zu grosse Anzahl wird gekuerzt
            need_args(n, args, 3);
            const std::string& txt = args[0].as_string(line);
            const auto off = utf8_offsets(txt);
            const double len = (double)(off.size() - 1);
            const double start = whole_number(args[1], line, "substr", "start");
            const double cnt   = whole_number(args[2], line, "substr", "anzahl");
            if (start < 0.0 || start > len)
                throw std::runtime_error(err(line, "substr(): start " + args[1].to_string() + " ausserhalb des Textes (Laenge " +
                                                   std::to_string(off.size() - 1) + ")"));
            if (cnt < 0.0) throw std::runtime_error(err(line, "substr(): anzahl darf nicht negativ sein"));
            const size_t a = (size_t)start;
            const size_t b = (size_t)std::min(start + cnt, len);
            return SkullValue(txt.substr(off[a], off[b] - off[a]));
        }
        if (name == "num") {      // Text -> Zahl (strikt: "12", "-3.5"; sonst Fehler)
            need_args(n, args, 1);
            if (args[0].kind == SkullValue::Kind::NUMBER) return args[0];
            const std::string& txt = args[0].as_string(line);
            double d;
            if (!parse_number(txt, d))
                throw std::runtime_error(err(line, "num(): '" + txt.substr(0, 40) + "' ist keine Zahl"));
            return SkullValue(d);
        }

        auto it = functions.find(name);
        if (it != functions.end()) {
            const SkullFunction& fn = it->second;
            if (args.size() != fn.params.size())
                throw std::runtime_error("Zeile " + std::to_string(line) + ": '" + name + "' erwartet " +
                                         std::to_string(fn.params.size()) + " Argument(e), bekommen: " +
                                         std::to_string(args.size()));
            DepthGuard guard(call_depth, line);
            auto fn_env = std::make_shared<Environment>(fn.closure);
            for (size_t i = 0; i < fn.params.size(); ++i) fn_env->define(fn.params[i], args[i]);
            try {
                for (const auto& s : *fn.body) exec_stmt(s.get(), fn_env);
            } catch (ReturnSignal& ret) {
                return ret.value;
            }
            return SkullValue();
        }
        throw std::runtime_error("Zeile " + std::to_string(line) + ": Unbekannte Funktion '" + name + "'");
    }

    // Felder eines model-Blocks auswerten (dim, vocab)
    template <typename Cfg>
    void apply_model_fields(const std::string& model_name, Cfg& cfg, std::shared_ptr<Environment> env) {
        auto it = models.find(model_name);
        if (it == models.end()) return;
        for (const auto& f : it->second->fields) {
            SkullValue val = eval_expr(f.value.get(), env);
            if (f.name == "dim")     cfg.dim     = to_size(val, f.line, "dim");
            if (f.name == "vocab")   cfg.vocab   = to_size(val, f.line, "vocab");
            if (f.name == "context") cfg.context = to_size(val, f.line, "context");
            if (f.name == "heads")   cfg.heads   = to_size(val, f.line, "heads");
            if (f.name == "layers")  cfg.layers  = to_size(val, f.line, "layers");
        }
    }

    void exec_stmt(const StmtNode* node, std::shared_ptr<Environment> env) {
        if (!node) return;
        if (auto* n = dynamic_cast<const AssignStmt*>(node)) {
            env->assign(n->name, eval_expr(n->value.get(), env));
            return;
        }
        if (auto* n = dynamic_cast<const ModelStmt*>(node)) {
            std::vector<std::pair<std::string, int>> names;
            for (const auto& f : n->fields) names.push_back({f.name, f.line});
            warn_fields("model", n->name, names, {"dim", "vocab", "context", "heads", "layers"}, {});
            models[n->name] = n;
            std::cout << "[Skull] Modell '" << n->name << "' definiert\n";
            return;
        }

        // ---- TRAIN ----
        if (auto* n = dynamic_cast<const TrainStmt*>(node)) {
            TrainConfig cfg;
            apply_model_fields(n->model_name, cfg, env);
            std::vector<std::pair<std::string, int>> names;
            for (const auto& f : n->fields) names.push_back({f.name, f.line});
            warn_fields("train", n->model_name, names,
                        {"data", "out", "epochs", "rate", "batch", "dim", "vocab", "steps",
                         "context", "heads", "layers", "val", "val_skip", "patience", "threads",
                         "seed", "checkpoint", "stop_after", "resume",
                         "bpe", "bpe_vocab", "gpu", "prefer_amd"}, {});
            for (const auto& f : n->fields) {
                SkullValue val = eval_expr(f.value.get(), env);
                if (f.name == "data")       cfg.data_path  = val.as_string(f.line);
                if (f.name == "out")        cfg.out_path   = val.as_string(f.line);
                if (f.name == "epochs")     cfg.epochs     = to_int(val, f.line, "epochs");
                if (f.name == "rate")       cfg.rate       = val.as_number(f.line);
                if (f.name == "batch")      cfg.batch      = to_int(val, f.line, "batch");
                if (f.name == "dim")        cfg.dim        = to_size(val, f.line, "dim");
                if (f.name == "vocab")      cfg.vocab      = to_size(val, f.line, "vocab");
                if (f.name == "steps")      cfg.steps      = to_size(val, f.line, "steps");
                if (f.name == "context")    cfg.context    = to_size(val, f.line, "context");
                if (f.name == "heads")      cfg.heads      = to_size(val, f.line, "heads");
                if (f.name == "layers")     cfg.layers     = to_size(val, f.line, "layers");
                if (f.name == "val")        cfg.val        = val.as_number(f.line);
                if (f.name == "val_skip")   cfg.val_skip   = to_size(val, f.line, "val_skip");
                if (f.name == "patience")   cfg.patience   = to_int(val, f.line, "patience");
                if (f.name == "threads")    cfg.threads    = to_int(val, f.line, "threads");
                if (f.name == "seed")       cfg.seed       = (unsigned)to_size(val, f.line, "seed");
                if (f.name == "checkpoint") cfg.checkpoint = to_int(val, f.line, "checkpoint");
                if (f.name == "stop_after") cfg.stop_after = to_int(val, f.line, "stop_after");
                if (f.name == "resume")     cfg.resume_path = val.as_string(f.line);
                if (f.name == "bpe")        cfg.use_bpe    = val.is_truthy();
                if (f.name == "bpe_vocab")  cfg.bpe_vocab  = to_int(val, f.line, "bpe_vocab");
                if (f.name == "gpu")        cfg.use_gpu    = val.is_truthy();
                if (f.name == "prefer_amd") cfg.prefer_amd = val.is_truthy();
            }
            try {
                last_train_ = skull_train(cfg);
                has_train_  = true;
            } catch (const std::runtime_error& e) {
                throw std::runtime_error("Zeile " + std::to_string(n->line) + ": " + e.what());
            }
            return;
        }

        // ---- GENERATE ----
        if (auto* n = dynamic_cast<const GenerateStmt*>(node)) {
            GenerateConfig cfg;
            apply_model_fields(n->model_name, cfg, env);
            std::vector<std::pair<std::string, int>> names;
            for (const auto& f : n->fields) names.push_back({f.name, f.line});
            warn_fields("generate", n->model_name, names,
                        {"weights", "prompt", "tokens", "temperature", "dim", "vocab",
                         "context", "heads", "layers", "shift", "seed", "top_k", "top_p"}, {});
            for (const auto& f : n->fields) {
                SkullValue val = eval_expr(f.value.get(), env);
                if (f.name == "weights")     cfg.weights_path = val.as_string(f.line);
                if (f.name == "prompt")      cfg.prompt       = val.as_string(f.line);
                if (f.name == "tokens")      cfg.tokens       = to_int(val, f.line, "tokens");
                if (f.name == "temperature") cfg.temperature  = val.as_number(f.line);
                if (f.name == "dim")         cfg.dim          = to_size(val, f.line, "dim");
                if (f.name == "vocab")       cfg.vocab        = to_size(val, f.line, "vocab");
                if (f.name == "context")     cfg.context      = to_size(val, f.line, "context");
                if (f.name == "heads")       cfg.heads        = to_size(val, f.line, "heads");
                if (f.name == "layers")      cfg.layers       = to_size(val, f.line, "layers");
                if (f.name == "shift")       cfg.shift        = val.is_truthy();
                if (f.name == "seed")        { cfg.seed = (unsigned)to_size(val, f.line, "seed"); cfg.has_seed = true; }
                if (f.name == "top_k")       cfg.top_k        = to_size(val, f.line, "top_k");
                if (f.name == "top_p")       cfg.top_p        = val.as_number(f.line);
            }
            try {
                skull_generate(cfg);
            } catch (const std::runtime_error& e) {
                throw std::runtime_error("Zeile " + std::to_string(n->line) + ": " + e.what());
            }
            return;
        }

        if (auto* n = dynamic_cast<const FuncStmt*>(node)) {
            SkullFunction fn;
            fn.name = n->name;
            fn.params = n->params;
            fn.body = const_cast<std::vector<std::unique_ptr<StmtNode>>*>(&n->body);
            fn.closure = env;
            functions[n->name] = std::move(fn);
            return;
        }
        if (auto* n = dynamic_cast<const ReturnStmt*>(node)) {
            SkullValue val;
            if (n->value) val = eval_expr(n->value.get(), env);
            throw ReturnSignal{val, n->line};
        }
        if (auto* n = dynamic_cast<const IfStmt*>(node)) {
            SkullValue cond = eval_expr(n->condition.get(), env);
            auto be = std::make_shared<Environment>(env);
            if (cond.is_truthy()) { for (const auto& s : n->then_body) exec_stmt(s.get(), be); }
            else                  { for (const auto& s : n->else_body) exec_stmt(s.get(), be); }
            return;
        }
        if (auto* n = dynamic_cast<const IndexAssignStmt*>(node)) {
            auto* target = static_cast<const IndexExpr*>(n->target.get());
            SkullValue val = eval_expr(n->value.get(), env);
            SkullValue obj = eval_expr(target->object.get(), env);
            SkullValue idx = eval_expr(target->index.get(), env);
            if (obj.kind == SkullValue::Kind::STRING)
                throw std::runtime_error(err(n->line, "Text laesst sich nicht aendern; baue einen neuen Text "
                                                      "(z. B. substr(...) + \"x\" + substr(...))"));
            if (obj.kind != SkullValue::Kind::LIST)
                throw std::runtime_error(err(n->line, "[ ] = geht nur bei Listen"));
            const size_t i = resolve_index(idx, obj.list->items.size(), n->line, "der Liste");
            if (val.kind == SkullValue::Kind::LIST && list_contains(val, obj.list.get()))
                throw std::runtime_error(err(n->line, "Eine Liste kann sich nicht selbst enthalten"));
            obj.list->items[i] = std::move(val);
            return;
        }
        if (auto* n = dynamic_cast<const ForEachStmt*>(node)) {
            SkullValue it = eval_expr(n->iterable.get(), env);
            if (it.kind == SkullValue::Kind::LIST) {
                // Die Anzahl der Durchlaeufe steht beim Start fest; push/pop in der Schleife machen sie nicht endlos
                const size_t count = it.list->items.size();
                for (size_t i = 0; i < count && i < it.list->items.size(); ++i) {
                    auto le = std::make_shared<Environment>(env);
                    le->define(n->var_name, it.list->items[i]);
                    for (const auto& s : n->body) exec_stmt(s.get(), le);
                }
            } else if (it.kind == SkullValue::Kind::STRING) {
                const std::string txt = it.text;
                for (size_t i = 0; i < txt.size(); ) {
                    const size_t len = utf8_char_len(txt, i);
                    auto le = std::make_shared<Environment>(env);
                    le->define(n->var_name, SkullValue(txt.substr(i, len)));
                    for (const auto& s : n->body) exec_stmt(s.get(), le);
                    i += len;
                }
            } else {
                throw std::runtime_error(err(n->line, "for ... in braucht eine Liste oder einen Text (Zahlenbereich: for i in 1..5)"));
            }
            return;
        }
        if (auto* n = dynamic_cast<const ForStmt*>(node)) {
            double start = eval_expr(n->start.get(), env).as_number(n->line);
            double end   = eval_expr(n->end.get(), env).as_number(n->line);
            if (!(std::fabs(start) <= 9e15) || !(std::fabs(end) <= 9e15))   // schliesst auch nan und inf aus
                throw std::runtime_error(err(n->line, "for: Anfang und Ende muessen endliche Zahlen (hoechstens 9e15) sein"));
            for (long long i = (long long)start; i <= (long long)end; ++i) {
                auto le = std::make_shared<Environment>(env);
                le->define(n->var_name, SkullValue((double)i));
                for (const auto& s : n->body) exec_stmt(s.get(), le);
            }
            return;
        }
        if (auto* n = dynamic_cast<const WhileStmt*>(node)) {
            int guard = 10000000;
            while (eval_expr(n->condition.get(), env).is_truthy()) {
                auto le = std::make_shared<Environment>(env);
                for (const auto& s : n->body) exec_stmt(s.get(), le);
                if (--guard <= 0) throw std::runtime_error("Zeile " + std::to_string(n->line) + ": while-Schleife zu lang");
            }
            return;
        }
        if (auto* n = dynamic_cast<const ExprStmt*>(node)) { eval_expr(n->expr.get(), env); return; }
        throw std::runtime_error("Zeile " + std::to_string(node->line) + ": Unbekannter Statement-Typ");
    }

public:
    // Die Grenzen sind einstellbar, damit Tests (Fuzzer) mit kleinen Werten laufen koennen.
    explicit Interpreter(size_t max_list = SKULL_MAX_LIST, size_t max_string = SKULL_MAX_STRING)
        : max_list_(max_list), max_string_(max_string) {
        global_env = std::make_shared<Environment>();
    }

    void run(const ProgramNode* program) {
        try {
            for (const auto& stmt : program->statements) exec_stmt(stmt.get(), global_env);
        } catch (const ReturnSignal& r) {
            throw std::runtime_error("Zeile " + std::to_string(r.line) +
                                     ": 'return' ist nur innerhalb einer Funktion erlaubt");
        }
    }
};
