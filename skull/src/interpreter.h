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
#include "ast.h"
#include "tensor.h"
#include "trainer.h"
#include "generator.h"
#include "version.h"

// ============================================================
//  SKULL INTERPRETER
//  Fuehrt den AST direkt aus (Tree-Walking).
// ============================================================

struct SkullValue {
    enum class Kind { NUMBER, STRING, BOOL, TENSOR, NOTHING } kind;
    double number = 0.0;
    std::string text;
    bool flag = false;
    TensorPtr tensor;

    SkullValue() : kind(Kind::NOTHING) {}
    SkullValue(double v) : kind(Kind::NUMBER), number(v) {}
    SkullValue(const std::string& s) : kind(Kind::STRING), text(s) {}
    SkullValue(bool b) : kind(Kind::BOOL), flag(b) {}
    SkullValue(TensorPtr t) : kind(Kind::TENSOR), tensor(t) {}

    bool is_truthy() const {
        if (kind == Kind::NUMBER) return number != 0.0;
        if (kind == Kind::BOOL)   return flag;
        if (kind == Kind::STRING) return !text.empty();
        if (kind == Kind::TENSOR) return tensor != nullptr;
        return false;
    }
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
    std::string to_string() const {
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
        return "";
    }
    void print() const {
        if (kind == Kind::TENSOR && tensor) tensor->print();
        else std::cout << to_string();
    }
};

// Gleichheit mit Typ: 1 == "1" ist false. Zahl und Bool vergleichen als Zahl.
inline bool skull_values_equal(const SkullValue& a, const SkullValue& b) {
    using K = SkullValue::Kind;
    auto numeric = [](const SkullValue& v) { return v.kind == K::NUMBER || v.kind == K::BOOL; };
    if (numeric(a) && numeric(b)) return a.as_number() == b.as_number();
    if (a.kind != b.kind) return false;
    switch (a.kind) {
        case K::STRING:  return a.text == b.text;
        case K::TENSOR:  return a.tensor == b.tensor;   // gleiche Tensor-Instanz
        case K::NOTHING: return true;
        default:         return false;
    }
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

    SkullValue eval_expr(const ExprNode* node, std::shared_ptr<Environment> env) {
        if (!node) throw std::runtime_error("Leerer Ausdruck");
        if (auto* n = dynamic_cast<const NumberExpr*>(node)) return SkullValue(n->value);
        if (auto* n = dynamic_cast<const StringExpr*>(node)) return SkullValue(n->value);
        if (auto* n = dynamic_cast<const BoolExpr*>(node))   return SkullValue(n->value);
        if (auto* n = dynamic_cast<const IdentExpr*>(node))  return env->get(n->name, n->line);
        if (auto* n = dynamic_cast<const UnaryExpr*>(node))  return SkullValue(!eval_expr(n->operand.get(), env).is_truthy());
        if (auto* n = dynamic_cast<const BinaryExpr*>(node)) return eval_binary(n, env);
        if (auto* n = dynamic_cast<const CallExpr*>(node))   return eval_call(n, env);
        throw std::runtime_error("Unbekannter Ausdruck");
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
        if (op == "+" && (lv.kind == SkullValue::Kind::STRING || rv.kind == SkullValue::Kind::STRING))
            return SkullValue(lv.to_string() + rv.to_string());
        if (op == "==") return SkullValue(skull_values_equal(lv, rv));
        if (op == "!=") return SkullValue(!skull_values_equal(lv, rv));

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
                         "context", "heads", "layers",
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
                if (f.name == "bpe")        cfg.use_bpe    = val.is_truthy();
                if (f.name == "bpe_vocab")  cfg.bpe_vocab  = to_int(val, f.line, "bpe_vocab");
                if (f.name == "gpu")        cfg.use_gpu    = val.is_truthy();
                if (f.name == "prefer_amd") cfg.prefer_amd = val.is_truthy();
            }
            try {
                skull_train(cfg);
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
                         "context", "heads", "layers"}, {});
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
        if (auto* n = dynamic_cast<const ForStmt*>(node)) {
            double start = eval_expr(n->start.get(), env).as_number(n->line);
            double end   = eval_expr(n->end.get(), env).as_number(n->line);
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
    Interpreter() { global_env = std::make_shared<Environment>(); }

    void run(const ProgramNode* program) {
        try {
            for (const auto& stmt : program->statements) exec_stmt(stmt.get(), global_env);
        } catch (const ReturnSignal& r) {
            throw std::runtime_error("Zeile " + std::to_string(r.line) +
                                     ": 'return' ist nur innerhalb einer Funktion erlaubt");
        }
    }
};
