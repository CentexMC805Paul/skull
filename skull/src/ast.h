#pragma once
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include "lexer.h"

// ============================================================
//  SKULL AST
// ============================================================

struct Node {
    int line = 0;
    int col  = 0;
    virtual ~Node() = default;
};

// --- Ausdruecke ---
struct ExprNode : Node { virtual ~ExprNode() = default; };

struct NumberExpr : ExprNode {
    double value;
    NumberExpr(double v, int l, int c) : value(v) { line=l; col=c; }
};
struct StringExpr : ExprNode {
    std::string value;
    StringExpr(std::string v, int l, int c) : value(std::move(v)) { line=l; col=c; }
};
struct BoolExpr : ExprNode {
    bool value;
    BoolExpr(bool v, int l, int c) : value(v) { line=l; col=c; }
};
struct IdentExpr : ExprNode {
    std::string name;
    IdentExpr(std::string n, int l, int c) : name(std::move(n)) { line=l; col=c; }
};
struct BinaryExpr : ExprNode {
    std::string op;
    std::unique_ptr<ExprNode> left, right;
    BinaryExpr(std::string o, std::unique_ptr<ExprNode> l,
               std::unique_ptr<ExprNode> r, int ln, int c)
        : op(std::move(o)), left(std::move(l)), right(std::move(r)) { line=ln; col=c; }
};
struct UnaryExpr : ExprNode {
    std::string op;                       // "not"
    std::unique_ptr<ExprNode> operand;
    UnaryExpr(std::string o, std::unique_ptr<ExprNode> e, int l, int c)
        : op(std::move(o)), operand(std::move(e)) { line=l; col=c; }
};
// [a, b, c]
struct ListExpr : ExprNode {
    std::vector<std::unique_ptr<ExprNode>> items;
    ListExpr(std::vector<std::unique_ptr<ExprNode>> it, int l, int c) : items(std::move(it)) { line=l; col=c; }
};
// liste[i], text[i]
struct IndexExpr : ExprNode {
    std::unique_ptr<ExprNode> object, index;
    IndexExpr(std::unique_ptr<ExprNode> o, std::unique_ptr<ExprNode> i, int l, int c)
        : object(std::move(o)), index(std::move(i)) { line=l; col=c; }
};
struct CallExpr : ExprNode {
    std::string name;
    std::vector<std::unique_ptr<ExprNode>> args;
    CallExpr(std::string n, std::vector<std::unique_ptr<ExprNode>> a, int l, int c)
        : name(std::move(n)), args(std::move(a)) { line=l; col=c; }
};

// --- Statements ---
struct StmtNode : Node { virtual ~StmtNode() = default; };

struct AssignStmt : StmtNode {
    std::string name;
    std::unique_ptr<ExprNode> value;
    AssignStmt(std::string n, std::unique_ptr<ExprNode> v, int l, int c)
        : name(std::move(n)), value(std::move(v)) { line=l; col=c; }
};

struct ModelField {
    std::string name;
    std::unique_ptr<ExprNode> value;
    int line, col;
};
struct ModelStmt : StmtNode {
    std::string name;
    std::vector<ModelField> fields;
    ModelStmt(std::string n, std::vector<ModelField> f, int l, int c)
        : name(std::move(n)), fields(std::move(f)) { line=l; col=c; }
};

struct TrainField {
    std::string name;
    std::unique_ptr<ExprNode> value;
    int line, col;
};
struct TrainStmt : StmtNode {
    std::string model_name;
    std::vector<TrainField> fields;
    TrainStmt(std::string m, std::vector<TrainField> f, int l, int c)
        : model_name(std::move(m)), fields(std::move(f)) { line=l; col=c; }
};

// generate MeinModell { weights = "..." prompt = "..." tokens = 100 }
struct GenerateField {
    std::string name;
    std::unique_ptr<ExprNode> value;
    int line, col;
};
struct GenerateStmt : StmtNode {
    std::string model_name;
    std::vector<GenerateField> fields;
    GenerateStmt(std::string m, std::vector<GenerateField> f, int l, int c)
        : model_name(std::move(m)), fields(std::move(f)) { line=l; col=c; }
};

struct FuncStmt : StmtNode {
    std::string name;
    std::vector<std::string> params;
    std::vector<std::unique_ptr<StmtNode>> body;
    FuncStmt(std::string n, std::vector<std::string> p,
             std::vector<std::unique_ptr<StmtNode>> b, int l, int c)
        : name(std::move(n)), params(std::move(p)), body(std::move(b)) { line=l; col=c; }
};
struct ReturnStmt : StmtNode {
    std::unique_ptr<ExprNode> value;
    ReturnStmt(std::unique_ptr<ExprNode> v, int l, int c)
        : value(std::move(v)) { line=l; col=c; }
};
struct IfStmt : StmtNode {
    std::unique_ptr<ExprNode> condition;
    std::vector<std::unique_ptr<StmtNode>> then_body, else_body;
    IfStmt(std::unique_ptr<ExprNode> cond,
           std::vector<std::unique_ptr<StmtNode>> then_b,
           std::vector<std::unique_ptr<StmtNode>> else_b, int l, int c)
        : condition(std::move(cond)), then_body(std::move(then_b)),
          else_body(std::move(else_b)) { line=l; col=c; }
};
struct ForStmt : StmtNode {
    std::string var_name;
    std::unique_ptr<ExprNode> start, end;
    std::vector<std::unique_ptr<StmtNode>> body;
    ForStmt(std::string v, std::unique_ptr<ExprNode> s,
            std::unique_ptr<ExprNode> e,
            std::vector<std::unique_ptr<StmtNode>> b, int l, int c)
        : var_name(std::move(v)), start(std::move(s)), end(std::move(e)),
          body(std::move(b)) { line=l; col=c; }
};
struct WhileStmt : StmtNode {
    std::unique_ptr<ExprNode> condition;
    std::vector<std::unique_ptr<StmtNode>> body;
    WhileStmt(std::unique_ptr<ExprNode> cond,
              std::vector<std::unique_ptr<StmtNode>> b, int l, int c)
        : condition(std::move(cond)), body(std::move(b)) { line=l; col=c; }
};
// liste[i] = wert
struct IndexAssignStmt : StmtNode {
    std::unique_ptr<ExprNode> target;   // immer ein IndexExpr
    std::unique_ptr<ExprNode> value;
    IndexAssignStmt(std::unique_ptr<ExprNode> t, std::unique_ptr<ExprNode> v, int l, int c)
        : target(std::move(t)), value(std::move(v)) { line=l; col=c; }
};
// for x in liste { ... }
struct ForEachStmt : StmtNode {
    std::string var_name;
    std::unique_ptr<ExprNode> iterable;
    std::vector<std::unique_ptr<StmtNode>> body;
    ForEachStmt(std::string v, std::unique_ptr<ExprNode> it, std::vector<std::unique_ptr<StmtNode>> b, int l, int c)
        : var_name(std::move(v)), iterable(std::move(it)), body(std::move(b)) { line=l; col=c; }
};
struct ExprStmt : StmtNode {
    std::unique_ptr<ExprNode> expr;
    ExprStmt(std::unique_ptr<ExprNode> e, int l, int c)
        : expr(std::move(e)) { line=l; col=c; }
};

struct ProgramNode : Node {
    std::vector<std::unique_ptr<StmtNode>> statements;
};
