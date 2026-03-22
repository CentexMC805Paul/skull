#pragma once
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>
#include "lexer.h"
#include "ast.h"

// ============================================================
//  SKULL PARSER  v0.5.0
//  Neu: generate-Statement
// ============================================================

class Parser {
private:
    std::vector<Token> tokens;
    size_t             pos;

    Token& peek(int offset = 0) {
        size_t p = pos + offset;
        if (p >= tokens.size()) return tokens.back();
        return tokens[p];
    }
    Token advance() {
        Token t = tokens[pos];
        if (pos < tokens.size() - 1) pos++;
        return t;
    }
    Token expect(TokenKind kind) {
        Token t = peek();
        if (t.kind != kind)
            throw std::runtime_error(
                "Zeile " + std::to_string(t.line) +
                ", Spalte " + std::to_string(t.col) +
                ": Erwartet '" + token_kind_name(kind) +
                "', gefunden '" + t.value + "'"
            );
        return advance();
    }
    bool check(TokenKind kind) { return peek().kind == kind; }
    bool match(TokenKind kind) { if (check(kind)) { advance(); return true; } return false; }

    // ---- Ausdruecke ----
    std::unique_ptr<ExprNode> parse_factor() {
        Token t = peek();
        if (t.kind == TokenKind::NUMBER) {
            advance();
            return std::make_unique<NumberExpr>(std::stod(t.value), t.line, t.col);
        }
        if (t.kind == TokenKind::STRING) {
            advance();
            return std::make_unique<StringExpr>(t.value, t.line, t.col);
        }
        if (t.kind == TokenKind::KW_TRUE)  { advance(); return std::make_unique<BoolExpr>(true,  t.line, t.col); }
        if (t.kind == TokenKind::KW_FALSE) { advance(); return std::make_unique<BoolExpr>(false, t.line, t.col); }
        if (t.kind == TokenKind::IDENTIFIER) {
            advance();
            if (check(TokenKind::L_PAREN)) return parse_call(t);
            return std::make_unique<IdentExpr>(t.value, t.line, t.col);
        }
        if (t.kind == TokenKind::MINUS) {
            advance();
            if (check(TokenKind::NUMBER)) {
                double v = -std::stod(peek().value); advance();
                return std::make_unique<NumberExpr>(v, t.line, t.col);
            }
            auto zero  = std::make_unique<NumberExpr>(0.0, t.line, t.col);
            auto right = parse_factor();
            return std::make_unique<BinaryExpr>("-", std::move(zero), std::move(right), t.line, t.col);
        }
        if (t.kind == TokenKind::L_PAREN) {
            advance(); auto e = parse_expr(); expect(TokenKind::R_PAREN); return e;
        }
        throw std::runtime_error(
            "Zeile " + std::to_string(t.line) +
            ", Spalte " + std::to_string(t.col) +
            ": Unerwartetes Token '" + t.value + "'"
        );
    }

    std::unique_ptr<ExprNode> parse_call(Token name_tok) {
        expect(TokenKind::L_PAREN);
        std::vector<std::unique_ptr<ExprNode>> args;
        if (!check(TokenKind::R_PAREN)) {
            args.push_back(parse_expr());
            while (match(TokenKind::COMMA)) args.push_back(parse_expr());
        }
        expect(TokenKind::R_PAREN);
        return std::make_unique<CallExpr>(name_tok.value, std::move(args),
                                          name_tok.line, name_tok.col);
    }

    std::unique_ptr<ExprNode> parse_term() {
        auto node = parse_factor(); if (!node) return nullptr;
        while (check(TokenKind::STAR) || check(TokenKind::SLASH)) {
            Token op = advance(); auto right = parse_factor();
            if (!right) throw std::runtime_error("Ausdruck erwartet!");
            node = std::make_unique<BinaryExpr>(op.value, std::move(node),
                                                std::move(right), op.line, op.col);
        }
        return node;
    }

    std::unique_ptr<ExprNode> parse_addition() {
        auto node = parse_term(); if (!node) return nullptr;
        while (check(TokenKind::PLUS) || check(TokenKind::MINUS)) {
            Token op = advance(); auto right = parse_term();
            if (!right) throw std::runtime_error("Ausdruck erwartet!");
            node = std::make_unique<BinaryExpr>(op.value, std::move(node),
                                                std::move(right), op.line, op.col);
        }
        return node;
    }

    std::unique_ptr<ExprNode> parse_comparison() {
        auto node = parse_addition(); if (!node) return nullptr;
        while (check(TokenKind::LESS)    || check(TokenKind::GREATER) ||
               check(TokenKind::LESS_EQ) || check(TokenKind::GREATER_EQ) ||
               check(TokenKind::EQ_EQ)   || check(TokenKind::NOT_EQ)) {
            Token op = advance(); auto right = parse_addition();
            if (!right) throw std::runtime_error("Ausdruck erwartet!");
            node = std::make_unique<BinaryExpr>(op.value, std::move(node),
                                                std::move(right), op.line, op.col);
        }
        return node;
    }

    std::unique_ptr<ExprNode> parse_expr() { return parse_comparison(); }

    // ---- Bloecke ----
    std::vector<std::unique_ptr<StmtNode>> parse_block() {
        expect(TokenKind::L_BRACE);
        std::vector<std::unique_ptr<StmtNode>> stmts;
        while (!check(TokenKind::R_BRACE) && !check(TokenKind::END_OF_FILE))
            stmts.push_back(parse_stmt());
        expect(TokenKind::R_BRACE);
        return stmts;
    }

    // ---- Feld-Block: { name = expr  name = expr ... } ----
    template<typename FieldT>
    std::vector<FieldT> parse_fields() {
        expect(TokenKind::L_BRACE);
        std::vector<FieldT> fields;
        while (!check(TokenKind::R_BRACE) && !check(TokenKind::END_OF_FILE)) {
            Token fn = expect(TokenKind::IDENTIFIER);
            expect(TokenKind::EQUALS);
            auto fv = parse_expr();
            fields.push_back({ fn.value, std::move(fv), fn.line, fn.col });
        }
        expect(TokenKind::R_BRACE);
        return fields;
    }

    // ---- Statements ----
    std::unique_ptr<StmtNode> parse_define() {
        Token def_tok = advance();
        Token next = peek();

        if (next.kind == TokenKind::KW_MODEL) {
            advance();
            Token name = expect(TokenKind::IDENTIFIER);
            auto fields = parse_fields<ModelField>();
            return std::make_unique<ModelStmt>(name.value, std::move(fields),
                                               def_tok.line, def_tok.col);
        }
        if (next.kind == TokenKind::KW_FUNC) {
            advance();
            Token name = expect(TokenKind::IDENTIFIER);
            expect(TokenKind::L_PAREN);
            std::vector<std::string> params;
            if (!check(TokenKind::R_PAREN)) {
                params.push_back(expect(TokenKind::IDENTIFIER).value);
                while (match(TokenKind::COMMA))
                    params.push_back(expect(TokenKind::IDENTIFIER).value);
            }
            expect(TokenKind::R_PAREN);
            auto body = parse_block();
            return std::make_unique<FuncStmt>(name.value, std::move(params),
                                              std::move(body), def_tok.line, def_tok.col);
        }
        if (next.kind == TokenKind::IDENTIFIER) {
            Token name = advance();
            expect(TokenKind::EQUALS);
            auto val = parse_expr();
            return std::make_unique<AssignStmt>(name.value, std::move(val),
                                                def_tok.line, def_tok.col);
        }
        throw std::runtime_error(
            "Zeile " + std::to_string(next.line) +
            ": Nach 'define' erwartet: Name, 'model' oder 'func'"
        );
    }

    std::unique_ptr<StmtNode> parse_train() {
        Token tok = advance();
        Token model_name = expect(TokenKind::IDENTIFIER);
        auto fields = parse_fields<TrainField>();
        return std::make_unique<TrainStmt>(model_name.value, std::move(fields),
                                           tok.line, tok.col);
    }

    // generate MeinModell { weights = "..." prompt = "..." tokens = 100 }
    std::unique_ptr<StmtNode> parse_generate() {
        Token tok = advance();
        Token model_name = expect(TokenKind::IDENTIFIER);
        auto fields = parse_fields<GenerateField>();
        return std::make_unique<GenerateStmt>(model_name.value, std::move(fields),
                                              tok.line, tok.col);
    }

    std::unique_ptr<StmtNode> parse_if() {
        Token tok = advance();
        auto cond = parse_expr();
        auto then_body = parse_block();
        std::vector<std::unique_ptr<StmtNode>> else_body;
        if (check(TokenKind::KW_ELSE)) { advance(); else_body = parse_block(); }
        return std::make_unique<IfStmt>(std::move(cond), std::move(then_body),
                                        std::move(else_body), tok.line, tok.col);
    }

    std::unique_ptr<StmtNode> parse_for() {
        Token tok = advance();
        Token var = expect(TokenKind::IDENTIFIER);
        expect(TokenKind::KW_IN);
        auto start = parse_expr();
        expect(TokenKind::DOTDOT);
        auto end = parse_expr();
        auto body = parse_block();
        return std::make_unique<ForStmt>(var.value, std::move(start), std::move(end),
                                         std::move(body), tok.line, tok.col);
    }

    std::unique_ptr<StmtNode> parse_while() {
        Token tok = advance();
        auto cond = parse_expr();
        auto body = parse_block();
        return std::make_unique<WhileStmt>(std::move(cond), std::move(body),
                                           tok.line, tok.col);
    }

    std::unique_ptr<StmtNode> parse_return() {
        Token tok = advance();
        if (check(TokenKind::R_BRACE) || check(TokenKind::END_OF_FILE))
            return std::make_unique<ReturnStmt>(nullptr, tok.line, tok.col);
        auto val = parse_expr();
        return std::make_unique<ReturnStmt>(std::move(val), tok.line, tok.col);
    }

    std::unique_ptr<StmtNode> parse_stmt() {
        Token t = peek();

        if (t.kind == TokenKind::KW_DEFINE)   return parse_define();
        if (t.kind == TokenKind::KW_TRAIN)    return parse_train();
        if (t.kind == TokenKind::KW_GENERATE) return parse_generate();
        if (t.kind == TokenKind::KW_IF)       return parse_if();
        if (t.kind == TokenKind::KW_FOR)      return parse_for();
        if (t.kind == TokenKind::KW_WHILE)    return parse_while();
        if (t.kind == TokenKind::KW_RETURN)   return parse_return();

        // Re-Zuweisung: x = expr
        if (t.kind == TokenKind::IDENTIFIER && peek(1).kind == TokenKind::EQUALS) {
            Token name = advance(); advance();
            auto val = parse_expr();
            return std::make_unique<AssignStmt>(name.value, std::move(val),
                                                name.line, name.col);
        }

        auto expr = parse_expr();
        return std::make_unique<ExprStmt>(std::move(expr), t.line, t.col);
    }

public:
    explicit Parser(std::vector<Token> toks) : tokens(std::move(toks)), pos(0) {}

    std::unique_ptr<ProgramNode> parse() {
        auto program = std::make_unique<ProgramNode>();
        while (!check(TokenKind::END_OF_FILE))
            program->statements.push_back(parse_stmt());
        return program;
    }
};
