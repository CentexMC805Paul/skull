#pragma once
#include <string>
#include <vector>

// ============================================================
//  SKULL LEXER
// ============================================================

enum class TokenKind {
    NUMBER, STRING, IDENTIFIER,

    // Keywords
    KW_DEFINE, KW_MODEL, KW_TRAIN, KW_GENERATE,
    KW_FUNC, KW_RETURN,
    KW_IF, KW_ELSE, KW_FOR, KW_IN, KW_WHILE,
    KW_TRUE, KW_FALSE,
    KW_AND, KW_OR, KW_NOT,          // and, or, not  (und '!')

    // Operatoren
    EQUALS, PLUS, MINUS, STAR, SLASH, PERCENT,
    LESS, GREATER, LESS_EQ, GREATER_EQ, EQ_EQ, NOT_EQ,
    DOTDOT,

    // Klammern & Trennzeichen
    L_BRACE, R_BRACE,
    L_PAREN, R_PAREN,
    L_BRACKET, R_BRACKET,
    COMMA, DOT, SEMICOLON,

    END_OF_FILE, UNKNOWN
};

inline std::string token_kind_name(TokenKind k) {
    switch (k) {
        case TokenKind::NUMBER:       return "Zahl";
        case TokenKind::STRING:       return "String";
        case TokenKind::IDENTIFIER:   return "Name";
        case TokenKind::KW_DEFINE:    return "define";
        case TokenKind::KW_MODEL:     return "model";
        case TokenKind::KW_TRAIN:     return "train";
        case TokenKind::KW_GENERATE:  return "generate";
        case TokenKind::KW_FUNC:      return "func";
        case TokenKind::KW_RETURN:    return "return";
        case TokenKind::KW_IF:        return "if";
        case TokenKind::KW_ELSE:      return "else";
        case TokenKind::KW_FOR:       return "for";
        case TokenKind::KW_IN:        return "in";
        case TokenKind::KW_WHILE:     return "while";
        case TokenKind::KW_TRUE:      return "true";
        case TokenKind::KW_FALSE:     return "false";
        case TokenKind::KW_AND:       return "and";
        case TokenKind::KW_OR:        return "or";
        case TokenKind::KW_NOT:       return "not";
        case TokenKind::EQUALS:       return "=";
        case TokenKind::PLUS:         return "+";
        case TokenKind::MINUS:        return "-";
        case TokenKind::STAR:         return "*";
        case TokenKind::SLASH:        return "/";
        case TokenKind::PERCENT:      return "%";
        case TokenKind::LESS:         return "<";
        case TokenKind::GREATER:      return ">";
        case TokenKind::LESS_EQ:      return "<=";
        case TokenKind::GREATER_EQ:   return ">=";
        case TokenKind::EQ_EQ:        return "==";
        case TokenKind::NOT_EQ:       return "!=";
        case TokenKind::DOTDOT:       return "..";
        case TokenKind::L_BRACE:      return "{";
        case TokenKind::R_BRACE:      return "}";
        case TokenKind::L_PAREN:      return "(";
        case TokenKind::R_PAREN:      return ")";
        case TokenKind::L_BRACKET:    return "[";
        case TokenKind::R_BRACKET:    return "]";
        case TokenKind::COMMA:        return ",";
        case TokenKind::DOT:          return ".";
        case TokenKind::SEMICOLON:    return ";";
        case TokenKind::END_OF_FILE:  return "Ende";
        default:                      return "?";
    }
}

struct Token {
    TokenKind   kind;
    std::string value;
    int         line;
    int         col;
};

class Lexer {
private:
    std::string source;
    size_t      pos;
    int         line;
    int         col;

    bool is_alpha(char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
    }

    char peek(int offset = 0) {
        size_t p = pos + offset;
        return (p < source.size()) ? source[p] : '\0';
    }

    char advance() {
        char c = source[pos++];
        if (c == '\n') { line++; col = 1; } else { col++; }
        return c;
    }

    void skip_whitespace_and_comments() {
        while (pos < source.size()) {
            char c = peek();
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') { advance(); continue; }
            if (c == '/' && peek(1) == '/') {
                while (pos < source.size() && peek() != '\n') advance();
                continue;
            }
            if (c == '/' && peek(1) == '*') {
                advance(); advance();
                while (pos < source.size()) {
                    if (peek() == '*' && peek(1) == '/') { advance(); advance(); break; }
                    advance();
                }
                continue;
            }
            break;
        }
    }

    Token read_number(int sl, int sc) {
        std::string num;
        bool has_dot = false;
        while (pos < source.size() && (isdigit(peek()) || peek() == '.')) {
            if (peek() == '.' && peek(1) == '.') break;
            if (peek() == '.') { if (has_dot) break; has_dot = true; }
            num += advance();
        }
        return { TokenKind::NUMBER, num, sl, sc };
    }

    Token read_identifier_or_keyword(int sl, int sc) {
        std::string id;
        while (pos < source.size() && (is_alpha(peek()) || isdigit(peek())))
            id += advance();

        if (id == "define")   return { TokenKind::KW_DEFINE,   id, sl, sc };
        if (id == "model")    return { TokenKind::KW_MODEL,    id, sl, sc };
        if (id == "train")    return { TokenKind::KW_TRAIN,    id, sl, sc };
        if (id == "generate") return { TokenKind::KW_GENERATE, id, sl, sc };
        if (id == "func")     return { TokenKind::KW_FUNC,     id, sl, sc };
        if (id == "return")   return { TokenKind::KW_RETURN,   id, sl, sc };
        if (id == "if")       return { TokenKind::KW_IF,       id, sl, sc };
        if (id == "else")     return { TokenKind::KW_ELSE,     id, sl, sc };
        if (id == "for")      return { TokenKind::KW_FOR,      id, sl, sc };
        if (id == "in")       return { TokenKind::KW_IN,       id, sl, sc };
        if (id == "while")    return { TokenKind::KW_WHILE,    id, sl, sc };
        if (id == "true")     return { TokenKind::KW_TRUE,     id, sl, sc };
        if (id == "false")    return { TokenKind::KW_FALSE,    id, sl, sc };
        if (id == "and")      return { TokenKind::KW_AND,      id, sl, sc };
        if (id == "or")       return { TokenKind::KW_OR,       id, sl, sc };
        if (id == "not")      return { TokenKind::KW_NOT,      id, sl, sc };
        return { TokenKind::IDENTIFIER, id, sl, sc };
    }

    Token read_string(int sl, int sc) {
        advance(); // oeffnendes "
        std::string str;
        while (pos < source.size() && peek() != '"') {
            if (peek() == '\\') {
                advance();
                char esc = advance();
                switch (esc) {
                    case 'n': str += '\n'; break;
                    case 't': str += '\t'; break;
                    case '"': str += '"';  break;
                    case '\\': str += '\\'; break;
                    default:  str += esc;  break;
                }
            } else {
                str += advance();
            }
        }
        if (pos < source.size()) advance(); // schliessendes "
        return { TokenKind::STRING, str, sl, sc };
    }

public:
    Lexer(const std::string& src) : source(src), pos(0), line(1), col(1) {}

    Token next() {
        skip_whitespace_and_comments();
        if (pos >= source.size()) return { TokenKind::END_OF_FILE, "", line, col };

        int sl = line, sc = col;
        char c = peek();

        if (isdigit(c))  return read_number(sl, sc);
        if (is_alpha(c)) return read_identifier_or_keyword(sl, sc);
        if (c == '"')    return read_string(sl, sc);

        // Zwei-Zeichen-Tokens
        if (c == '<' && peek(1) == '=') { advance(); advance(); return { TokenKind::LESS_EQ,    "<=", sl, sc }; }
        if (c == '>' && peek(1) == '=') { advance(); advance(); return { TokenKind::GREATER_EQ, ">=", sl, sc }; }
        if (c == '=' && peek(1) == '=') { advance(); advance(); return { TokenKind::EQ_EQ,      "==", sl, sc }; }
        if (c == '!' && peek(1) == '=') { advance(); advance(); return { TokenKind::NOT_EQ,     "!=", sl, sc }; }
        if (c == '.' && peek(1) == '.') { advance(); advance(); return { TokenKind::DOTDOT,     "..", sl, sc }; }

        advance();
        switch (c) {
            case '=': return { TokenKind::EQUALS,    "=",  sl, sc };
            case '+': return { TokenKind::PLUS,      "+",  sl, sc };
            case '-': return { TokenKind::MINUS,     "-",  sl, sc };
            case '*': return { TokenKind::STAR,      "*",  sl, sc };
            case '/': return { TokenKind::SLASH,     "/",  sl, sc };
            case '%': return { TokenKind::PERCENT,   "%",  sl, sc };
            case '!': return { TokenKind::KW_NOT,    "!",  sl, sc };   // '!=' wurde oben schon erkannt
            case '<': return { TokenKind::LESS,      "<",  sl, sc };
            case '>': return { TokenKind::GREATER,   ">",  sl, sc };
            case '{': return { TokenKind::L_BRACE,   "{",  sl, sc };
            case '}': return { TokenKind::R_BRACE,   "}",  sl, sc };
            case '(': return { TokenKind::L_PAREN,   "(",  sl, sc };
            case ')': return { TokenKind::R_PAREN,   ")",  sl, sc };
            case '[': return { TokenKind::L_BRACKET, "[",  sl, sc };
            case ']': return { TokenKind::R_BRACKET, "]",  sl, sc };
            case ',': return { TokenKind::COMMA,     ",",  sl, sc };
            case '.': return { TokenKind::DOT,       ".",  sl, sc };
            case ';': return { TokenKind::SEMICOLON, ";",  sl, sc };
            default:  return { TokenKind::UNKNOWN, std::string(1, c), sl, sc };
        }
    }

    std::vector<Token> tokenize() {
        std::vector<Token> tokens;
        while (true) {
            Token t = next();
            tokens.push_back(t);
            if (t.kind == TokenKind::END_OF_FILE) break;
        }
        return tokens;
    }
};
