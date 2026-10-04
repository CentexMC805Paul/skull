#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "version.h"
#include "stack.h"
#include "lexer.h"
#include "ast.h"
#include "parser.h"
#include "interpreter.h"

// Stack fuer den Interpreter-Thread (siehe stack.h). Nur reservierter
// Adressraum; belegt wird nur, was die Rekursion wirklich braucht.
static constexpr std::size_t INTERPRETER_STACK_BYTES = 512u * 1024u * 1024u;

static int run_file(const char* path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "FEHLER: Datei nicht gefunden: " << path << "\n";
        std::cerr << "Hinweis: Fuehre Skull im richtigen Verzeichnis aus.\n";
        std::cerr << "Beispiel: cd skull && ./build/skull examples/hello.skull\n";
        return 1;
    }
    std::stringstream buf;
    buf << file.rdbuf();

    try {
        Lexer  lexer(buf.str());
        Parser parser(lexer.tokenize());
        auto   program = parser.parse();
        Interpreter interp;
        interp.run(program.get());
    } catch (const std::exception& e) {
        std::cout.flush();
        std::cerr << "\n[FEHLER] " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cout.flush();
        std::cerr << "\n[FEHLER] Unbekannter interner Fehler\n";
        return 1;
    }
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc >= 2 && (std::string(argv[1]) == "--version" || std::string(argv[1]) == "-v")) {
        std::cout << "Skull v" SKULL_VERSION "\n";
        return 0;
    }

    std::cout << "=== Skull v" SKULL_VERSION " ===\n\n";

    if (argc < 2) {
        std::cerr << "Verwendung: skull <datei.skull>\n";
        std::cerr << "\nBeispiele:\n";
        std::cerr << "  skull examples/hello.skull\n";
        std::cerr << "  skull examples/train.skull\n";
        std::cerr << "  skull examples/generate.skull\n";
        return 1;
    }

    const char* path = argv[1];
    return run_with_stack(INTERPRETER_STACK_BYTES, [path]() { return run_file(path); });
}
