#include <iostream>
#include <fstream>
#include <sstream>
#include "lexer.h"
#include "ast.h"
#include "parser.h"
#include "interpreter.h"

int main(int argc, char* argv[]) {
    std::cout << "=== Skull v1.0.0 ===\n\n";

    if (argc < 2) {
        std::cerr << "Verwendung: skull <datei.skull>\n";
        std::cerr << "\nBeispiele:\n";
        std::cerr << "  skull examples\\hello.skull\n";
        std::cerr << "  skull examples\\train.skull\n";
        std::cerr << "  skull examples\\generate.skull\n";
        return 1;
    }

    std::ifstream file(argv[1]);
    if (!file.is_open()) {
        std::cerr << "FEHLER: Datei nicht gefunden: " << argv[1] << "\n";
        std::cerr << "Hinweis: Führe Skull im richtigen Verzeichnis aus.\n";
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
        std::cerr << "\n[FEHLER] " << e.what() << "\n";
        return 1;
    }
    return 0;
}
