#include <iostream>
#include <fstream>
#include <sstream>
#include "lexer.h"
#include "ast.h"
#include "parser.h"
#include "interpreter.h"

int main(int argc, char* argv[]) {
    std::cout << "=== Skull v0.7.0 ===\n\n";

    if (argc < 2) {
        std::cerr << "Verwendung: skull.exe <datei.skull>\n";
        std::cerr << "\nBeispiele:\n";
        std::cerr << "  skull.exe examples\\tokenizer_test.skull\n";
        std::cerr << "  skull.exe examples\\generate_demo.skull\n";
        std::cerr << "  skull.exe examples\\train_demo.skull\n";
        return 1;
    }

    std::ifstream file(argv[1]);
    if (!file.is_open()) {
        std::cerr << "FEHLER: Datei nicht gefunden: " << argv[1] << "\n";
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
