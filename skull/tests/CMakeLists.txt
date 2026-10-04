# ============================================================
#  Skull-Tests (ctest)
#
#  Jeder Test fuehrt ein .skull-Skript aus und prueft Exit-Code und Ausgabe
#  (siehe run_case.cmake). Die Beispiele und Testdaten werden nach
#  build/testenv kopiert, damit die dabei entstehenden .weights-Dateien
#  den Quellbaum nicht veraendern.
#
#  Mehrere Regressionstests sichern Fehler ab, die frueher auftraten
#  (Speicherleck, 500-Token-Limit, falscher Autograd, Out-of-Bounds ...).
#  Das Speicherleck prueft autograd_no_leak ueber den Builtin live_tensors();
#  mit -DSKULL_SANITIZE=ON zusaetzlich AddressSanitizer/LeakSanitizer.
# ============================================================

# ---- C++-Test: Gradiententest des Transformers (endliche Differenzen) ----
add_executable(skull_gradcheck gradcheck.cpp)
target_include_directories(skull_gradcheck PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(skull_gradcheck PRIVATE skull_flags)
set_target_properties(skull_gradcheck PROPERTIES CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/tests"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/tests")
add_test(NAME transformer_gradcheck COMMAND skull_gradcheck)

# ---- C++-Test: Zufallszahlen muessen auf jedem System gleich sein (src/rng.h) ----
add_executable(skull_rngcheck rngcheck.cpp)
target_include_directories(skull_rngcheck PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(skull_rngcheck PRIVATE skull_flags)
set_target_properties(skull_rngcheck PROPERTIES CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/tests"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/tests")
add_test(NAME rng_platform_independent COMMAND skull_rngcheck)

set(SKULL_TESTENV "${CMAKE_BINARY_DIR}/testenv")
file(REMOVE_RECURSE "${SKULL_TESTENV}")
file(COPY "${CMAKE_SOURCE_DIR}/examples" DESTINATION "${SKULL_TESTENV}")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/cases" DESTINATION "${SKULL_TESTENV}")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/data"  DESTINATION "${SKULL_TESTENV}")

# skull_case(<name> <skript> [EXIT <code>] [MATCH <regex>] [NO_MATCH <regex>])
#   Regexe duerfen kein ';' enthalten (CMake-Listentrenner).
function(skull_case name script)
    cmake_parse_arguments(C "" "EXIT;MATCH;NO_MATCH" "" ${ARGN})
    if(NOT DEFINED C_EXIT)
        set(C_EXIT 0)
    endif()
    add_test(NAME ${name}
        COMMAND ${CMAKE_COMMAND}
            "-DSKULL=$<TARGET_FILE:skull>"
            "-DSCRIPT=${script}"
            "-DWORKDIR=${SKULL_TESTENV}"
            "-DEXPECT_EXIT=${C_EXIT}"
            "-DEXPECT_MATCH=${C_MATCH}"
            "-DEXPECT_NO_MATCH=${C_NO_MATCH}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/run_case.cmake")
endfunction()

# ---- Beispiele muessen laufen ----
skull_case(example_hello        examples/hello.skull        MATCH "quadrat\\(5\\) = 25")
skull_case(example_test         examples/test.skull         MATCH "Summe 1 bis 100 = 5050")
skull_case(example_tensor_test  examples/tensor_test.skull  MATCH "Training abgeschlossen!")
skull_case(example_train_demo   examples/train_demo.skull   MATCH "Training fertig!" NO_MATCH "WARNUNG")
skull_case(example_train        examples/train.skull        MATCH "Training abgeschlossen!")
skull_case(example_generate     examples/generate.skull     MATCH "--- Ende ---")
skull_case(example_generate_demo examples/generate_demo.skull MATCH "--- Ende ---")
skull_case(example_tokenizer    examples/tokenizer_test.skull MATCH "Test 1 abgeschlossen!")
skull_case(example_formats      examples/format_test.skull  MATCH "Alle Format-Tests abgeschlossen!")
skull_case(example_transformer  examples/transformer.skull  MATCH "Transformer \\(2 Schicht.*--- Ende ---")

# Mehrere Beispiele schreiben dieselbe Gewichte-Datei (examples/training_data.txt.weights);
# sie duerfen deshalb bei "ctest -j" nicht gleichzeitig laufen.
set_tests_properties(example_test example_train_demo example_train example_generate
                     example_generate_demo example_tokenizer
                     PROPERTIES RESOURCE_LOCK examples_weights)

# ---- Sprache ----
skull_case(lang_return_toplevel     cases/lang_return_toplevel.skull     EXIT 1 MATCH "nur innerhalb einer Funktion")
skull_case(lang_recursion_limit     cases/lang_recursion_limit.skull     EXIT 1 MATCH "Rekursionstiefe ueberschritten")
skull_case(lang_recursion_ok        cases/lang_recursion_ok.skull        MATCH "SUM: 20100")
skull_case(lang_equality            cases/lang_equality.skull            MATCH "EQ: false true true true")
skull_case(lang_unknown_field       cases/lang_unknown_field.skull       MATCH "Unbekanntes Feld 'epoch'")
skull_case(lang_type_error          cases/lang_type_error.skull          EXIT 1 MATCH "Zeile 3: Text .* erwartet")
skull_case(lang_else_if             cases/lang_else_if.skull             MATCH "KLASSE: negativ null klein gross")
skull_case(lang_logic               cases/lang_logic.skull               MATCH "LOGIK: false true false true true true.*KURZ1 ok.*KURZ2 ok.*RANG: true false.*NE: true false")
skull_case(lang_modulo              cases/lang_modulo.skull              MATCH "MOD: 1 2 -2 0 1.5.*DREI: 5")
skull_case(lang_modulo_zero         cases/lang_modulo_zero.skull         EXIT 1 MATCH "Zeile 1: Modulo durch 0")

# ---- Autograd / Speicher ----
skull_case(autograd_diamond         cases/autograd_diamond.skull         MATCH "GRAD OK" NO_MATCH "GRAD FALSCH")
skull_case(autograd_no_leak         cases/autograd_no_leak.skull         MATCH "LEAK-TEST OK" NO_MATCH "LEAK-TEST FEHLER")

# ---- Training ----
skull_case(train_missing_file       cases/train_missing_file.skull       EXIT 1 MATCH "Datei nicht gefunden")
skull_case(train_no_extension       cases/train_no_extension.skull       MATCH "Format erkannt: TXT")
skull_case(train_empty_file         cases/train_empty_file.skull         EXIT 1 MATCH "Kein Text extrahiert")
skull_case(train_one_token          cases/train_one_token.skull          EXIT 1 MATCH "mindestens 2 Tokens")
skull_case(train_all_data           cases/train_all_data.skull           MATCH "xyxyxyxyxyxyxyxy")
skull_case(train_batch              cases/train_batch.skull              MATCH "Batch:    8.*xyxyxyxyxyxyxyxy")
skull_case(train_steps_window       cases/train_steps_window.skull       MATCH "Schritte/Epoche: 100")

# ---- Generator ----
skull_case(gen_small_vocab          cases/gen_small_vocab.skull          MATCH "--- Ende ---")
skull_case(gen_temperature_zero     cases/gen_temperature_zero.skull     MATCH "greedy.*das [a-z]")
skull_case(gen_missing_weights      cases/gen_missing_weights.skull      EXIT 1 MATCH "Gewichte nicht gefunden")
skull_case(gen_truncated_weights    cases/gen_truncated_weights.skull    EXIT 1 MATCH "abgeschnitten oder beschaedigt")
skull_case(gen_garbage_weights      cases/gen_garbage_weights.skull      EXIT 1 MATCH "unplausible Groessen")
skull_case(gen_not_weights          cases/gen_not_weights.skull          EXIT 1 MATCH "Magic .* fehlt")
skull_case(bpe_roundtrip            cases/bpe_roundtrip.skull            MATCH "BPE-Tokenizer.*--- Ende ---")

# ---- Transformer (Attention) ----
skull_case(tf_learns_context        cases/tf_learns_context.skull        MATCH "Modell:   Transformer.*das ist ein test\\.")
skull_case(bigram_control           cases/bigram_control.skull           MATCH "Modell:   Bigram" NO_MATCH "ein test")
skull_case(tf_bpe                   cases/tf_bpe.skull                   MATCH "Transformer.*BPE-Tokenizer.*--- Ende ---")
skull_case(tf_bad_heads             cases/tf_bad_heads.skull             EXIT 1 MATCH "durch heads")
skull_case(tf_context_clip          cases/tf_context_clip.skull          MATCH "context wurde auf 245 reduziert.*--- Ende ---")
skull_case(tf_unused_fields         cases/tf_unused_fields.skull         MATCH "WARNUNG.*'heads' und 'layers' wirken nur mit context > 1")
skull_case(tf_context_mismatch      cases/tf_context_mismatch.skull      MATCH "WARNUNG. Modell definiert context=4.*heads=1.*layers=3")
skull_case(tf_truncated_weights     cases/tf_truncated_weights.skull     EXIT 1 MATCH "abgeschnitten oder beschaedigt")
skull_case(tf_batch_steps           cases/tf_batch_steps.skull           MATCH "Sequenzen/Epoche: 20")
