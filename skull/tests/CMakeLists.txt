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

# ---- C++-Tests (eigene Executables, gleiche Compiler-Flags wie das Programm) ----
# skull_cpp_test(<test-name> <quelle.cpp> [WORKING_DIRECTORY <dir>])
function(skull_cpp_test name source)
    cmake_parse_arguments(T "" "WORKING_DIRECTORY" "" ${ARGN})
    get_filename_component(exe "${source}" NAME_WE)
    add_executable(skull_${exe} ${source})
    target_include_directories(skull_${exe} PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_link_libraries(skull_${exe} PRIVATE skull_flags)
    set_target_properties(skull_${exe} PROPERTIES CXX_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/tests"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/tests")
    if(T_WORKING_DIRECTORY)
        add_test(NAME ${name} COMMAND skull_${exe} WORKING_DIRECTORY "${T_WORKING_DIRECTORY}")
    else()
        add_test(NAME ${name} COMMAND skull_${exe})
    endif()
endfunction()

# Gradiententest des Transformers (endliche Differenzen), Kausalitaet, Adam
skull_cpp_test(transformer_gradcheck gradcheck.cpp)
# Zufallszahlen muessen auf jedem System gleich sein (src/rng.h)
skull_cpp_test(rng_platform_independent rngcheck.cpp)
# Trainingsrahmen: bestes Validierungs-Modell, Early Stopping, Datenaufteilung, Checkpoints
skull_cpp_test(training_framework traincheck.cpp WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
# Sampling: seed, top_k, top_p
skull_cpp_test(generator_sampling gencheck.cpp WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")

# ---- C++-Test: BPE-Tokenizer (src/tokenizer.h) gegen eine naive Referenz ----
add_executable(skull_bpecheck bpecheck.cpp)
target_include_directories(skull_bpecheck PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(skull_bpecheck PRIVATE skull_flags)
set_target_properties(skull_bpecheck PROPERTIES CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/tests"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/tests")
add_test(NAME bpe_matches_reference COMMAND skull_bpecheck)

set(SKULL_TESTENV "${CMAKE_BINARY_DIR}/testenv")
file(REMOVE_RECURSE "${SKULL_TESTENV}")
file(COPY "${CMAKE_SOURCE_DIR}/examples" DESTINATION "${SKULL_TESTENV}")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/cases" DESTINATION "${SKULL_TESTENV}")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/data"  DESTINATION "${SKULL_TESTENV}")

# Fuzzing (Parser), generierte Programme, Modelltest fuer Listen/Text, Grenzen.
# Laeuft in build/testenv, dort liegen examples/ und cases/ als Ausgangsmaterial.
skull_cpp_test(language_fuzz_and_model fuzz.cpp WORKING_DIRECTORY "${SKULL_TESTENV}")

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

# Listen und Text
skull_case(lang_lists               cases/lang_lists.skull               MATCH "LISTEN OK" NO_MATCH "FALSCH")
skull_case(lang_text                cases/lang_text.skull                MATCH "TEXT OK" NO_MATCH "FALSCH")
skull_case(lang_list_index_range    cases/lang_list_index_range.skull    EXIT 1 MATCH "Zeile 2: Index 2 ausserhalb der Liste .Laenge 2, erlaubt 0..1 oder -1..-2.")
skull_case(lang_list_index_negative cases/lang_list_index_negative.skull EXIT 1 MATCH "Zeile 2: Index -3 ausserhalb der Liste")
skull_case(lang_list_index_empty    cases/lang_list_index_empty.skull    EXIT 1 MATCH "ausserhalb der Liste .leer.")
skull_case(lang_list_index_frac     cases/lang_list_index_frac.skull     EXIT 1 MATCH "Index muss eine ganze Zahl sein .bekommen: 0.5.")
skull_case(lang_list_pop_empty      cases/lang_list_pop_empty.skull      EXIT 1 MATCH "pop..: die Liste ist leer")
skull_case(lang_list_cycle          cases/lang_list_cycle.skull          EXIT 1 MATCH "Zeile 2: Eine Liste kann sich nicht selbst enthalten")
skull_case(lang_list_cycle_indirect cases/lang_list_cycle_indirect.skull EXIT 1 MATCH "Zeile 3: Eine Liste kann sich nicht selbst enthalten")
skull_case(lang_list_plus_number    cases/lang_list_plus_number.skull    EXIT 1 MATCH "Eine Liste laesst sich nur mit einer Liste oder einem Text addieren")
skull_case(lang_text_immutable      cases/lang_text_immutable.skull      EXIT 1 MATCH "Text laesst sich nicht aendern")
skull_case(lang_text_index_range    cases/lang_text_index_range.skull    EXIT 1 MATCH "Index 3 ausserhalb des Textes")
skull_case(lang_num_bad             cases/lang_num_bad.skull             EXIT 1 MATCH "num..: '12abc' ist keine Zahl")
skull_case(lang_num_nan             cases/lang_num_nan.skull             EXIT 1 MATCH "num..: 'nan' ist keine Zahl")
skull_case(lang_num_empty           cases/lang_num_empty.skull           EXIT 1 MATCH "num..: '' ist keine Zahl")
skull_case(lang_substr_range        cases/lang_substr_range.skull        EXIT 1 MATCH "substr..: start 4 ausserhalb des Textes .Laenge 3.")
skull_case(lang_len_number          cases/lang_len_number.skull          EXIT 1 MATCH "len.. braucht eine Liste oder einen Text")
skull_case(lang_index_number        cases/lang_index_number.skull        EXIT 1 MATCH "Zeile 2: . . geht nur bei Listen und Text")
skull_case(lang_foreach_number      cases/lang_foreach_number.skull      EXIT 1 MATCH "for ... in braucht eine Liste oder einen Text")
skull_case(lang_for_huge            cases/lang_for_huge.skull            EXIT 1 MATCH "for: Anfang und Ende muessen endliche Zahlen")

# Parser: Fehlermeldungen statt Absturz
skull_case(lang_assign_target       cases/lang_assign_target.skull       EXIT 1 MATCH "Zeile 2, Spalte 3: Links von '=' muss ein Name oder ein Index")
skull_case(lang_number_huge         cases/lang_number_huge.skull         EXIT 1 MATCH "Zeile 1, Spalte 5: Zahl zu gross")
skull_case(lang_nesting_limit       cases/lang_nesting_limit.skull       EXIT 1 MATCH "zu tief verschachtelt .hoechstens 200 Ebenen.")
skull_case(lang_nesting_parens      cases/lang_nesting_parens.skull      EXIT 1 MATCH "zu tief verschachtelt .hoechstens 200 Ebenen.")
skull_case(lang_chain_limit         cases/lang_chain_limit.skull         EXIT 1 MATCH "Ausdruck zu lang .hoechstens 20000 Glieder")

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
skull_case(bpe_not_capped           cases/bpe_not_capped.skull           MATCH "Endgroesse Vokabular: 1000 Tokens.*vocab=1000, Bigram, BPE-Tokenizer.*--- Ende ---" NO_MATCH "Endgroesse Vokabular: 756")

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

# ---- Validierung ----
skull_case(val_split_info           cases/val_split_info.skull           MATCH "Validierung: 250 Token \\(10 % vom Dateiende\\), Training: 2250 Token.*Val: .*Perplexitaet.*Beste Validierung: Epoche")
skull_case(val_builtins             cases/val_builtins.skull             MATCH "BUILTINS OK" NO_MATCH "BUILTINS FALSCH")
skull_case(val_off                  cases/val_off.skull                  EXIT 1 MATCH "last_val_loss\\(\\): das letzte train hatte keine Validierung")
skull_case(val_off_info             cases/val_off_info.skull             MATCH "Loss: " NO_MATCH "Val: ")
skull_case(val_bad_value            cases/val_bad_value.skull            EXIT 1 MATCH "'val' muss zwischen 0 und 0.5")
skull_case(val_tiny_data            cases/val_tiny_data.skull            MATCH "Validierung uebersprungen")
skull_case(val_transformer          cases/val_transformer.skull          MATCH "Modell:   Transformer.*Val: .*VAL: true")
skull_case(val_patience_without_val cases/val_patience_without_val.skull MATCH "'patience' wirkt nur mit Validierung")
skull_case(val_last_loss_before_train cases/val_last_loss_before_train.skull EXIT 1 MATCH "last_loss\\(\\): es lief noch kein train")
skull_case(val_skip_info            cases/val_skip_info.skull            MATCH "Validierung zaehlt nicht die ersten 8 Positionen.*SKIP OK" NO_MATCH "SKIP FALSCH")
skull_case(val_skip_too_big         cases/val_skip_too_big.skull         EXIT 1 MATCH "'val_skip' .8. muss kleiner als 'context' .8. sein")
skull_case(val_skip_bigram          cases/val_skip_bigram.skull          MATCH "'val_skip' wirkt nur mit context > 1")
skull_case(val_skip_no_val          cases/val_skip_no_val.skull          MATCH "'val_skip' wirkt nur mit Validierung")
skull_case(res_builtins             cases/res_builtins.skull             MATCH "RES OK" NO_MATCH "FALSCH")
skull_case(res_params_before_train  cases/res_params_before_train.skull  EXIT 1 MATCH "last_params..: es lief noch kein train")
skull_case(res_seconds_before_train cases/res_seconds_before_train.skull EXIT 1 MATCH "last_seconds..: es lief noch kein train")
skull_case(gen_shift                cases/gen_shift.skull                MATCH "Transformer.*--- Ende ---")
skull_case(gen_long_transformer     cases/gen_long_transformer.skull     MATCH "Transformer.*--- Ende ---")

# ---- Checkpoints, seed, Sampling ----
skull_case(ckpt_resume              cases/ckpt_resume.skull              MATCH "Pausiert nach Epoche 2.*Fortgesetzt aus data/ckpt_a.weights.ckpt: Epoche 2 war abgeschlossen, weiter ab Epoche 3.*EPOCHEN GESAMT: true")
skull_case(ckpt_periodic            cases/ckpt_periodic.skull            MATCH "weiter ab Epoche 6")
skull_case(ckpt_missing             cases/ckpt_missing.skull             EXIT 1 MATCH "Checkpoint nicht gefunden")
skull_case(ckpt_wrong_model         cases/ckpt_wrong_model.skull         EXIT 1 MATCH "gehoert zu einem Transformer")
skull_case(ckpt_stop_ignored        cases/ckpt_stop_ignored.skull        MATCH "'stop_after' \\(5\\) ist nicht kleiner als 'epochs' \\(2\\)")
skull_case(gen_seed_topk_topp       cases/gen_seed_topk_topp.skull       MATCH "Seed:     5.*Top-k:    5.*Top-p:    0.9.*--- Ende ---")
skull_case(gen_top_p_bad            cases/gen_top_p_bad.skull            EXIT 1 MATCH "'top_p' muss groesser als 0")
skull_case(train_seed               cases/train_seed.skull               MATCH "SEED OK")
