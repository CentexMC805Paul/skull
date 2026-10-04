cmake_minimum_required(VERSION 3.15)

# ============================================================
#  SKULL BUILD
#  Plattformen: Linux, macOS, Windows (MSVC)
#  Standard: CPU (AVX2 wenn verfuegbar). OpenCL ist optional.
# ============================================================

# Versionsnummer aus src/version.h lesen (einzige Quelle)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/src/version.h" _skull_version_line
     REGEX "^#define SKULL_VERSION ")
string(REGEX REPLACE ".*\"([0-9]+\\.[0-9]+\\.[0-9]+)\".*" "\\1" SKULL_VERSION "${_skull_version_line}")

project(Skull VERSION ${SKULL_VERSION} LANGUAGES CXX)

option(SKULL_USE_OPENCL   "OpenCL-Geraeteerkennung einbauen (braucht OpenCL-Header + Bibliothek)" OFF)
option(SKULL_ENABLE_AVX2  "AVX2/FMA-Instruktionen verwenden (nur x86-64; fuer aeltere CPUs auf OFF stellen)" ON)
option(SKULL_SANITIZE     "AddressSanitizer + UBSan + Container-Assertions einschalten (gcc/clang)" OFF)
option(SKULL_TSAN         "ThreadSanitizer einschalten: findet Datenrennen (gcc/clang, nicht zusammen mit SKULL_SANITIZE)" OFF)
option(SKULL_BUILD_TESTS  "Tests (ctest) bauen" ON)

# Ohne Angabe: Release (Ein-Konfigurations-Generatoren wie Make/Ninja)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build-Typ" FORCE)
endif()

# Gemeinsame Compiler-/Linker-Flags fuer das Programm und die Tests
# (so laufen auch die Tests mit denselben Sanitizer- und AVX2-Einstellungen).
add_library(skull_flags INTERFACE)
target_compile_features(skull_flags INTERFACE cxx_std_17)

add_executable(skull src/main.cpp)
target_include_directories(skull PRIVATE src)
target_link_libraries(skull PRIVATE skull_flags)
set_target_properties(skull PROPERTIES CXX_EXTENSIONS OFF)

# Programm immer direkt in build/ ablegen (auch bei Visual Studio, das sonst
# nach build/Release/ schreibt) - so stimmen die Pfade in README und Skripten.
set_target_properties(skull PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY                "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG          "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE        "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL     "${CMAKE_BINARY_DIR}")

# ---- Architektur ----
set(_skull_x86 FALSE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64|X86_64)$")
    set(_skull_x86 TRUE)
endif()
set(SKULL_AVX2_ACTIVE FALSE)

# ---- Threads (Interpreter-Thread mit grossem Stack, parallele Trainingsschritte) ----
if(NOT WIN32)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(skull_flags INTERFACE Threads::Threads)
endif()

# ---- Compiler-Flags ----
if(MSVC)
    target_compile_options(skull_flags INTERFACE /W3 /EHsc /MP /utf-8)
    # Der Interpreter laeuft normalerweise in einem Thread mit grossem Stack
    # (src/stack.h). Dieser Haupt-Stack ist nur der Rueckfall, falls das
    # Anlegen des Threads scheitert (Windows-Standard waere nur 1 MB).
    target_link_options(skull PRIVATE /STACK:16777216)
    if(SKULL_ENABLE_AVX2 AND _skull_x86)
        target_compile_options(skull_flags INTERFACE /arch:AVX2)
        set(SKULL_AVX2_ACTIVE TRUE)
    endif()
else()
    target_compile_options(skull_flags INTERFACE -Wall -Wextra -Wpedantic)
    if(SKULL_ENABLE_AVX2 AND _skull_x86)
        target_compile_options(skull_flags INTERFACE -mavx2 -mfma)
        set(SKULL_AVX2_ACTIVE TRUE)
    endif()
endif()

# ---- Sanitizer (fuer Tests/CI) ----
if(SKULL_SANITIZE)
    if(MSVC)
        message(FATAL_ERROR "SKULL_SANITIZE wird nur mit gcc/clang unterstuetzt")
    endif()
    target_compile_options(skull_flags INTERFACE
        -fsanitize=address,undefined -fno-sanitize-recover=undefined
        -fno-omit-frame-pointer -g -O1)
    target_compile_definitions(skull_flags INTERFACE _GLIBCXX_ASSERTIONS)
    target_link_options(skull_flags INTERFACE -fsanitize=address,undefined)
endif()

# ---- ThreadSanitizer (fuer Tests/CI) ----
if(SKULL_TSAN)
    if(MSVC)
        message(FATAL_ERROR "SKULL_TSAN wird nur mit gcc/clang unterstuetzt")
    endif()
    if(SKULL_SANITIZE)
        message(FATAL_ERROR "SKULL_TSAN und SKULL_SANITIZE schliessen sich aus (ThreadSanitizer und AddressSanitizer nicht kombinierbar)")
    endif()
    target_compile_options(skull_flags INTERFACE -fsanitize=thread -fno-omit-frame-pointer -g -O1)
    target_link_options(skull_flags INTERFACE -fsanitize=thread)
endif()

# ---- OpenCL (optional) ----
# Nur Geraeteerkennung (gpu_info(), gpu = true). Das Training selbst
# laeuft derzeit immer auf der CPU.
if(SKULL_USE_OPENCL)
    find_package(OpenCL REQUIRED)
    target_compile_definitions(skull PRIVATE SKULL_USE_OPENCL=1)
    target_link_libraries(skull PRIVATE OpenCL::OpenCL)
endif()

# ---- Installation ----
install(TARGETS skull RUNTIME DESTINATION bin)
install(DIRECTORY examples DESTINATION share/skull)

# ---- Tests ----
if(SKULL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---- Zusammenfassung ----
if(CMAKE_BUILD_TYPE)
    set(_skull_build_type "${CMAKE_BUILD_TYPE}")
else()
    set(_skull_build_type "Multi-Config (--config waehlt)")
endif()
message(STATUS "")
message(STATUS "Skull ${SKULL_VERSION}")
message(STATUS "  Plattform:   ${CMAKE_SYSTEM_NAME} (${CMAKE_SYSTEM_PROCESSOR})")
message(STATUS "  Build-Typ:   ${_skull_build_type}")
message(STATUS "  AVX2:        ${SKULL_AVX2_ACTIVE}")
message(STATUS "  OpenCL:      ${SKULL_USE_OPENCL}")
message(STATUS "  Sanitizer:   ${SKULL_SANITIZE} (Thread: ${SKULL_TSAN})")
message(STATUS "  Tests:       ${SKULL_BUILD_TESTS}")
message(STATUS "")
