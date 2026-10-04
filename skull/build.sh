#!/bin/bash
# ============================================================
#  SKULL BUILD-SKRIPT (Linux / macOS)
#  Einfach ausfuehren: ./build.sh
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

show_help() {
    echo -e "${BLUE}=== Skull Build-Skript ===${NC}"
    echo ""
    echo "Verwendung:"
    echo "  ./build.sh              # Standard: CPU (AVX2, falls die CPU es kann)"
    echo "  ./build.sh --gpu        # Mit OpenCL-Geraeteerkennung (braucht OpenCL-Header)"
    echo "  ./build.sh --debug      # Debug-Modus (langsamer, mehr Infos)"
    echo "  ./build.sh --test       # Nach dem Bauen alle Tests ausfuehren"
    echo "  ./build.sh --clean      # Build-Verzeichnis loeschen"
    echo "  ./build.sh --help       # Diese Hilfe"
    echo ""
    echo "Nach dem Build:"
    echo "  ./build/skull examples/hello.skull"
    echo "  ./build/skull examples/train.skull"
    echo "  ./build/skull examples/generate.skull"
    echo ""
}

# Immer im Verzeichnis dieses Skripts arbeiten
cd "$(dirname "$0")" || exit 1

USE_GPU=false
DEBUG=false
CLEAN=false
RUN_TESTS=false

for arg in "$@"; do
    case $arg in
        --gpu)        USE_GPU=true ;;
        --debug)      DEBUG=true ;;
        --test)       RUN_TESTS=true ;;
        --clean)      CLEAN=true ;;
        --help|-h)    show_help; exit 0 ;;
        *)
            echo -e "${RED}Unbekannte Option: $arg${NC}"
            show_help
            exit 1
            ;;
    esac
done

if [ "$CLEAN" = true ]; then
    rm -rf build
    echo -e "${GREEN}Build-Verzeichnis geloescht.${NC}"
    exit 0
fi

# ---- Abhaengigkeiten ----
echo -e "${BLUE}=== Pruefe Abhaengigkeiten ===${NC}"
if ! command -v cmake >/dev/null 2>&1; then
    echo -e "${RED}FEHLER: cmake ist nicht installiert!${NC}"
    echo "  Ubuntu/Debian: sudo apt install cmake"
    echo "  Fedora:        sudo dnf install cmake"
    echo "  macOS:         brew install cmake"
    exit 1
fi
CXX_FOUND=""
for c in "${CXX:-}" g++ clang++ c++; do
    if [ -n "$c" ] && command -v "$c" >/dev/null 2>&1; then CXX_FOUND="$c"; break; fi
done
if [ -z "$CXX_FOUND" ]; then
    echo -e "${RED}FEHLER: Kein C++-Compiler gefunden!${NC}"
    echo "  Ubuntu/Debian: sudo apt install build-essential"
    echo "  Fedora:        sudo dnf install gcc-c++"
    echo "  macOS:         xcode-select --install"
    exit 1
fi
echo -e "Compiler: ${GREEN}$($CXX_FOUND --version | head -n 1)${NC}"
echo -e "CMake:    ${GREEN}$(cmake --version | head -n 1)${NC}"

# ---- Konfiguration ----
CMAKE_ARGS=(-S . -B build)

if [ "$DEBUG" = true ]; then
    CMAKE_ARGS+=(-DCMAKE_BUILD_TYPE=Debug)
    echo -e "${YELLOW}Debug-Modus aktiviert${NC}"
else
    CMAKE_ARGS+=(-DCMAKE_BUILD_TYPE=Release)
fi

if [ "$USE_GPU" = true ]; then
    CMAKE_ARGS+=(-DSKULL_USE_OPENCL=ON)
    echo -e "${GREEN}OpenCL-Geraeteerkennung aktiviert${NC} (das Training laeuft trotzdem auf der CPU)"
fi

# AVX2 nur verwenden, wenn die CPU es kann (sonst "Illegal instruction")
if [ -r /proc/cpuinfo ] && ! grep -q avx2 /proc/cpuinfo; then
    CMAKE_ARGS+=(-DSKULL_ENABLE_AVX2=OFF)
    echo -e "${YELLOW}AVX2 nicht verfuegbar - baue ohne AVX2${NC}"
fi

echo ""
echo -e "${BLUE}=== Konfiguriere ===${NC}"
echo "  cmake ${CMAKE_ARGS[*]}"
if ! cmake "${CMAKE_ARGS[@]}"; then
    echo -e "${RED}FEHLER: CMake ist fehlgeschlagen!${NC}"
    if [ "$USE_GPU" = true ]; then
        echo "  OpenCL-Header fehlen? Ubuntu: sudo apt install opencl-headers ocl-icd-opencl-dev"
        echo "  Ohne GPU-Erkennung bauen: ./build.sh"
    fi
    exit 1
fi

# ---- Bauen ----
echo ""
echo -e "${BLUE}=== Kompiliere Skull ===${NC}"
NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
if ! cmake --build build -j"$NUM_CORES"; then
    echo -e "${RED}FEHLER: Kompilierung fehlgeschlagen!${NC}"
    exit 1
fi

if [ ! -f build/skull ]; then
    echo -e "${RED}FEHLER: build/skull wurde nicht erzeugt!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=== BUILD ERFOLGREICH: $(pwd)/build/skull ===${NC}"

# ---- Tests ----
if [ "$RUN_TESTS" = true ]; then
    echo ""
    echo -e "${BLUE}=== Fuehre Tests aus ===${NC}"
    if ! (cd build && ctest --output-on-failure -j"$NUM_CORES"); then
        echo -e "${RED}FEHLER: Tests sind fehlgeschlagen!${NC}"
        exit 1
    fi
fi

echo ""
echo "Probiere es aus:"
echo "  ./build/skull examples/hello.skull      # Grundlagen"
echo "  ./build/skull examples/train.skull      # Modell trainieren"
echo "  ./build/skull examples/generate.skull   # Text generieren"
echo ""
