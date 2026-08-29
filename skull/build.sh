#!/bin/bash
# ============================================================
#  SKULL EINFACHES BUILD-SKRIPT v1.0.0
#  Einfach ausführen: ./build.sh
# ============================================================

# Farben für bessere Lesbarkeit
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================
#  HILFE ANZEIGEN
# ============================================================
show_help() {
    echo -e "${BLUE}=== Skull Build-Skript v1.0.0 ===${NC}"
    echo ""
    echo "Verwendung:"
    echo "  ./build.sh              # Standard: CPU mit AVX2"
    echo "  ./build.sh --gpu       # Mit OpenCL GPU-Unterstützung"
    echo "  ./build.sh --cuda      # Mit CUDA GPU-Unterstützung (NVIDIA)"
    echo "  ./build.sh --debug     # Debug-Modus (langsamer, mehr Infos)"
    echo "  ./build.sh --clean     # Build-Verzeichnis bereinigen"
    echo "  ./build.sh --help      # Diese Hilfe anzeigen"
    echo ""
    echo "Beispiele nach dem Build:"
    echo "  ./skull examples/hello.skull"
    echo "  ./skull examples/train.skull"
    echo "  ./skull examples/generate.skull"
    echo ""
}

# ============================================================
#  FEHLERBEHANDLUNG
# ============================================================
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}FEHLER: $1 ist nicht installiert!${NC}"
        echo ""
        case $1 in
            "cmake")
                echo "Installiere CMake:"
                echo "  Ubuntu/Debian: sudo apt install cmake"
                echo "  Fedora: sudo dnf install cmake"
                echo "  macOS: brew install cmake"
                ;;
            "g++"|"gcc")
                echo "Installiere GCC/G++:"
                echo "  Ubuntu/Debian: sudo apt install build-essential"
                echo "  Fedora: sudo dnf install gcc-c++"
                echo "  macOS: brew install gcc" 
                ;;
            "git")
                echo "Installiere Git:"
                echo "  Ubuntu/Debian: sudo apt install git"
                echo "  Fedora: sudo dnf install git"
                echo "  macOS: brew install git"
                ;;
        esac
        exit 1
    fi
}

# ============================================================
#  HAUPT-SKRIPT
# ============================================================

# Argumente parsen
USE_GPU=false
USE_CUDA=false
DEBUG=false
CLEAN=false
HELP=false

for arg in "$@"; do
    case $arg in
        --gpu)
            USE_GPU=true
            ;;
        --cuda)
            USE_CUDA=true
            USE_GPU=true
            ;;
        --debug)
            DEBUG=true
            ;;
        --clean)
            CLEAN=true
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
    esac
done

# Hilfe anzeigen wenn keine Argumente
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}Keine Argumente angegeben. Standard: CPU-Only Build.${NC}"
    echo ""
fi

# Bereinigen
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}=== Bereinige Build-Verzeichnis ===${NC}"
    rm -rf build
    echo -e "${GREEN}Fertig! Build-Verzeichnis gelöscht.${NC}"
    exit 0
fi

# Abhängigkeiten prüfen
echo -e "${BLUE}=== Prüfe Abhängigkeiten ===${NC}"
check_command cmake
check_command g++
check_command git

# Plattform erkennen
PLATFORM="Linux"
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macOS"
elif [[ "$OSTYPE" == "cygwin"* || "$OSTYPE" == "msys"* ]]; then
    PLATFORM="Windows"
fi

echo -e "Plattform: ${GREEN}$PLATFORM${NC}"

# Compiler prüfen
COMPILER=$(g++ --version | head -n 1)
echo -e "Compiler: ${GREEN}$COMPILER${NC}"

# CMake Version prüfen
CMAKE_VERSION=$(cmake --version | head -n 1)
echo -e "CMake: ${GREEN}$CMAKE_VERSION${NC}"

# ============================================================
#  BUILD-KONFIGURATION
# ============================================================

echo ""
echo -e "${BLUE}=== Konfiguriere Build ===${NC}"

# Build-Verzeichnis erstellen
if [ ! -d "build" ]; then
    mkdir -p build
    echo -e "${GREEN}Build-Verzeichnis erstellt${NC}"
fi

cd build

# CMake-Befehl zusammenbauen
CMAKE_CMD="cmake .. -DCMAKE_BUILD_TYPE=Release"

if [ "$DEBUG" = true ]; then
    CMAKE_CMD="cmake .. -DCMAKE_BUILD_TYPE=Debug"
    echo -e "${YELLOW}Debug-Modus aktiviert (langsamer, mehr Infos)${NC}"
fi

if [ "$USE_GPU" = true ]; then
    CMAKE_CMD="$CMAKE_CMD -DSKULL_USE_OPENCL=ON"
    echo -e "${GREEN}OpenCL GPU-Unterstützung aktiviert${NC}"
fi

if [ "$USE_CUDA" = true ]; then
    CMAKE_CMD="$CMAKE_CMD -DSKULL_USE_CUDA=ON"
    echo -e "${GREEN}CUDA GPU-Unterstützung aktiviert${NC}"
fi

# AVX2 prüfen
if grep -q avx2 /proc/cpuinfo 2>/dev/null || [[ "$PLATFORM" == "macOS" ]]; then
    echo -e "${GREEN}AVX2-Unterstützung erkannt${NC}"
else
    echo -e "${YELLOW}AVX2 nicht verfügbar - Fallback zu Standard-Optimierungen${NC}"
fi

echo ""
echo -e "${BLUE}Ausführender Befehl:${NC}"
echo "  $CMAKE_CMD"
echo ""

# CMake ausführen
echo -e "${BLUE}=== Führe CMake aus ===${NC}"
if ! eval $CMAKE_CMD; then
    echo -e "${RED}FEHLER: CMake ist fehlgeschlagen!${NC}"
    echo ""
    echo "Mögliche Lösungen:"
    echo "  1. Installiere fehlende Abhängigkeiten (siehe oben)"
    echo "  2. Für CUDA: Installiere CUDA Toolkit"
    echo "     Ubuntu: sudo apt install nvidia-cuda-toolkit"
    echo "  3. Für OpenCL: Installiere OpenCL-Header"
    echo "     Ubuntu: sudo apt install opencl-headers ocl-icd-opencl-dev"
    exit 1
fi

echo ""
echo -e "${BLUE}=== Kompiliere Skull ===${NC}"

# Anzahl der CPU-Kerne erkennen
NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Kompilieren
if ! cmake --build . -j$NUM_CORES; then
    echo -e "${RED}FEHLER: Kompilierung fehlgeschlagen!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=== BUILD ERFOLGREICH! ===${NC}"
echo ""

# Executable prüfen
if [ -f "skull" ]; then
    echo -e "${GREEN}skull Executable erstellt in: $(pwd)/skull${NC}"
    echo ""
    echo -e "${BLUE}=== Testen ===${NC}"
    echo "Führe folgende Befehle aus, um Skull zu testen:"
    echo ""
    echo "  cd .."
    echo "  ./build/skull examples/hello.skull      # Grundlagen testen"
    echo "  ./build/skull examples/train.skull      # Modell trainieren"
    echo "  ./build/skull examples/generate.skull  # Text generieren"
    echo ""
else
    echo -e "${RED}FEHLER: skull Executable nicht gefunden!${NC}"
    exit 1
fi

cd ..
echo ""
echo -e "${GREEN}Skull v1.0.0 ist bereit zur Verwendung!${NC}"
