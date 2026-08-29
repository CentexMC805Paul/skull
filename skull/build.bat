@echo off
:: ============================================================
::  SKULL EINFACHES BUILD-SKRIPT FÜR WINDOWS v1.0.0
::  Einfach ausführen: build.bat
:: ============================================================

:: Farben für bessere Lesbarkeit (Windows 10+)
@echo off
setlocal enabledelayedexpansion

:: ============================================================
::  HILFE ANZEIGEN
:: ============================================================
:show_help
    echo.
    echo === Skull Build-Skript v1.0.0 ===
    echo.
    echo Verwendung:
    echo   build.bat              # Standard: CPU mit AVX2
    echo   build.bat --gpu       # Mit OpenCL GPU-Unterstützung
    echo   build.bat --cuda      # Mit CUDA GPU-Unterstützung (NVIDIA)
    echo   build.bat --debug     # Debug-Modus (langsamer, mehr Infos)
    echo   build.bat --clean     # Build-Verzeichnis bereinigen
    echo   build.bat --help      # Diese Hilfe anzeigen
    echo.
    echo Beispiele nach dem Build:
    echo   skull.exe examples\hello.skull
    echo   skull.exe examples\train.skull
    echo   skull.exe examples\generate.skull
    echo.
    goto :eof

:: ============================================================
::  ARGUMENTE PARSEN
:: ============================================================
set USE_GPU=0
set USE_CUDA=0
set DEBUG=0
set CLEAN=0
set HELP=0

:parse_args
if "%1"=="" goto :end_parse
if "%1"=="--gpu" set USE_GPU=1 & shift & goto :parse_args
if "%1"=="--cuda" set USE_CUDA=1 & set USE_GPU=1 & shift & goto :parse_args
if "%1"=="--debug" set DEBUG=1 & shift & goto :parse_args
if "%1"=="--clean" set CLEAN=1 & shift & goto :parse_args
if "%1"=="--help" set HELP=1 & shift & goto :parse_args
if "%1"=="-h" set HELP=1 & shift & goto :parse_args
:end_parse

:: Hilfe anzeigen wenn gewünscht
if %HELP%==1 (
    call :show_help
    exit /b 0
)

:: ============================================================
::  BEREINIGEN
:: ============================================================
if %CLEAN%==1 (
    echo === Bereinige Build-Verzeichnis ===
    if exist build rmdir /s /q build
    echo Fertig! Build-Verzeichnis gelöscht.
    exit /b 0
)

:: ============================================================
::  VISUAL STUDIO PRÜFEN
:: ============================================================
echo === Prüfe Visual Studio ===

:: Versuche verschiedene VS-Versionen
set VCVARS_BAT=""
set VS_FOUND=0

:: VS 2022 Community
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set VCVARS_BAT="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    set VS_FOUND=1
    echo Visual Studio 2022 Community gefunden
)

:: VS 2022 Professional
if %VS_FOUND%==0 if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    set VCVARS_BAT="C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    set VS_FOUND=1
    echo Visual Studio 2022 Professional gefunden
)

:: VS 2019 Community
if %VS_FOUND%==0 if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set VCVARS_BAT="C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
    set VS_FOUND=1
    echo Visual Studio 2019 Community gefunden
)

:: VS 2017 Community
if %VS_FOUND%==0 if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set VCVARS_BAT="C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"
    set VS_FOUND=1
    echo Visual Studio 2017 Community gefunden
)

:: Kein VS gefunden
if %VS_FOUND%==0 (
    echo FEHLER: Visual Studio nicht gefunden!
    echo.
    echo Installiere Visual Studio 2022 mit C++-Tools:
    echo   https://visualstudio.microsoft.com/downloads/
    echo.
    echo Oder installiere die Build-Tools:
    echo   https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo.
    pause
    exit /b 1
)

:: VS-Umgebung laden
echo.
echo Lade Visual Studio Umgebung...
call "%VCVARS_BAT%" > nul 2>&1

:: ============================================================
::  BUILD-KONFIGURATION
:: ============================================================
echo.
echo === Konfiguriere Build ===

:: Build-Verzeichnis erstellen
if not exist build mkdir build
cd build

:: CMake-Befehl zusammenbauen
set CMAKE_CMD=cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release

if %DEBUG%==1 (
    set CMAKE_CMD=cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Debug
    echo Debug-Modus aktiviert (langsamer, mehr Infos)
)

if %USE_GPU%==1 (
    set CMAKE_CMD=%CMAKE_CMD% -DSKULL_USE_OPENCL=ON
    echo OpenCL GPU-Unterstützung aktiviert
)

if %USE_CUDA%==1 (
    set CMAKE_CMD=%CMAKE_CMD% -DSKULL_USE_CUDA=ON
    echo CUDA GPU-Unterstützung aktiviert
)

echo.
echo Ausführender Befehl:
echo   %CMAKE_CMD%
echo.

:: ============================================================
::  CMAKE AUSFÜHREN
:: ============================================================
echo === Führe CMake aus ===
%CMAKE_CMD%

if %ERRORLEVEL% neq 0 (
    echo.
    echo FEHLER: CMake ist fehlgeschlagen!
    echo.
    echo Mögliche Lösungen:
    echo   1. Installiere fehlende Abhängigkeiten
    echo   2. Für CUDA: Installiere CUDA Toolkit
    echo      https://developer.nvidia.com/cuda-downloads
    echo   3. Für OpenCL: Installiere OpenCL SDK
    echo      https://developer.nvidia.com/opencl
    echo.
    pause
    exit /b 1
)

echo.
echo === Kompiliere Skull ===

:: Anzahl der CPU-Kerne erkennen (einfach auf 8 setzen für Windows)
set NUM_CORES=8

:: Kompilieren
cmake --build . --config Release --parallel %NUM_CORES%

if %ERRORLEVEL% neq 0 (
    echo.
    echo FEHLER: Kompilierung fehlgeschlagen!
    echo.
    pause
    exit /b 1
)

echo.
echo ==============================
echo  BUILD ERFOLGREICH!
echo ==============================
echo.

:: Executable prüfen
if exist skull.exe (
    echo skull.exe erstellt in: %cd%\skull.exe
    echo.
    echo === Testen ===
    echo Führe folgende Befehle aus, um Skull zu testen:
    echo.
    echo   cd ..
    echo   skull.exe examples\hello.skull      # Grundlagen testen
    echo   skull.exe examples\train.skull      # Modell trainieren
    echo   skull.exe examples\generate.skull  # Text generieren
    echo.
) else (
    echo FEHLER: skull.exe nicht gefunden!
    pause
    exit /b 1
)

cd ..
echo.
echo Skull v1.0.0 ist bereit zur Verwendung!
pause
