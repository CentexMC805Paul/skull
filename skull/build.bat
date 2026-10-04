@echo off
setlocal EnableExtensions

rem ============================================================
rem  SKULL BUILD-SKRIPT (Windows)
rem  Einfach ausfuehren: build.bat
rem
rem  Voraussetzung: CMake und Visual Studio (2019 oder neuer) mit den
rem  C++-Werkzeugen. CMake waehlt automatisch das neueste installierte
rem  Visual Studio; ein Entwickler-Eingabeaufforderungs-Fenster ist
rem  nicht noetig.
rem ============================================================

cd /d "%~dp0"

set "USE_GPU=0"
set "DEBUG=0"
set "RUN_TESTS=0"
set "CLEAN=0"
set "RESULT=0"

rem Bei Doppelklick (cmd /c) am Ende pausieren, damit das Fenster nicht verschwindet
set "PAUSE_AT_END=0"
echo %cmdcmdline% | find /i "/c" >nul && set "PAUSE_AT_END=1"

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--gpu"   set "USE_GPU=1"   & shift & goto parse_args
if /i "%~1"=="--debug" set "DEBUG=1"     & shift & goto parse_args
if /i "%~1"=="--test"  set "RUN_TESTS=1" & shift & goto parse_args
if /i "%~1"=="--clean" set "CLEAN=1"     & shift & goto parse_args
if /i "%~1"=="--help"  goto show_help
if /i "%~1"=="-h"      goto show_help
echo Unbekannte Option: %~1
echo.
set "RESULT=1"
goto show_help
:args_done

if "%CLEAN%"=="1" (
    if exist build rmdir /s /q build
    echo Build-Verzeichnis geloescht.
    goto done
)

echo === Pruefe Abhaengigkeiten ===
where cmake >nul 2>&1
if errorlevel 1 (
    echo FEHLER: cmake wurde nicht gefunden.
    echo   Installiere CMake: https://cmake.org/download/
    echo   Installiere Visual Studio mit den C++-Werkzeugen: https://visualstudio.microsoft.com/downloads/
    set "RESULT=1"
    goto done
)

set "CONFIG=Release"
if "%DEBUG%"=="1" set "CONFIG=Debug"

set "CMAKE_OPTS=-DCMAKE_BUILD_TYPE=%CONFIG%"
if "%USE_GPU%"=="1" set "CMAKE_OPTS=%CMAKE_OPTS% -DSKULL_USE_OPENCL=ON"

echo.
echo === Konfiguriere (%CONFIG%) ===
echo   cmake -S . -B build %CMAKE_OPTS%
cmake -S . -B build %CMAKE_OPTS%
if errorlevel 1 goto cmake_failed

echo.
echo === Kompiliere Skull ===
cmake --build build --config %CONFIG% --parallel
if errorlevel 1 goto build_failed

if not exist build\skull.exe (
    echo FEHLER: build\skull.exe wurde nicht erzeugt!
    set "RESULT=1"
    goto done
)

echo.
echo === BUILD ERFOLGREICH: %cd%\build\skull.exe ===

if not "%RUN_TESTS%"=="1" goto tests_done
echo.
echo === Fuehre Tests aus ===
pushd build
ctest -C %CONFIG% --output-on-failure
if errorlevel 1 (
    popd
    goto tests_failed
)
popd
:tests_done

echo.
echo Probiere es aus:
echo   build\skull.exe examples\hello.skull      Grundlagen
echo   build\skull.exe examples\train.skull      Modell trainieren
echo   build\skull.exe examples\generate.skull   Text generieren
goto done

:cmake_failed
echo.
echo FEHLER: CMake ist fehlgeschlagen!
echo   Ist Visual Studio mit den C++-Werkzeugen installiert?
if "%USE_GPU%"=="1" echo   Fuer --gpu wird zusaetzlich ein OpenCL-SDK benoetigt. Ohne --gpu bauen geht auch.
set "RESULT=1"
goto done

:build_failed
echo.
echo FEHLER: Kompilierung fehlgeschlagen!
set "RESULT=1"
goto done

:tests_failed
echo.
echo FEHLER: Tests sind fehlgeschlagen!
set "RESULT=1"
goto done

:show_help
echo.
echo === Skull Build-Skript ===
echo.
echo Verwendung:
echo   build.bat            Standard: CPU (AVX2)
echo   build.bat --gpu      Mit OpenCL-Geraeteerkennung (braucht OpenCL-SDK)
echo   build.bat --debug    Debug-Modus
echo   build.bat --test     Nach dem Bauen alle Tests ausfuehren
echo   build.bat --clean    Build-Verzeichnis loeschen
echo   build.bat --help     Diese Hilfe
echo.
goto done

:done
if "%PAUSE_AT_END%"=="1" pause
exit /b %RESULT%
