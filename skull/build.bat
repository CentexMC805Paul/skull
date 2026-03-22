@echo off
echo === Skull Build System ===
echo.

:: Modus pruefen
set GPU_MODE=0
if "%1"=="--gpu" (
    set GPU_MODE=1
    echo Modus: GPU (OpenCL)
) else (
    echo Modus: CPU (AVX2 SIMD)
    echo Tipp: build.bat --gpu fuer GPU-Support
)
echo.

:: Visual Studio finden
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
    goto :build
)
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
    goto :build
)
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
    goto :build
)
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
    goto :build
)
echo FEHLER: Visual Studio nicht gefunden.
pause
exit /b 1

:build
echo Kompiliere Skull v0.7.0...
echo.

if "%GPU_MODE%"=="1" (
    cl /std:c++17 /O2 /W3 /EHsc /arch:AVX2 /DSKULL_USE_OPENCL /I src src\main.cpp /Fe:skull.exe
) else (
    cl /std:c++17 /O2 /W3 /EHsc /arch:AVX2 /I src src\main.cpp /Fe:skull.exe
)

if %ERRORLEVEL% == 0 (
    echo.
    echo ==============================
    echo  Build erfolgreich!
    echo  skull.exe erstellt
    echo ==============================
    echo.
    echo Testen:
    echo   skull.exe examples\tokenizer_test.skull
    echo   skull.exe examples\generate_demo.skull
) else (
    echo.
    echo Build fehlgeschlagen!
)

echo.
pause
