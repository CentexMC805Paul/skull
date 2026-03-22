@echo off
echo Raeume alten Skull-Code auf...

:: Alte .skull Dateien loeschen
del /f /q "D:\Projekte(mit Code)\projekt_skull\autograd_demo.skull" 2>nul
del /f /q "D:\Projekte(mit Code)\projekt_skull\benchmark.skull" 2>nul
del /f /q "D:\Projekte(mit Code)\projekt_skull\modell.skull" 2>nul
del /f /q "D:\Projekte(mit Code)\projekt_skull\train.skull" 2>nul
del /f /q "D:\Projekte(mit Code)\projekt_skull\transformer_demo.skull" 2>nul

:: Alte C++ Dateien loeschen
del /f /q "D:\Projekte(mit Code)\projekt_skull\main.cpp" 2>nul

:: Alte .exe loeschen
del /f /q "D:\Projekte(mit Code)\projekt_skull\skull.exe" 2>nul

:: Alte Trainingsdaten loeschen
del /f /q "D:\Projekte(mit Code)\projekt_skull\daten.jsonl" 2>nul

:: Alten SkullCompiler Ordner komplett loeschen
rmdir /s /q "D:\Projekte(mit Code)\projekt_skull\SkullCompiler" 2>nul

echo.
echo Fertig! Alte Dateien geloescht.
echo.
echo Neue Struktur:
echo   projekt_skull\
echo     skull\          (Projekt 1 - Die Sprache)
echo     skull-research\ (Projekt 2 - Die Forschung)
echo     README.md
echo     .gitignore
echo.
pause
