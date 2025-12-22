@echo off
title LTC Audio Analyzer
echo.
echo  LTC Audio to Text Analyzer
echo  ==========================
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH
    pause
    exit /b 1
)

REM Check for numpy
python -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing numpy...
    pip install numpy
)

echo.
echo Drag and drop a video/audio file onto this window, or type the path:
echo.

if "%~1"=="" (
    python "%~dp0ltc_audio_to_text.py"
) else (
    python "%~dp0ltc_audio_to_text.py" "%~1"
)

echo.
pause
