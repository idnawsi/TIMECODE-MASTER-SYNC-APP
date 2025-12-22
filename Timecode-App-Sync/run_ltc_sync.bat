@echo off
title LTC Timecode Sync
echo.
echo  LTC Timecode Sync
echo  =================
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check for numpy
python -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing numpy...
    pip install numpy
)

REM Check for tkinterdnd2 (optional, for drag-and-drop support)
python -c "import tkinterdnd2" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing tkinterdnd2 for drag-and-drop support...
    pip install tkinterdnd2
)

REM Check for Pillow (for video preview)
python -c "from PIL import Image" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing Pillow for video preview...
    pip install Pillow
)

REM Check for ffmpeg in PATH
ffprobe -version >nul 2>&1
if errorlevel 1 (
    REM Try winget installation path
    set "FFMPEG_PATH=C:\Users\%USERNAME%\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin"
    if exist "%FFMPEG_PATH%\ffprobe.exe" (
        echo [INFO] Adding FFmpeg to PATH for this session...
        set "PATH=%FFMPEG_PATH%;%PATH%"
    ) else (
        echo.
        echo [WARNING] FFmpeg not found!
        echo Install with: winget install Gyan.FFmpeg
        echo Then restart this terminal.
        echo.
    )
)

echo [INFO] Starting LTC Sync App...
echo.
python "%~dp0ltc_sync_app.py"

if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with error
    pause
)
