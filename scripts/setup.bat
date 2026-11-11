@echo off
rem ---------------------------------------------------------------------------
rem  machine-vision - one-shot setup (Windows)
rem  Creates .venv, installs pinned requirements, downloads + exports models.
rem ---------------------------------------------------------------------------
setlocal
chcp 65001 >nul
title machine-vision setup
cd /d "%~dp0.."

echo.
echo  ============================================================
echo   machine-vision setup
echo   %CD%
echo  ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] python not found in PATH.
    echo         Install Python 3.11+ and tick "Add to PATH".
    pause
    exit /b 1
)
python --version

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] creating virtual environment .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] could not create .venv
        pause
        exit /b 1
    )
) else (
    echo [1/4] .venv already exists
)

set "PY=%CD%\.venv\Scripts\python.exe"

echo [2/4] upgrading pip ...
"%PY%" -m pip install --upgrade pip --quiet

echo [3/4] installing requirements (this takes a while, ~2 GB) ...
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] pip install failed - see the message above.
    pause
    exit /b 1
)

rem easyocr pulls opencv-python-headless and breaks cv2.imshow; put the GUI build back
"%PY%" -c "import cv2; cv2.namedWindow('t'); cv2.destroyAllWindows()" 2>nul
if errorlevel 1 (
    echo [i] restoring the GUI build of OpenCV ...
    "%PY%" -m pip install --force-reinstall --no-deps opencv-contrib-python==5.0.0.93 --quiet
)

echo [4/4] downloading models ...
"%PY%" scripts\download_models.py
if errorlevel 1 (
    echo.
    echo [WARN] some models are missing - the app will not start until they are.
    echo        Re-run: "%PY%" scripts\download_models.py
    pause
    exit /b 1
)

echo.
echo  ============================================================
echo   Done. Launch with:   scripts\start_vision.bat
echo  ============================================================
pause
