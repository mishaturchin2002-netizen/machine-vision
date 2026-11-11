@echo off
rem ---------------------------------------------------------------------------
rem  machine-vision launcher
rem
rem  Default profile:  --track --speed --eyes
rem  Extra flags are appended, e.g.  start_vision.bat --bg-blur
rem ---------------------------------------------------------------------------
setlocal
chcp 65001 >nul
title machine-vision
cd /d "%~dp0.."

set "PY=%CD%\.venv\Scripts\python.exe"

rem --- hard requirement: the project venv created by setup.bat ---------------
if not exist "%PY%" (
    echo.
    echo  ============================================================
    echo   [ERROR] Виртуальное окружение .venv не найдено.
    echo.
    echo   Сначала запустите setup.bat
    echo   ^(он создаст .venv и установит зависимости^)
    echo.
    echo   Ожидаемый путь:
    echo     %PY%
    echo  ============================================================
    echo.
    pause
    exit /b 1
)

if not exist "%CD%\src\realtime_vision.py" (
    echo [ERROR] src\realtime_vision.py not found in %CD%
    echo         Launch this file from the repository's scripts\ folder.
    pause
    exit /b 1
)

set "PROFILE=--track --speed --eyes"
if not "%~1"=="" set "PROFILE=--track --speed --eyes %*"

echo.
echo  ============================================================
echo   machine-vision
echo   profile : %PROFILE%
echo.
echo   default flags: tracking IDs, speedometer, gaze
echo   add your own, e.g.:
echo     start_vision.bat --bg-blur
echo     start_vision.bat --trails --async-grab
echo     start_vision.bat --depth-map
echo     start_vision.bat --benchmark
echo.
echo   press  q  in the video window to stop
echo  ============================================================
echo.

"%PY%" src\realtime_vision.py --track --speed --eyes %*
if errorlevel 1 (
    echo.
    echo [ERROR] exited with an error - see the log above.
    pause
    exit /b 1
)

echo.
echo Stopped.
endlocal
