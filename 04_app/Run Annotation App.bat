@echo off
REM ============================================================
REM  catraj - trajectory annotation
REM  Double-click this file to launch the app in your browser.
REM ============================================================

title catraj - trajectory annotation

REM Run from the repository root (this file lives in 04_app\), wherever it
REM was launched from.
cd /d "%~dp0.."

REM Some Intel MKL builds abort the process the first time a LAPACK routine
REM runs unless the threading layer is fixed first, and it must be set before
REM NumPy loads.
set MKL_THREADING_LAYER=SEQUENTIAL

REM The conda environment holding geopandas, movingpandas, panel and the rest.
REM Looks in the per-user and the shared conda folders; set PYTHON yourself
REM to use a different interpreter.
if not defined PYTHON set PYTHON=%USERPROFILE%\.conda\envs\catraj\python.exe
if not exist "%PYTHON%" set PYTHON=C:\ProgramData\miniconda3\envs\catraj\python.exe

if not exist "%PYTHON%" (
    echo.
    echo   Could not find the Python environment at:
    echo     %PYTHON%
    echo.
    echo   Edit this file and set PYTHON to your own interpreter, or run:
    echo     conda activate catraj
    echo     python 04_app\run_app.py
    echo.
    pause
    exit /b 1
)

echo.
echo   Starting the annotation app...
echo   Your browser will open at http://localhost:5006/annotation_app
echo.
echo   Close this window to stop the app.
echo.

"%PYTHON%" "04_app\run_app.py" %*

REM Keep the window open if it exited unexpectedly, so the error stays readable.
if errorlevel 1 pause
