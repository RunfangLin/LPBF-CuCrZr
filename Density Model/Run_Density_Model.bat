@echo off
setlocal

cd /d "%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"

rem Prefer the project virtual environment; fall back to Python on PATH.
set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

"%PYTHON_EXE%" --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: No Python interpreter was found.
    echo.
    echo Install Python 3.11 or later, then from the project root run:
    echo     python -m venv .venv
    echo     .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -c "import numpy, pandas, sklearn, xgboost, joblib, matplotlib" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Required packages are missing from:
    echo     %PYTHON_EXE%
    echo.
    echo Install them with:
    echo     "%PYTHON_EXE%" -m pip install -r "%~dp0..\requirements.txt"
    echo.
    pause
    exit /b 1
)

echo Training the density model with:
echo     %PYTHON_EXE%
echo.

"%PYTHON_EXE%" -B "%~dp0Density_Model.py"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
    echo Density model training completed successfully.
) else (
    echo ERROR: Density_Model.py exited with code %EXIT_CODE%.
)
pause

endlocal & exit /b %EXIT_CODE%
