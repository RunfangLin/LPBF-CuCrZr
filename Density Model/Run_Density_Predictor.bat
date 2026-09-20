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

"%PYTHON_EXE%" -c "import tkinter, joblib, pandas, xgboost, sklearn" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Required packages are missing from:
    echo     %PYTHON_EXE%
    echo.
    echo Install them with:
    echo     "%PYTHON_EXE%" -m pip install -r "%~dp0..\requirements.txt"
    echo.
    echo Note: the predictor checks that the installed xgboost and
    echo scikit-learn versions match those recorded in results_summary.json,
    echo and refuses to load the model otherwise.
    echo.
    pause
    exit /b 1
)

if not exist "%~dp0Density_Results\density_model.pkl" (
    echo ERROR: No fitted model was found at:
    echo     %~dp0Density_Results\density_model.pkl
    echo.
    echo Run Run_Density_Model.bat first.
    echo.
    pause
    exit /b 1
)

"%PYTHON_EXE%" -B "%~dp0Density_Predictor.py"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo ERROR: Density_Predictor.py exited with code %EXIT_CODE%.
    pause
)

endlocal & exit /b %EXIT_CODE%
