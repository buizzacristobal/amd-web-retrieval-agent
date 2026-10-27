@echo off
setlocal enabledelayedexpansion

echo ==========================================================
echo   AMD Intelligent Web Retrieval Agent (Match 4)
echo ==========================================================

REM 1. Detect Python command
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set PY_CMD=python
) else (
    where py >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set PY_CMD=py
    ) else (
        echo [ERROR] Python was not found in your PATH.
        echo Please download and install Python from https://www.python.org/
        echo (Make sure to check "Add Python to PATH" during installation)
        pause
        exit /b 1
    )
)

REM 2. Setup Virtual Environment
if not exist ".venv" (
    echo [1/4] Creating virtual environment (.venv)...
    !PY_CMD! -m venv .venv
)

REM 3. Activate Virtual Environment
echo [2/4] Activating virtual environment...
call .venv\Scripts\activate.bat

REM 4. Install Dependencies
echo [3/4] Installing requirements...
pip install -r requirements.txt

REM 5. Run Verification Benchmark Suite
echo [4/4] Running Automated Verification Benchmark Suite...
python verify_pipeline.py

REM 6. Launch Streamlit UI
echo.
echo Launching Streamlit Dashboard on http://localhost:8501 ...
streamlit run app.py
pause
