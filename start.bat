@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] First-time setup: creating virtual environment and installing dependencies...
    python -m venv .venv
    if errorlevel 1 goto :error
    call .venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 goto :error
) else (
    call .venv\Scripts\activate
)

echo.
echo [2/3] Checking generated data...
if not exist "data\processed\fpgrowth frequent association rules.csv" (
    echo Project data has not been generated yet.
    echo Run build_data.bat first. BERT generation may take time on CPU.
    echo.
    pause
    exit /b 0
)

echo [3/3] Launching Streamlit...
python -m streamlit run app.py
exit /b 0

:error
echo.
echo Setup failed. Check the error above.
pause
exit /b 1
