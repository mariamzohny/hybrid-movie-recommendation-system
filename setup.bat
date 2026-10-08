@echo off
setlocal
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 goto :error
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 goto :error
echo.
echo Setup complete.
echo Next: run build_data.bat, then run_app.bat.
pause
exit /b 0
:error
echo.
echo Setup failed. Check the error above.
pause
exit /b 1
