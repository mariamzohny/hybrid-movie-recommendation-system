@echo off
setlocal
cd /d "%~dp0"
if exist .venv\Scripts\activate call .venv\Scripts\activate
python pipeline.py
if errorlevel 1 goto :error
echo.
echo Pipeline finished successfully.
pause
exit /b 0
:error
echo.
echo Pipeline failed. Check the error above.
pause
exit /b 1
