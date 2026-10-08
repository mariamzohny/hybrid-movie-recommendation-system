@echo off
setlocal
cd /d "%~dp0"
if exist .venv\Scripts\activate call .venv\Scripts\activate
python -m streamlit run app.py
