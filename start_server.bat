@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
uvicorn portablellm.main:app --reload
pause