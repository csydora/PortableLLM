@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0server_control.py" start
if errorlevel 1 (
    pause
    exit /b 1
)
pause

