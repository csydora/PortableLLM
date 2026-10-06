@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m portablellm.server_control start
if errorlevel 1 (
    pause
    exit /b 1
)
pause

