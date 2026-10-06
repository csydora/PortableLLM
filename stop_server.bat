@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m portablellm.server_control stop
pause

