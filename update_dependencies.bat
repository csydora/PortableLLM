@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    set "SETUP_STEP=Creating the Python environment"
    py -3.12 -m venv .venv
    if errorlevel 1 goto :failed
)
set "SETUP_STEP=Installing common dependencies"
".venv\Scripts\python.exe" -m pip install -r "installation\requirements.txt"
if errorlevel 1 goto :failed
set "SETUP_STEP=Installing and checking the selected llama.cpp backend"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0installation\install_gpu_backend.ps1"
if errorlevel 1 goto :failed
echo.
echo Dependency setup completed successfully.
pause
exit /b 0

:failed
set "SETUP_EXIT_CODE=%ERRORLEVEL%"
echo.
echo Dependency setup failed during: %SETUP_STEP%
echo Exit code: %SETUP_EXIT_CODE%
echo The error details are above. Copy them before closing this window.
pause
exit /b %SETUP_EXIT_CODE%

