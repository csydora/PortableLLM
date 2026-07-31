@echo off
setlocal

cd /d "%~dp0"

call .venv\Scripts\activate.bat

echo Cleaning...

rmdir /S /Q build 2>nul
rmdir /S /Q dist 2>nul
rmdir /S /Q __pycache__ 2>nul

echo Building...

pyinstaller --clean PortableLLM.spec

if errorlevel 1 (
    pause
    exit /b
)

mkdir build

xcopy dist\PortableLLM build\ /E /I /Y

copy config.json build\

xcopy portablellm\models build\portablellm\models\ /E /I /Y
xcopy portablellm\prompts build\portablellm\prompts\ /E /I /Y
xcopy portablellm\embedding_models build\portablellm\embedding_models\ /E /I /Y

echo.
echo Build complete.
pause