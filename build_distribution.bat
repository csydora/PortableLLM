@echo off
setlocal
cd /d "%~dp0"

echo Building PortableLLM.zip...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'Stop'; try {" ^
  "$root = [System.IO.Path]::GetFullPath('%~dp0').TrimEnd('\');" ^
  "$zip = Join-Path $root 'PortableLLM.zip';" ^
  "$stage = Join-Path $env:TEMP ('PortableLLM-package-' + [guid]::NewGuid().ToString('N'));" ^
  "New-Item -ItemType Directory -Path $stage | Out-Null;" ^
  "$files = @(Get-ChildItem -LiteralPath $root -Force -File -Recurse | Where-Object { $_.FullName -notmatch '[\\/](?:\.git|\.venv|build|dist|logs|__pycache__|models--sentence-transformers--all-MiniLM-L6-v2)[\\/]' -and $_.Extension -ine '.gguf' -and $_.Extension -ine '.pyc' -and $_.Extension -ine '.pyo' -and $_.Name -ine 'PortableLLM.zip' -and $_.Name -ine '.server.json' -and $_.Name -notlike '*.log' });" ^
  "foreach ($file in $files) { $relative = $file.FullName.Substring($root.Length + 1); $destination = Join-Path $stage $relative; $parent = Split-Path -Parent $destination; New-Item -ItemType Directory -Force -Path $parent | Out-Null; Copy-Item -LiteralPath $file.FullName -Destination $destination; }" ^
  "Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -CompressionLevel Optimal -Force;" ^
  "Write-Host ('Created: ' + $zip); Write-Host ('Files packaged: ' + $files.Count);" ^
  "} catch { Write-Error $_; exit 1 } finally { if ($stage -and (Test-Path -LiteralPath $stage)) { Remove-Item -LiteralPath $stage -Recurse -Force } }"

if errorlevel 1 (
    echo.
    echo Distribution build failed.
    pause
    exit /b 1
)

echo.
echo Distribution build completed.
pause
endlocal
