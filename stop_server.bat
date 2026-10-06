@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "$root = [System.IO.Path]::GetFullPath('%~dp0');" ^
  "$uvicorn = Join-Path $root '.venv\Scripts\uvicorn.exe';" ^
  "$servers = Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -ieq $uvicorn -and $_.CommandLine -match 'portablellm\.main:app' };" ^
  "if (-not $servers) { Write-Output 'PortableLLM server not found.'; exit 0 };" ^
  "foreach ($server in $servers) { & taskkill.exe /PID $server.ProcessId /T /F }"
endlocal