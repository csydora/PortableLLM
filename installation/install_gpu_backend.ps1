param(
    [ValidateSet("auto", "cuda", "hip", "vulkan", "cpu")]
    [string]$Backend = "auto"
)

# Native pip/Python stderr is intentionally captured below so backend failures
# can be reported and the installer can continue to the next candidate.
$ErrorActionPreference = "Continue"
$installDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $installDir
$null = Set-Location -LiteralPath $root
$python = Join-Path $root ".venv\Scripts\python.exe"
$configPath = Join-Path $root "config.json"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python environment not found: $python"
}

if ($Backend -eq "auto") {
    $configuredBackend = "auto"
    if (Test-Path -LiteralPath $configPath) {
        $projectConfig = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json -ErrorAction Stop
        if ($projectConfig.gpu_backend) {
            $configuredBackend = [string]$projectConfig.gpu_backend
        }
    }
    if ($configuredBackend -ne "auto") {
        $Backend = $configuredBackend.ToLowerInvariant()
    }
}

if ($Backend -eq "auto") {
    $adapterNames = @()
    try {
        $adapterNames = @(Get-CimInstance Win32_VideoController -ErrorAction Stop | ForEach-Object { $_.Name })
    }
    catch {
    }
    if (-not $adapterNames) {
        try {
            $adapterNames = @(Get-PnpDevice -Class Display -PresentOnly -ErrorAction Stop | ForEach-Object { $_.FriendlyName })
        }
        catch {
            Write-Warning "Could not query display adapters; use gpu_backend in config.json to select a backend manually."
        }
    }

    if ($adapterNames -match "NVIDIA") {
        $Backend = "cuda"
    }
    elseif ($adapterNames -match "AMD|Radeon") {
        $Backend = "hip"
    }
    else {
        $Backend = "cpu"
    }
}

$Backend = $Backend.ToLowerInvariant()
if ($Backend -notin @("cuda", "hip", "vulkan", "cpu")) {
    throw "Unsupported gpu_backend '$Backend'. Choose auto, cuda, hip, vulkan, or cpu."
}

function Install-SelectedBackend([string]$Name) {
    $requirements = Join-Path $installDir "requirements-$Name.txt"
    Write-Host "Installing llama.cpp backend: $Name"

    $output = & $python -m pip uninstall -y llama-cpp-python 2>&1
    $commandExitCode = $LASTEXITCODE
    $output | Out-Host
    if ($commandExitCode -ne 0) {
        return $false
    }

    $output = & $python -m pip install -r $requirements 2>&1
    $commandExitCode = $LASTEXITCODE
    $output | Out-Host
    if ($commandExitCode -ne 0) {
        Write-Warning "Could not install the $Name backend wheel."
        return $false
    }

    $output = & $python -c "import portablellm; import llama_cpp; print('llama-cpp-python', llama_cpp.__version__); supported = llama_cpp.llama_supports_gpu_offload(); print('GPU backend compiled:', supported); raise SystemExit(1 if '$Name' != 'cpu' and not supported else 0)" 2>&1
    $commandExitCode = $LASTEXITCODE
    $output | Out-Host
    if ($commandExitCode -ne 0) {
        Write-Warning "The $Name backend could not be imported or has no compiled GPU support."
        if ($Name -eq "hip") {
            Write-Warning "For HIP on Windows, check the AMD driver/runtime and libomp140.x86_64.dll. Trying the next backend."
        }
        return $false
    }

    return $true
}

$candidates = switch ($Backend) {
    "hip" { @("hip", "vulkan", "cpu") }
    "cuda" { @("cuda", "vulkan", "cpu") }
    "vulkan" { @("vulkan", "cpu") }
    default { @("cpu") }
}

foreach ($candidate in $candidates) {
    if (Install-SelectedBackend $candidate) {
        Write-Host "Installed backend: $candidate"
        exit 0
    }
}

throw "Could not install a usable llama.cpp backend. Check network access and the backend selection in config.json."
