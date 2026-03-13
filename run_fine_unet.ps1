# ======================
# Run fine_unet.py
# ======================

# Set error handling
$ErrorActionPreference = "Stop"

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   fine_unet.py Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 1. Check virtual environment
Write-Host "`n[1/4] Checking virtual environment..." -ForegroundColor Yellow
$venvPath = Join-Path $scriptDir ".venv"

if (-Not (Test-Path $venvPath)) {
    Write-Host "[!] Virtual environment not found: $venvPath" -ForegroundColor Red
    Write-Host "[*] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[X] Failed to create virtual environment!" -ForegroundColor Red
        exit 1
    }
}
Write-Host "[OK] Virtual environment ready" -ForegroundColor Green

# 2. Activate virtual environment
Write-Host "`n[2/4] Activating virtual environment..." -ForegroundColor Yellow
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"

if (-Not (Test-Path $activateScript)) {
    Write-Host "[X] Activation script not found: $activateScript" -ForegroundColor Red
    exit 1
}

# Call activation script
& $activateScript
Write-Host "[OK] Virtual environment activated" -ForegroundColor Green

# 4. Run script
Write-Host "`n[4/4] Starting fine_unet.py..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Cyan

$pythonScript = Join-Path $scriptDir "PythonScripts\tests\fine_unet.py"
$pythonScriptsRoot = Join-Path $scriptDir "PythonScripts"

if (-Not (Test-Path $pythonScript)) {
    Write-Host "[X] Script not found: $pythonScript" -ForegroundColor Red
    exit 1
}

if (-Not (Test-Path $pythonScriptsRoot)) {
    Write-Host "[X] PythonScripts root not found: $pythonScriptsRoot" -ForegroundColor Red
    exit 1
}

# Ensure project imports like `from utils...` and `from rebuild...` can be resolved.
if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
    $env:PYTHONPATH = $pythonScriptsRoot
} else {
    $env:PYTHONPATH = "$pythonScriptsRoot;$env:PYTHONPATH"
}

# Run Python script from project root so config.ini relative paths remain valid
python $pythonScript

# Check exit status
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n----------------------------------------" -ForegroundColor Cyan
    Write-Host "[OK] Script executed successfully!" -ForegroundColor Green
} else {
    Write-Host "`n----------------------------------------" -ForegroundColor Cyan
    Write-Host "[X] Script execution failed (exit code: $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "========================================" -ForegroundColor Cyan
