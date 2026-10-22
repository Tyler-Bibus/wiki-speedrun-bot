[CmdletBinding()]
param(
    [string]$KevPath,
    [ValidateRange(1, 65535)]
    [int]$Port = 8008,
    [string]$Checkpoint,
    [string]$Python
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
if (-not $KevPath) { $KevPath = Join-Path $projectRoot '.cache\kev' }
$configPath = Join-Path $projectRoot 'configs\baseline.toml'
$resolvedKevPath = Resolve-Path -LiteralPath $KevPath -ErrorAction SilentlyContinue
if (-not $resolvedKevPath -or -not (Test-Path -LiteralPath (Join-Path $resolvedKevPath.Path 'kev\serve.py'))) {
    throw "Kev source was not found at '$KevPath'. Run scripts/setup_kev.ps1 from the project root first."
}

if (-not $Python) {
    $Python = Join-Path $resolvedKevPath.Path '.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Prepared Python was not found at '$Python'. Run scripts/setup_kev.ps1 first, or pass -Python with the prepared environment's Python executable."
}
$Python = (Resolve-Path -LiteralPath $Python).Path

if (-not $Checkpoint) {
    if (-not (Test-Path -LiteralPath $configPath)) {
        throw "Config file '$configPath' was not found. Add checkpoint to configs/baseline.toml or pass -Checkpoint."
    }
    $Checkpoint = Get-Content -LiteralPath $configPath -Raw | & $Python -c "import sys, tomllib; print(tomllib.loads(sys.stdin.read())['checkpoint'])"
    if ($LASTEXITCODE -ne 0 -or -not $Checkpoint) {
        throw "Could not read checkpoint from '$configPath'. Fix the config or pass -Checkpoint."
    }
}

$hfCache = Join-Path $projectRoot '.cache\huggingface'
New-Item -ItemType Directory -Force -Path $hfCache | Out-Null
$env:HF_HOME = $hfCache
if (-not $env:KEV_FUSED) { $env:KEV_FUSED = '0' }
if (-not $env:KEV_CUDA_GRAPHS) { $env:KEV_CUDA_GRAPHS = '0' }

# This pinned release auto-selects CUDA; verify it first instead of silently
# allowing a CPU fallback. Newer upstream releases also have a --device flag.
& $Python (Join-Path $PSScriptRoot 'check_gpu.py')
if ($LASTEXITCODE -ne 0) { throw 'The CUDA runtime check failed. Run scripts/setup_kev.ps1 first.' }
Write-Host "Starting Kev checkpoint $Checkpoint on 127.0.0.1:$Port using CUDA."
Write-Host "Using prepared Python: $Python"
Write-Host "Hugging Face cache: $hfCache"
Push-Location $resolvedKevPath.Path
try {
    & $Python -m kev.serve --run $Checkpoint --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) { throw "Kev server exited with code $LASTEXITCODE." }
} finally {
    Pop-Location
}
