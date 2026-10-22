[CmdletBinding()]
param([string]$Python = 'python')

# Keep dependencies, upstream code, and model weights inside this repository.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$kevPath = Join-Path $projectRoot '.cache\kev'
$sourceRevision = '6b719c3c3f367295f6ef336f4f751cf5ff970abc'

Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        & $Python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Creating the project virtual environment failed.' }
    }
    $projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    & $projectPython -m pip install 'uv==0.13.4'
    if ($LASTEXITCODE -ne 0) { throw 'Installing uv failed.' }
    $uv = Join-Path $projectRoot '.venv\Scripts\uv.exe'
    $env:UV_CACHE_DIR = Join-Path $projectRoot '.cache\uv'
    & $uv sync --locked
    if ($LASTEXITCODE -ne 0) { throw 'Installing the Wikispeedia package failed.' }

    if (-not (Test-Path -LiteralPath $kevPath)) {
        & git clone --depth 1 --branch kev-1.0 https://github.com/jaredpalmer/kev.git $kevPath
        if ($LASTEXITCODE -ne 0) { throw 'Downloading the Kev source failed.' }
    }
    $revision = & git -C $kevPath rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $revision.Trim() -ne $sourceRevision) {
        throw "Expected Kev source revision $sourceRevision in $kevPath. Use the kev-1.0 release."
    }
    & $uv sync --locked --extra serve --no-dev --project $kevPath --python $projectPython
    if ($LASTEXITCODE -ne 0) { throw 'Installing Kev serving dependencies failed.' }

    # Upstream's lock selects CPU-only torch on Windows. Replace it explicitly;
    # launch the prepared Python directly afterward so uv does not undo this.
    $kevPython = Join-Path $kevPath '.venv\Scripts\python.exe'
    & $uv pip install --python $kevPython 'torch==2.8.0+cu128' --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'Installing CUDA PyTorch failed.' }
    & $kevPython scripts/check_gpu.py
    if ($LASTEXITCODE -ne 0) { throw 'The CUDA runtime check failed.' }
    Write-Host 'Setup complete. Run .\scripts\serve_kev.ps1 in a second terminal.'
} finally {
    Pop-Location
}
