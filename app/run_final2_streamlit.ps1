param(
    [switch]$OpenBrowser
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv-torch-cuda\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python environment was not found: $Python"
}

Set-Location $RepoRoot
$env:PYTHONPATH = Join-Path $RepoRoot "src"
$Arguments = @(
    "-m", "streamlit", "run", "app\final2_streamlit_dashboard.py",
    "--server.address", "127.0.0.1",
    "--server.port", "8767",
    "--server.maxUploadSize", "1024",
    "--browser.gatherUsageStats", "false"
)
if ($OpenBrowser) {
    $Arguments += "--server.headless=false"
} else {
    $Arguments += "--server.headless=true"
}

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "BaFuse Final2 Streamlit dashboard stopped with exit code $LASTEXITCODE"
}
