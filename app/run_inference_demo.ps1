param(
    [int]$Port = 8765,
    [switch]$OpenBrowser
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

if (Test-Path -LiteralPath (Join-Path $RepoRoot ".venv-torch-cuda\Scripts\python.exe")) {
    $Python = Join-Path $RepoRoot ".venv-torch-cuda\Scripts\python.exe"
} else {
    $Python = "python"
}

$env:PYTHONPATH = Join-Path $RepoRoot "src"
$Arguments = @("app\inference_dashboard.py", "--host", "127.0.0.1", "--port", "$Port")
if ($OpenBrowser) {
    $Arguments += "--open-browser"
}

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "BaFuse inference dashboard stopped with exit code $LASTEXITCODE"
}
