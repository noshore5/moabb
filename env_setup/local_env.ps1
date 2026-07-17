param(
    [string]$EnvName = "moabb-env-win"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$EnvPath = Join-Path $RepoRoot $EnvName
$Reqs = Join-Path $ScriptDir "local_env_requirements.txt"
$Activate = Join-Path $EnvPath "Scripts\Activate.ps1"

if (Test-Path $EnvPath) {
    if (-not (Test-Path $Activate)) {
        throw "$EnvPath exists but is not a venv (missing Scripts\Activate.ps1)"
    }
    Write-Host "Reusing existing env at $EnvPath"
} else {
    $Python = $null
    try {
        $Python = & py -3.11 -c "import sys; print(sys.executable)"
    } catch {
        $Python = $null
    }
    if (-not $Python) {
        throw "Python 3.11 not found. Install it and ensure 'py -3.11' works."
    }

    & $Python -m venv $EnvPath
    Write-Host "Created $EnvPath (Python 3.11)."
}

& $Activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e $RepoRoot

if (Test-Path $Reqs) {
    python -m pip install -r $Reqs
}

Write-Host "Packages installed into $EnvPath."
Write-Host "Activate with: $Activate"
