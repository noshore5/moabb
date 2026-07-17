param(
    [string]$EnvName = "<name>"
)

$ErrorActionPreference = "Stop"

python -m venv $EnvName
& "$EnvName\Scripts\Activate.ps1"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .

if (Test-Path "requirements.txt") {
    python -m pip install -r requirements.txt
}