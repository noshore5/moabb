# Local env setup

Python **3.11** required. Run from anywhere; venv is created at the repo root.
Idempotent: if the target env already exists it is reused (never deleted /
recreated). Pip install of the local repo and `local_env_requirements.txt` is
re-run; already-satisfied packages are left alone.

## Windows (PowerShell)

```powershell
.\env_setup\local_env.ps1
.\moabb-env-win\Scripts\Activate.ps1
```

Optional custom name: `.\env_setup\local_env.ps1 my-env`

## Unix

```bash
bash env_setup/local_env.sh
source moabb-env/bin/activate
```

Optional custom name: `bash env_setup/local_env.sh my-env`

## What gets installed

1. Editable MOABB (`pip install -e .`)
2. Extras from `local_env_requirements.txt`
