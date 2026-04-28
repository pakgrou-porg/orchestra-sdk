#Requires -Version 5.1
<#
.SYNOPSIS
    Orchestra SDK — Windows PowerShell launcher

.DESCRIPTION
    Activates the virtual environment, loads .env, and passes all arguments
    to the orchestra CLI.

.EXAMPLE
    .\deploy\orchestra.ps1 run --config conductor_config.yaml
    .\deploy\orchestra.ps1 status --config conductor_config.yaml --all
    .\deploy\orchestra.ps1 migrate --config conductor_config.yaml
    .\deploy\orchestra.ps1 check
    .\deploy\orchestra.ps1 setup

.NOTES
    If you see "execution policy" errors, run once as Administrator:
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#>

param(
    [Parameter(Position = 0)]
    [string]$Command = "",

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot  = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvDir   = Join-Path $RepoRoot ".venv"
$EnvFile   = Join-Path $RepoRoot ".env"
$Python    = Join-Path $VenvDir "Scripts\python.exe"
$Orchestra = Join-Path $VenvDir "Scripts\orchestra.exe"
$Activate  = Join-Path $VenvDir "Scripts\Activate.ps1"

# ── Activate virtual environment ─────────────────────────────────────────────
if (-not (Test-Path $Activate)) {
    Write-Error "Virtual environment not found at $VenvDir`nRun: python deploy\setup.py"
    exit 1
}
& $Activate

# ── Load .env ────────────────────────────────────────────────────────────────
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line -match "^([^=]+)=(.*)$") {
            $key   = $Matches[1].Trim()
            $value = $Matches[2].Trim()
            [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
} else {
    Write-Warning ".env not found at $EnvFile`nRun: python deploy\setup.py  or  copy deploy\.env.example .env"
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
if (-not $Command) {
    Write-Host ""
    Write-Host "Orchestra SDK launcher" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\deploy\orchestra.ps1 <command> [options]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  setup                   Run the interactive setup wizard"
    Write-Host "  check                   Run the health-check validator"
    Write-Host "  run     --config FILE   Start a Conductor session"
    Write-Host "  status  --config FILE   Show session status"
    Write-Host "  migrate --config FILE   Apply database migrations"
    Write-Host "  inspect --config FILE   Inspect session memories and git log"
    Write-Host "  reset   --config FILE   Revert workspace to a previous iteration"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\deploy\orchestra.ps1 run --config conductor_config.yaml"
    Write-Host "  .\deploy\orchestra.ps1 status --config conductor_config.yaml --all"
    exit 0
}

switch ($Command.ToLower()) {
    "setup" {
        & $Python (Join-Path $PSScriptRoot "setup.py") @Rest
    }
    "check" {
        & $Python (Join-Path $PSScriptRoot "check.py") --env $EnvFile @Rest
    }
    default {
        & $Orchestra $Command --env $EnvFile @Rest
    }
}

exit $LASTEXITCODE
