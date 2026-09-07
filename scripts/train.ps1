# Start or resume training on Windows.
#
#   .\scripts\train.ps1                      # stage 1 (fast config)
#   .\scripts\train.ps1 -Stage 2             # stage 2
#   .\scripts\train.ps1 -Workers 8           # override worker count
#
# Activates the venv, picks the right config, and -- the point of this script --
# resumes from last.pt automatically when one exists.  Losing a terminal should
# cost you nothing but the steps since the last checkpoint.

param(
    [int]$Stage = 1,
    [int]$Workers = 0,
    [string]$Config = "",
    [switch]$Fresh          # ignore an existing checkpoint and start over
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

# --- venv ---------------------------------------------------------------
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "No .venv found. Create it first:" -ForegroundColor Red
    Write-Host "    py -3.11 -m venv .venv"
    Write-Host "    .venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}
# Calling python.exe directly needs no Activate.ps1 and no execution policy
# change -- and, more usefully, it ignores whatever venv happens to be active.
# VS Code auto-activates any environment it finds on the machine, so a prompt
# reading (some_other_env) is common and would otherwise run a Python with none
# of this project's packages installed.
if ($env:VIRTUAL_ENV -and ($env:VIRTUAL_ENV -notlike "*$((Get-Location).Path)*")) {
    Write-Host "Note: '$env:VIRTUAL_ENV' is active, but this script uses the project's" -ForegroundColor Yellow
    Write-Host "      .venv regardless -- no need to deactivate anything.`n" -ForegroundColor Yellow
}

# --- config -------------------------------------------------------------
if ($Config -eq "") {
    $Config = switch ($Stage) {
        1 { "configs\a4000_stage1_fast.yaml" }
        2 { "configs\a4000_stage2_fast.yaml" }
        3 { "configs\stage3_finetune.yaml" }
        default { throw "unknown stage $Stage" }
    }
}
if (-not (Test-Path $Config)) { throw "config not found: $Config" }

$ckpt = if ($Stage -eq 3) { "checkpoints\stage3" } else { "checkpoints\a4000_stage$Stage" }

$argList = @("tools\train.py", "--config", $Config, "--ckpt-dir", $ckpt)

if ($Workers -gt 0) { $argList += @("--workers", "$Workers") }

# --- resume, or chain from the previous stage ---------------------------
$last = Join-Path $ckpt "last.pt"
if ((Test-Path $last) -and (-not $Fresh)) {
    $age = [math]::Round(((Get-Date) - (Get-Item $last).LastWriteTime).TotalMinutes, 1)
    Write-Host "Resuming from $last (written $age min ago)" -ForegroundColor Green
    $argList += @("--resume", $last)
} else {
    if ($Fresh -and (Test-Path $last)) {
        Write-Host "-Fresh given: ignoring $last and starting from scratch" -ForegroundColor Yellow
    } else {
        Write-Host "No checkpoint in $ckpt -- starting from scratch" -ForegroundColor Yellow
    }
    $prev = "checkpoints\a4000_stage$($Stage - 1)\last.pt"
    if ($Stage -gt 1 -and (Test-Path $prev)) {
        Write-Host "Initialising from the previous stage: $prev" -ForegroundColor Green
        $argList += @("--init", $prev)
    }
}

Write-Host "`n$py $($argList -join ' ')`n" -ForegroundColor Cyan
& $py $argList
