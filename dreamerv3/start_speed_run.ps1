# Stage-2 fine-tune: add time bonus on top of mario_run1 checkpoint.
# Usage: .\start_speed_run.ps1
#
# This script:
#   1. Copies latest.pt (and a small subset of train_eps) from mario_run1 into mario_run2_speed
#   2. Launches dreamer with the mario_speed config (time_bonus_scale=0.5)
#   3. Dreamer auto-loads latest.pt because it sits in the new logdir

$ErrorActionPreference = "Stop"

$src = "logs\mario_run1"
$dst = "logs\mario_run2_speed"

if (-not (Test-Path "$src\latest.pt")) {
    Write-Host "ERROR: $src\latest.pt not found. Is mario_run1 still running?" -ForegroundColor Red
    exit 1
}

Write-Host "==> Creating $dst and copying checkpoint..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "$dst\train_eps" | Out-Null
New-Item -ItemType Directory -Force -Path "$dst\eval_eps" | Out-Null

# Copy model checkpoint (contains world model + actor + critic + optim state).
Copy-Item "$src\latest.pt" "$dst\latest.pt" -Force
Write-Host "    latest.pt copied." -ForegroundColor Green

# Also copy recent train episodes so dataset has something to sample from immediately.
# (We only keep the newer half so stale-reward episodes don't dominate for long.)
$recent = Get-ChildItem "$src\train_eps\*.npz" | Sort-Object LastWriteTime -Descending | Select-Object -First 40
foreach ($f in $recent) {
    Copy-Item $f.FullName "$dst\train_eps\" -Force
}
Write-Host "    Copied $($recent.Count) recent train episodes for warmup dataset." -ForegroundColor Green

Write-Host "==> Launching fine-tune run (mario_speed)..." -ForegroundColor Cyan
$env:PYTHONUNBUFFERED = "1"
& "c:\Users\31660\mario-rl\.venv-gpu\Scripts\python.exe" dreamer.py `
    --configs mario_speed `
    --logdir $dst
