# Rescue v2 launcher: load mario_run1 latest.pt, wipe old eps (already done),
# run with death_penalty=200 to break the x~2900 local optimum.
$ErrorActionPreference = "Continue"

$dst = "logs\mario_run3_rescue"

if (-not (Test-Path "$dst\latest.pt")) {
    Write-Host "ERROR: $dst\latest.pt not found. Rescue cannot continue." -ForegroundColor Red
    exit 1
}

$trainCount = (Get-ChildItem "$dst\train_eps\*.npz" -ErrorAction SilentlyContinue).Count
$evalCount  = (Get-ChildItem "$dst\eval_eps\*.npz"  -ErrorAction SilentlyContinue).Count
Write-Host "==> State:" -ForegroundColor Cyan
Write-Host "    latest.pt size: $(((Get-Item "$dst\latest.pt").Length/1MB).ToString('F1')) MB"
Write-Host "    existing train_eps: $trainCount"
Write-Host "    existing eval_eps:  $evalCount"

Write-Host "==> Launching mario_rescue v2 (death_penalty=200, wiped data, entropy=3e-3)..." -ForegroundColor Cyan
$env:PYTHONUNBUFFERED = "1"
& "c:\Users\31660\mario-rl\.venv-gpu\Scripts\python.exe" dreamer.py --configs mario_rescue --logdir $dst 2>&1
