# Rescue fine-tune: load mario_run1 checkpoint and continue training with
# death_penalty=50 to escape the x~2900 local optimum where Mario keeps
# falling into a pit because there was no negative reward for dying.
$ErrorActionPreference = "Stop"

$src = "logs\mario_run1"
$dst = "logs\mario_run3_rescue"

if (-not (Test-Path "$src\latest.pt")) {
    Write-Host "ERROR: $src\latest.pt not found." -ForegroundColor Red
    exit 1
}

Write-Host "==> Creating $dst and copying checkpoint..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "$dst\train_eps" | Out-Null
New-Item -ItemType Directory -Force -Path "$dst\eval_eps" | Out-Null

Copy-Item "$src\latest.pt" "$dst\latest.pt" -Force
Write-Host "    latest.pt copied ($(((Get-Item "$dst\latest.pt").Length/1MB).ToString('F1')) MB)" -ForegroundColor Green

# Copy the most recent 60 train episodes so the dataset has warm experience.
# Stale rewards from these old episodes will be diluted by new on-policy episodes
# within ~30k steps, but they bootstrap the world-model train loop.
$recent = Get-ChildItem "$src\train_eps\*.npz" | Sort-Object LastWriteTime -Descending | Select-Object -First 60
foreach ($f in $recent) {
    Copy-Item $f.FullName "$dst\train_eps\" -Force
}
Write-Host "    Copied $($recent.Count) recent train episodes." -ForegroundColor Green

Write-Host "==> Launching mario_rescue (death_penalty=50, +entropy bump)..." -ForegroundColor Cyan
$env:PYTHONUNBUFFERED = "1"
$ErrorActionPreference = "Continue"
# Redirect stderr -> stdout so PowerShell does not treat python warnings as errors.
& "c:\Users\31660\mario-rl\.venv-gpu\Scripts\python.exe" dreamer.py --configs mario_rescue --logdir $dst 2>&1
