# Resume rescue v5 after the previous process died silently at step ~24000.
# Uses direct invocation (no Tee-Object pipe) to avoid pipeline-related exits.
# Output goes to logs\rescue_resume.log via Python's unbuffered stdout.
$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"
$dst = "logs\mario_run3_rescue"
$log = "logs\rescue_resume.log"

Write-Host "==> Resuming from latest.pt (saved 1:14:35, step ~22000)" -ForegroundColor Cyan
Write-Host "    training data: $((Get-ChildItem $dst\train_eps\*.npz).Count) episodes"

# Direct call, stdout redirected to file. No tee pipe.
$cmd = "c:\Users\31660\mario-rl\.venv-gpu\Scripts\python.exe"
$dreamerArgs = @("dreamer.py", "--configs", "mario_rescue", "--logdir", $dst)
Start-Process -FilePath $cmd -ArgumentList $dreamerArgs -RedirectStandardOutput $log -RedirectStandardError "logs\rescue_resume.err" -NoNewWindow -PassThru | Select-Object Id, ProcessName
