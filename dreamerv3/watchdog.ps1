# Auto-restart watchdog for rescue training.
# Checks every 60s: if the dreamer python process died, relaunches it via resume_rescue.ps1.
# Run in a separate PowerShell window: .\watchdog.ps1
$ErrorActionPreference = "Continue"
Set-Location c:\Users\31660\mario-rl\dreamerv3

function Has-LiveDreamer {
    # A live dreamer python process has > 500 MB memory and is > 30s old.
    $procs = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
        $_.WS -gt 500MB -and ((Get-Date) - $_.StartTime).TotalSeconds -gt 30
    }
    return @($procs).Count -gt 0
}

Write-Host "==> Watchdog starting (checks every 60s)" -ForegroundColor Cyan
$restartCount = 0
while ($true) {
    if (Has-LiveDreamer) {
        $metricsAge = ((Get-Date) - (Get-Item logs\mario_run3_rescue\metrics.jsonl).LastWriteTime).TotalSeconds
        $stamp = Get-Date -Format "HH:mm:ss"
        Write-Host "[$stamp] alive (metrics ${metricsAge:F0}s old, restarts=$restartCount)" -ForegroundColor Green
    } else {
        $stamp = Get-Date -Format "HH:mm:ss"
        Write-Host "[$stamp] DEAD -- relaunching via resume_rescue.ps1" -ForegroundColor Red
        & .\resume_rescue.ps1
        $restartCount++
        Start-Sleep -Seconds 30  # let it boot before next check
    }
    Start-Sleep -Seconds 60
}
