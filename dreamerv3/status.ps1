# One-liner status checker. Run: .\status.ps1
$proc = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.WS -gt 500MB }
if (-not $proc) {
    Write-Host "[!] No rescue python process running" -ForegroundColor Red
    exit 1
}
Write-Host ("[+] PID {0} alive, CPU {1:F0}s, mem {2:F0}MB, started {3}" -f $proc.Id, $proc.CPU, ($proc.WS/1MB), $proc.StartTime) -ForegroundColor Green
$m = Get-Item c:\Users\31660\mario-rl\dreamerv3\logs\mario_run3_rescue\metrics.jsonl
$ageSec = [int]((Get-Date) - $m.LastWriteTime).TotalSeconds
$color = if ($ageSec -lt 60) { 'Green' } elseif ($ageSec -lt 300) { 'Yellow' } else { 'Red' }
Write-Host ("[+] metrics.jsonl last updated {0}s ago" -f $ageSec) -ForegroundColor $color
& c:\Users\31660\mario-rl\.venv-gpu\Scripts\python.exe c:\Users\31660\mario-rl\dreamerv3\check_rescue.py
