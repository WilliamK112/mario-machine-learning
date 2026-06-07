# auto_diagnose.ps1
# -----------------
# Watch the latest PPO run for new periodic checkpoints; for each new
# checkpoint run a quick stochastic diagnostic. If Mario clears the flag
# (or even gets close, x > $TriggerX), fall back to a full multi-seed
# best-of-N render with GIF and write FLAG_CLEAR.txt as a tripwire.
#
# Designed to run alongside training (CPU-only inference, doesn't touch
# the GPU the trainer is using).
#
# Usage:
#   pwsh -ExecutionPolicy Bypass -File .\auto_diagnose.ps1
#   pwsh -ExecutionPolicy Bypass -File .\auto_diagnose.ps1 -PollEverySec 60 -QuickSeeds 3
#
# Stop with Ctrl+C, or it stops itself when:
#   - flag is cleared
#   - the training process exits

param(
    [int]$PollEverySec = 60,
    [int]$QuickSeeds = 3,
    [int]$QuickMaxSteps = 1500,
    [int]$FullSeeds = 5,
    [int]$FullMaxSteps = 4000,
    [int]$TriggerX = 2500,
    [string]$RunDir = ""
)

$ErrorActionPreference = "Stop"
# Repo root = directory containing this script (works for any clone path).
$projectRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Get-Location }
$python = Join-Path $projectRoot ".venv-gpu\Scripts\python.exe"
$renderer = Join-Path $projectRoot "render_best_checkpoint.py"

function Get-LatestPpoRunDir {
    $candidates = Get-ChildItem (Join-Path $projectRoot "runs") -Directory -Filter "ppo_*" `
        -ErrorAction SilentlyContinue
    if (-not $candidates) { return $null }
    $scored = foreach ($d in $candidates) {
        $m = [regex]::Match($d.Name, "(\d{8}_\d{6})$")
        if ($m.Success) {
            [PSCustomObject]@{ Dir = $d; Stamp = $m.Groups[1].Value }
        }
    }
    ($scored | Sort-Object Stamp -Descending | Select-Object -First 1).Dir
}

function Get-LatestCheckpoint($ckptDir) {
    if (-not (Test-Path $ckptDir)) { return $null }
    Get-ChildItem $ckptDir -Filter "mario_ppo_*_steps.zip" -ErrorAction SilentlyContinue |
        Sort-Object { [int](($_.BaseName -split "_")[2]) } -Descending |
        Select-Object -First 1
}

function Get-StepFromCheckpoint($ckpt) {
    [int](($ckpt.BaseName -split "_")[2])
}

function Test-TrainerAlive {
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*train_mario*' }
    return ($null -ne $procs)
}

function Invoke-Diagnostic {
    param([System.IO.FileInfo]$Checkpoint, [int]$Seeds, [int]$MaxSteps, [string]$Modes, [string]$OutDir)
    $oldEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $stdoutBuf = & $python $renderer `
            --model $Checkpoint.FullName `
            --seeds $Seeds --max-steps $MaxSteps `
            --policy-modes $Modes `
            --output-dir $OutDir 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) {
            throw "renderer exited with code $LASTEXITCODE`n$stdoutBuf"
        }
    } finally {
        $ErrorActionPreference = $oldEAP
    }
    return $stdoutBuf
}

function Parse-DiagnosticAllRuns($outDir) {
    $jsonPath = Join-Path $outDir "all_runs.json"
    if (-not (Test-Path $jsonPath)) { return $null }
    return Get-Content $jsonPath -Raw | ConvertFrom-Json
}

if (-not $RunDir) {
    $runDirObj = Get-LatestPpoRunDir
    if (-not $runDirObj) { throw "no runs/ppo_* directory found" }
    $RunDir = $runDirObj.FullName
}
$ckptDir = Join-Path $RunDir "run\models\checkpoints"
$diagRoot = Join-Path $RunDir "run\auto_diagnose"
$logPath = Join-Path $diagRoot "auto_diagnose.log"
$csvPath = Join-Path $diagRoot "auto_diagnose.csv"
$flagFlag = Join-Path $diagRoot "FLAG_CLEAR.txt"
New-Item -ItemType Directory -Path $diagRoot -Force | Out-Null

if (-not (Test-Path $csvPath)) {
    "ts,step,best_max_x,avg_max_x,flag_seeds,n_seeds,best_return" | Out-File $csvPath -Encoding utf8
}

function Write-Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $logPath -Value $line
}

Write-Log "=========================================================="
Write-Log "auto_diagnose started"
Write-Log "  run_dir         : $RunDir"
Write-Log "  poll_every_sec  : $PollEverySec"
Write-Log "  quick_seeds     : $QuickSeeds (stochastic, max_steps=$QuickMaxSteps)"
Write-Log "  full_seeds      : $FullSeeds (both modes, max_steps=$FullMaxSteps) when x > $TriggerX or flag"
Write-Log "  log             : $logPath"
Write-Log "  csv             : $csvPath"
Write-Log "=========================================================="

$seenSteps = New-Object System.Collections.Generic.HashSet[int]

while ($true) {
    if (Test-Path $flagFlag) {
        Write-Log "FLAG_CLEAR.txt exists -> already done, exiting"
        break
    }

    if (-not (Test-TrainerAlive)) {
        Write-Log "trainer process is gone -> running one final diagnostic on last checkpoint and exiting"
    }

    $ck = Get-LatestCheckpoint $ckptDir
    if ($null -eq $ck) {
        Write-Log "no checkpoints yet, sleeping ${PollEverySec}s"
        Start-Sleep -Seconds $PollEverySec
        continue
    }

    $step = Get-StepFromCheckpoint $ck
    if ($seenSteps.Contains($step)) {
        if (-not (Test-TrainerAlive)) { break }
        Start-Sleep -Seconds $PollEverySec
        continue
    }

    Write-Log "new checkpoint detected: step=$step  ($($ck.Name))  -> quick diagnostic ${QuickSeeds}-seed stochastic"

    $quickOutDir = Join-Path $diagRoot ("step_{0:D7}_quick" -f $step)
    try {
        $null = Invoke-Diagnostic -Checkpoint $ck -Seeds $QuickSeeds -MaxSteps $QuickMaxSteps -Modes "stochastic" -OutDir $quickOutDir
    } catch {
        Write-Log "ERROR running quick diagnostic: $_"
        $seenSteps.Add($step) | Out-Null
        Start-Sleep -Seconds $PollEverySec
        continue
    }

    $runs = Parse-DiagnosticAllRuns $quickOutDir
    if (-not $runs) {
        Write-Log "  -> no all_runs.json, skipping"
        $seenSteps.Add($step) | Out-Null
        Start-Sleep -Seconds $PollEverySec
        continue
    }

    $maxXs = $runs | ForEach-Object { $_.max_x }
    $bestX = ($maxXs | Measure-Object -Maximum).Maximum
    $avgX = [math]::Round((($maxXs | Measure-Object -Average).Average), 1)
    $flagCount = ($runs | Where-Object { $_.flag } | Measure-Object).Count
    $bestRet = [math]::Round((($runs | ForEach-Object { $_.return } | Measure-Object -Maximum).Maximum), 1)

    Write-Log ("  -> best_x={0}  avg_x={1}  flags={2}/{3}  best_return={4}" -f $bestX, $avgX, $flagCount, $runs.Count, $bestRet)
    "{0},{1},{2},{3},{4},{5},{6}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $step, $bestX, $avgX, $flagCount, $runs.Count, $bestRet | Out-File $csvPath -Append -Encoding utf8

    $shouldFullRender = ($flagCount -gt 0) -or ($bestX -ge $TriggerX)
    if ($shouldFullRender) {
        Write-Log "  -> threshold hit (flag=$flagCount, x=$bestX >= $TriggerX), running full ${FullSeeds}-seed render with GIF"
        $fullOutDir = Join-Path $diagRoot ("step_{0:D7}_full" -f $step)
        $oldEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $fullStdout = & $python $renderer `
                --model $ck.FullName `
                --seeds $FullSeeds --max-steps $FullMaxSteps `
                --policy-modes "stochastic,deterministic" `
                --gif `
                --output-dir $fullOutDir 2>&1 | Out-String
            $highlights = $fullStdout -split "`n" | Where-Object { $_ -match '(BEST|wrote)' }
            foreach ($h in $highlights) { Write-Log "    $h" }
            if ($LASTEXITCODE -ne 0) {
                Write-Log "  ERROR full render exit code $LASTEXITCODE"
            }
        } catch {
            Write-Log "  ERROR running full render: $_"
        } finally {
            $ErrorActionPreference = $oldEAP
        }

        $fullRuns = Parse-DiagnosticAllRuns $fullOutDir
        if ($fullRuns) {
            $fullFlag = ($fullRuns | Where-Object { $_.flag } | Measure-Object).Count
            $fullBestX = ($fullRuns | ForEach-Object { $_.max_x } | Measure-Object -Maximum).Maximum
            Write-Log "  -> full eval: flags=$fullFlag/$($fullRuns.Count)  best_x=$fullBestX"
            if ($fullFlag -gt 0) {
                $msg = "FLAG_CLEAR at step=$step  (full eval: $fullFlag/$($fullRuns.Count) seeds)"
                Write-Log "*** $msg ***"
                Write-Log "*** rendered to: $fullOutDir ***"
                Set-Content -Path $flagFlag -Value $msg -Encoding utf8
                # Audible/visual notification
                try { [console]::beep(880, 250); [console]::beep(1320, 400) } catch {}
                break
            }
        }
    }

    $seenSteps.Add($step) | Out-Null
    if (-not (Test-TrainerAlive)) {
        Write-Log "trainer exited after final diagnostic, stopping"
        break
    }
    Start-Sleep -Seconds $PollEverySec
}

Write-Log "auto_diagnose stopped"
