param(
    [string]$Repo = "C:\Users\31660\mario-rl",
    [int]$PollSeconds = 120,
    [int]$SegmentTimesteps = 5000000
)

$ErrorActionPreference = "Continue"
Set-Location $Repo

$Python = Join-Path $Repo ".venv-gpu\Scripts\python.exe"
$RunsDir = Join-Path $Repo "runs"

function Write-Status([string]$Message) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$stamp] $Message"
}

function Get-MarioTrainingProcess {
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -like "*train_mario.py*" -and
            $_.CommandLine -like "*--whole-game*" -and
            $_.CommandLine -like "*mario_all_levels*"
        }
}

function Read-JsonFile([string]$Path) {
    try {
        return Get-Content $Path -Raw | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Get-StageIndex($Summary) {
    if ($null -ne $Summary.best_world_stage_index) {
        return [int]$Summary.best_world_stage_index
    }
    if ($null -ne $Summary.best_world_stage -and $Summary.best_world_stage.Count -ge 2) {
        return (([int]$Summary.best_world_stage[0] - 1) * 4) + [int]$Summary.best_world_stage[1]
    }
    return 0
}

function Test-ProgressBetter($Candidate, $CurrentBest) {
    if ($null -eq $CurrentBest) { return $true }

    if ($Candidate.Index -ne $CurrentBest.Index) {
        return $Candidate.Index -gt $CurrentBest.Index
    }
    if ($Candidate.Remaining -ne $CurrentBest.Remaining) {
        return $Candidate.Remaining -lt $CurrentBest.Remaining
    }
    if ($Candidate.MaxX -ne $CurrentBest.MaxX) {
        return $Candidate.MaxX -gt $CurrentBest.MaxX
    }
    if ($Candidate.Clears -ne $CurrentBest.Clears) {
        return $Candidate.Clears -gt $CurrentBest.Clears
    }
    if ($Candidate.Steps -ne $CurrentBest.Steps) {
        return $Candidate.Steps -gt $CurrentBest.Steps
    }
    return $Candidate.LastWriteTicks -gt $CurrentBest.LastWriteTicks
}

function Get-BestProgress {
    $best = $null
    $summaryFiles = Get-ChildItem $RunsDir -Directory -Filter "mario_all_levels*" -ErrorAction SilentlyContinue |
        ForEach-Object {
            Get-ChildItem $_.FullName -Recurse -Filter "summary.json" -ErrorAction SilentlyContinue
        }

    foreach ($file in $summaryFiles) {
        $summary = Read-JsonFile $file.FullName
        if ($null -eq $summary) { continue }
        $idx = Get-StageIndex $summary
        $clears = if ($null -ne $summary.best_stage_clears) { [double]$summary.best_stage_clears } else { 0.0 }
        $remaining = if ($null -ne $summary.best_remaining_distance) { [double]$summary.best_remaining_distance } else { [double]::PositiveInfinity }
        $maxX = if ($null -ne $summary.best_max_x) { [double]$summary.best_max_x } else { 0.0 }
        $steps = if ($null -ne $summary.num_timesteps) { [int64]$summary.num_timesteps } else { 0 }
        $candidate = [pscustomobject]@{
            Summary = $summary
            Path = $file.FullName
            Index = [int]$idx
            Clears = [double]$clears
            Remaining = [double]$remaining
            MaxX = [double]$maxX
            Steps = [int64]$steps
            LastWriteTicks = [int64]$file.LastWriteTimeUtc.Ticks
        }
        if (Test-ProgressBetter $candidate $best) {
            $best = $candidate
        }
    }
    return $best
}

function Test-FinalReached {
    $best = Get-BestProgress
    if ($null -eq $best) { return $false }
    return ($best.Index -ge 32 -and $best.Clears -ge 31)
}

function Get-ResumeModel {
    $patterns = @(
        (Join-Path $RunsDir "mario_all_levels*\models\checkpoints\mario_ppo_*_steps.zip"),
        (Join-Path $RunsDir "mario_all_levels*\models\mario_final.zip"),
        (Join-Path $RunsDir "mario_all_levels*\evaluations\best_eval.zip"),
        (Join-Path $Repo "pretrained\ppo_mario_1-1\best_eval.zip")
    )

    foreach ($pattern in $patterns) {
        $candidates = @(Get-ChildItem $pattern -File -ErrorAction SilentlyContinue)
        if ($candidates.Count -gt 0) {
            return $candidates | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
        }
    }

    return $null
}

function Start-TrainingSegment {
    $resume = Get-ResumeModel
    if ($null -eq $resume) {
        Write-Status "No resume model found. Sleeping before retry."
        Start-Sleep -Seconds $PollSeconds
        return
    }

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $runDir = Join-Path $RunsDir "mario_all_levels_auto_$timestamp"
    $logDir = Join-Path $runDir "logs"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $stdout = Join-Path $logDir "train_stdout.log"
    $commandFile = Join-Path $logDir "command.txt"

    $args = @(
        "train_mario.py",
        "--run-dir", $runDir,
        "--resume-model", $resume.FullName,
        "--whole-game", "--no-end-on-flag",
        "--timesteps", "$SegmentTimesteps",
        "--device", "cuda",
        "--action-set", "simple_pipe",
        "--n-envs", "8",
        "--gamma", "0.99", "--gae-lambda", "0.95",
        "--learning-rate", "2.0e-4",
        "--n-steps", "512", "--batch-size", "256",
        "--ent-coef", "0.02", "--n-epochs", "4", "--clip-range", "0.18",
        "--normalize-reward", "--norm-reward-clip", "10",
        "--forward-reward-scale", "0.10",
        "--backward-penalty-scale", "0.02",
        "--flag-bonus", "10", "--stage-clear-bonus", "3000",
        "--progress-reward-mode", "new_max",
        "--stall-steps", "45", "--stall-penalty", "0.15",
        "--end-on-stall-steps", "240", "--end-on-stall-penalty", "50",
        "--cage-escape-shaping",
        "--milestone-step", "50", "--milestone-bonus", "0.3",
        "--time-penalty-per-step", "0.01",
        "--action-bias-jump", "0.0",
        "--action-bias-down", "0.0",
        "--action-bias-noop", "-0.25",
        "--action-bias-left", "-0.45",
        "--skip-action-bias-on-resume",
        "--rnd-coef", "0.10",
        "--eval-freq", "50000", "--eval-episodes", "3", "--eval-steps", "10000",
        "--preview-freq", "5000", "--preview-steps", "2500", "--preview-episodes", "1",
        "--preview-fps", "15", "--no-preview-video",
        "--checkpoint-freq", "50000"
    )

    $command = "`"$Python`" " + ($args -join " ")
    $command | Set-Content -Path $commandFile -Encoding UTF8

    Write-Status "Starting next Mario all-level segment."
    Write-Status "Run dir: $runDir"
    Write-Status "Resume: $($resume.FullName)"
    Write-Status "Command: $command"

    & $Python @args 2>&1 | Tee-Object -FilePath $stdout -Append
    $exitCode = $LASTEXITCODE
    Write-Status "Training segment exited with code $exitCode."
}

Write-Status "Mario all-level watchdog started."
Write-Status "It will not stop training unless eval progress reaches world-stage index 32 with at least 31 stage clears."

while ($true) {
    if (Test-FinalReached) {
        $best = Get-BestProgress
        Write-Status "Final target reached. Best summary: $($best.Path)"
        break
    }

    $procs = @(Get-MarioTrainingProcess)
    if ($procs.Count -gt 0) {
        $best = Get-BestProgress
        if ($null -ne $best) {
            Write-Status "Training alive (PIDs: $($procs.ProcessId -join ', ')). Best index=$($best.Index), remaining=$($best.Remaining), max_x=$($best.MaxX), clears=$($best.Clears), steps=$($best.Steps)."
        } else {
            Write-Status "Training alive (PIDs: $($procs.ProcessId -join ', ')). No eval summaries yet."
        }
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    Write-Status "No Mario training process found. Starting or resuming training."
    Start-TrainingSegment
    Start-Sleep -Seconds 10
}
