<#
  sweep_runner.ps1 - auto-restarting launcher for the AI-judge sweep.

  Start it once from anywhere; it cd's to its own folder (the repo root) first:
      .\sweep_runner.ps1

  Behaviour:
   * Runs sweep_autojudge.py with the settings in the CONFIG block below.
   * Restarts ONLY on a crash (non-zero exit) - transient rate-limit / network.
     A clean finish (exit 0) means the deterministic ladder completed; it stops,
     because re-running the same settings just reproduces the same result.
   * Caps restarts at $MaxAttempts with $BackoffSec between tries, so a real
     (deterministic) bug can't loop forever.
   * Tees each attempt to .\sweep_logs\attempt_<stamp>_<n>.log for review.

  To push toward the MAE target, change the CONFIG block between rounds
  (Claude will tell you exactly what to edit) and relaunch.
#>

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot   # always run from the repo root, so ../ paths + the car_judge import resolve

# ============================ CONFIG (edit between rounds) ====================
$Data         = "../car_ratings_long_July8_July22_merged_with_color_and_Q23.csv"
$ImageRoot    = "../selected_2000_isometric_upload_chunks_renamed"
$Raters       = 20
$TestSize     = 20
$TargetMae    = "0.0"      # 0.0 = UNREACHABLE on purpose: never early-stop, keep
                           #   pushing for the lowest MAE until $MaxExp is spent.
$MaxExp       = 5          # ROUND 2 = e0 anchor + the four context sizes, and STOP.
                           #   The round-1 randomized phase was shown to be noise:
                           #   its winner (r6) ranked LAST on both fresh splits, and
                           #   no pairwise difference among the top 3 was significant.
                           #   Context size is the only lever with a real effect.
$ContextSizes = "12 16 20 24"  # ROUND 2 probes PAST the round-1 edge: live MAE was
                           #   still falling at N=12 (1.108/0.999/0.969/0.938 for
                           #   N=0/4/8/12), so the optimum may lie beyond 12.
                           #   NOTE: cost/latency scale with N - each call ships N+1 images.
$KeepGoing    = $false     # NO randomized phase - round 1 proved it only finds the
                           #   luckiest draw from a ~0.02-wide noise band.
$MaxAttempts  = 5          # crash-restarts before giving up
$BackoffSec   = 15
# =============================================================================

# Pin the interpreter: bare `python` here resolves to a standalone Python312, not
# the venv the working sweep ran under.
$Py = Join-Path $PSScriptRoot "car_ai_judges\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "venv interpreter not found at $Py" }

$logDir = Join-Path $PSScriptRoot "sweep_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$argsList = @(
    "sweep_autojudge.py",
    "--data", $Data,
    "--image-root", $ImageRoot,
    "--raters", $Raters,
    "--test-size", $TestSize,
    "--target-mae", $TargetMae,
    "--max-experiments", $MaxExp
)
if ($ContextSizes -ne "") { $argsList += "--context-sizes"; $argsList += $ContextSizes.Split(" ") }
if ($KeepGoing)           { $argsList += "--keep-going" }

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
Write-Host "=== sweep_runner $stamp ===" -ForegroundColor Cyan
Write-Host ("cwd    : {0}" -f (Get-Location))
Write-Host ("command: python {0}" -f ($argsList -join ' '))

$success = $false
for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    $log = Join-Path $logDir ("attempt_{0}_{1}.log" -f $stamp, $attempt)
    Write-Host ("`n--- attempt {0}/{1}  ({2}) ---" -f $attempt, $MaxAttempts, (Get-Date -Format o)) -ForegroundColor Yellow
    & $Py @argsList 2>&1 | Tee-Object -FilePath $log
    $code = $LASTEXITCODE
    if ($code -eq 0) {
        Write-Host ("attempt {0} succeeded (exit 0)." -f $attempt) -ForegroundColor Green
        $success = $true
        break
    }
    Write-Host ("attempt {0} FAILED (exit {1}). log: {2}" -f $attempt, $code, $log) -ForegroundColor Red
    if ($attempt -lt $MaxAttempts) {
        Write-Host ("restarting in {0}s..." -f $BackoffSec)
        Start-Sleep -Seconds $BackoffSec
    }
}

if (-not $success) {
    Write-Host ("`nGAVE UP after {0} attempts - likely a deterministic error, not a blip. Check {1}." -f $MaxAttempts, $logDir) -ForegroundColor Red
    exit 1
}
Write-Host ("`nDONE. results.tsv updated; logs in {0}." -f $logDir) -ForegroundColor Green
