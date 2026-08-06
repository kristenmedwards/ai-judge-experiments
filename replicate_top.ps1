<#
  replicate_top.ps1 - does the sweep's "winner" survive a different data split?

  The 30-experiment live sweep produced configs spanning only ~0.02 MAE, which is
  inside the measurement noise, and the winner (r6) was assembled almost entirely
  from levers the ladder had individually rejected. That is the signature of the
  luckiest draw, not of a better design.

  This re-measures the top 3 configs on split seeds 1 and 2 (the sweep used 0).
  Read it as:
    * r6 wins again on both fresh splits -> the interaction is real, adopt it.
    * the ranking reshuffles             -> the ~0.02 spread is noise, and
                                            e1_ctx12 (simplest: N=12 random,
                                            plain prompt) is the honest choice.

  Cost: 3 configs x 2 seeds x 400 calls = ~2,400 calls (~1h). Logs to sweep_logs\.

  NOTE: keep this file ASCII-only. Windows PowerShell 5.1 reads a BOM-less .ps1
  as ANSI, so a non-ASCII character inside a string literal is a parse error.
#>

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$Data      = "../car_ratings_long_July8_July22_merged_with_color_and_Q23.csv"
$ImageRoot = "../selected_2000_isometric_upload_chunks_renamed"
$Raters    = 20
$TestSize  = 20
$Seeds     = @(1, 2)
$Configs   = @(
    @{ Name = "r6";         Path = "best_judge_config.json" },   # sweep winner, 0.924
    @{ Name = "e5_persona"; Path = "configs/e5_persona.json" },  # ladder winner, 0.933
    @{ Name = "e1_ctx12";   Path = "configs/e1_ctx12.json" }     # simplest strong, 0.938
)

# Pin the interpreter: bare `python` on this box resolves to a standalone Python312
# that is NOT the venv the sweep ran under, and its openai package is unverified.
$Py = Join-Path $PSScriptRoot "car_ai_judges\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "venv interpreter not found at $Py" }

$logDir = Join-Path $PSScriptRoot "sweep_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log   = Join-Path $logDir ("replicate_{0}.log" -f $stamp)

Write-Host ("=== replicate_top {0} ===" -f $stamp) -ForegroundColor Cyan
$failed = 0
foreach ($seed in $Seeds) {
    foreach ($c in $Configs) {
        $tag = "{0}_seed{1}" -f $c.Name, $seed
        Write-Host ("`n--- {0} ---" -f $tag) -ForegroundColor Yellow
        $a = @(
            "run_autojudge.py",
            "--data", $Data, "--image-root", $ImageRoot,
            "--config", $c.Path,
            "--raters", $Raters, "--test-size", $TestSize,
            "--split-seed", $seed,
            "--out", ("outputs/replication/{0}.csv" -f $tag),
            "--note", ("REPLICATION seed={0} config={1}" -f $seed, $c.Name)
        )
        & $Py @a 2>&1 | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) {
            Write-Host ("{0} FAILED (exit {1})" -f $tag, $LASTEXITCODE) -ForegroundColor Red
            $failed++
        }
    }
}

if ($failed -gt 0) {
    Write-Host ("`n{0} run(s) failed - see {1}" -f $failed, $log) -ForegroundColor Red
    exit 1
}
Write-Host ("`nDONE. {0} runs appended to results.tsv (note=REPLICATION); log {1}" -f ($Seeds.Count * $Configs.Count), $log) -ForegroundColor Green
