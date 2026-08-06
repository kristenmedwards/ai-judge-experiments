<#
  search_150.ps1 - guide-experiment search queue at --raters 150, seeds 0 and 1.

  Crash-resilient version:
   * $ErrorActionPreference stays "Continue": with "Stop", PS 5.1 turns the FIRST
     stderr line of a native command under 2>&1 into a terminating error - that is
     exactly how v1 of this queue died mid-run (and ate the traceback).
   * Each run gets up to 2 attempts (transient API bursts).
   * Runs whose tag already appears in results.tsv are SKIPPED, so re-launching
     after a crash resumes instead of re-spending.
   * --max-retries 6 per API call inside the run.

  ASCII only (PS 5.1 reads BOM-less .ps1 as ANSI).
#>

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

$Data      = "../car_ratings_long_July8_July22_merged_with_color_and_Q23.csv"
$ImageRoot = "../selected_2000_isometric_upload_chunks_renamed"
$Raters    = 150
$TestSize  = 20
$Conc      = 8
$Pairs     = @(
    @{C = "b_persona";       S = 0}, @{C = "g_card";          S = 0},
    @{C = "g_twostage";      S = 0}, @{C = "g_card_twostage"; S = 0},
    @{C = "b_persona";       S = 1}, @{C = "g_card";          S = 1},
    @{C = "g_twostage";      S = 1}, @{C = "g_card_twostage"; S = 1}
)

$Py = Join-Path $PSScriptRoot "car_ai_judges\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "venv interpreter not found at $Py" }

$logDir = Join-Path $PSScriptRoot "sweep_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log   = Join-Path $logDir ("search150_{0}.log" -f $stamp)

$done = @()
if (Test-Path "results.tsv") {
    $done = Select-String -Path "results.tsv" -Pattern "search150 seed=(\d) config=(\S+)" -AllMatches |
        ForEach-Object { $_.Matches } |
        ForEach-Object { "{0}_s{1}" -f $_.Groups[2].Value, $_.Groups[1].Value }
}

$failed = 0
foreach ($p in $Pairs) {
    $tag = "{0}_s{1}" -f $p.C, $p.S
    if ($done -contains $tag) {
        Write-Host ("skip {0} (already in results.tsv)" -f $tag) -ForegroundColor DarkGray
        continue
    }
    $ok = $false
    foreach ($try in 1, 2) {
        Write-Host ("`n--- {0} attempt {1} ({2}) ---" -f $tag, $try, (Get-Date -Format o)) -ForegroundColor Yellow
        $a = @(
            "run_autojudge.py",
            "--data", $Data, "--image-root", $ImageRoot,
            "--config", ("configs/{0}.json" -f $p.C),
            "--raters", $Raters, "--test-size", $TestSize,
            "--split-seed", $p.S, "--concurrency", $Conc,
            "--max-retries", 6,
            "--run-tag", $tag,
            "--note", ("search150 seed={0} config={1}" -f $p.S, $p.C)
        )
        & $Py @a 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -eq 0) { $ok = $true; break }
        Write-Host ("{0} attempt {1} FAILED (exit {2})" -f $tag, $try, $LASTEXITCODE) -ForegroundColor Red
        Start-Sleep -Seconds 30
    }
    if (-not $ok) { $failed++ }
}

if ($failed -gt 0) { Write-Host ("{0} run(s) failed - see {1}" -f $failed, $log) -ForegroundColor Red; exit 1 }
Write-Host "DONE - all queue runs complete; results in results.tsv and outputs/runs/" -ForegroundColor Green
